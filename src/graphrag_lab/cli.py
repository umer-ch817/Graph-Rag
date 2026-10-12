"""Local CLI. Prepare, durable jobs/worker, and vector indexing (T02/T12a/T12c)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag_lab.config import load_config
from graphrag_lab.corpus import ensure_source, prepare_dataset
from graphrag_lab.models import HOTPOT_URL, SampleConfig

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_SOURCE = "data/raw/hotpot_dev_distractor_v1.json"
DEFAULT_OUTPUT = "artifacts/dataset"
DEFAULT_DB = "artifacts/jobs.sqlite3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphrag-lab", description="Reproducible GraphRAG experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Sample HotpotQA and write paragraph artifacts")
    prep.add_argument("--config", default=DEFAULT_CONFIG)
    prep.add_argument("--source", default=DEFAULT_SOURCE)
    prep.add_argument("--output", default=DEFAULT_OUTPUT)
    prep.add_argument("--source-url", default=HOTPOT_URL)
    prep.add_argument("--sample-size", type=int, default=None)
    prep.add_argument("--tuning-size", type=int, default=None)
    prep.add_argument("--seed", type=int, default=None)
    prep.add_argument(
        "--no-download", action="store_true", help="Fail if the source file is missing"
    )

    work = sub.add_parser("worker", help="Claim and execute queued SQLite jobs")
    work.add_argument("--db", default=DEFAULT_DB)
    work.add_argument("--worker-id", default=None)
    work.add_argument("--once", action="store_true", help="Execute one job then exit")
    work.add_argument("--max-jobs", type=int, default=None)
    work.add_argument("--lease-timeout-s", type=float, default=30.0)
    work.add_argument("--poll-interval-s", type=float, default=1.0)
    work.add_argument("--kinds", default=None, help="Comma-separated job kinds to claim")
    work.add_argument("--config", default=DEFAULT_CONFIG)

    demo = sub.add_parser("enqueue-demo", help="Enqueue a demo job with N items")
    demo.add_argument("--db", default=DEFAULT_DB)
    demo.add_argument("--kind", default="demo")
    demo.add_argument("--count", type=int, default=5)
    demo.add_argument("--prefix", default="item")
    demo.add_argument("--idempotency-key", default=None)
    demo.add_argument("--job-id", default=None)

    index = sub.add_parser("index", help="Enqueue a vector/graph index job")
    index.add_argument("--stage", required=True, choices=["vector", "graph"])
    index.add_argument("--snapshot-id", required=True)
    index.add_argument("--config", default=DEFAULT_CONFIG)
    index.add_argument("--db", default=DEFAULT_DB)
    index.add_argument("--idempotency-key", default=None)
    index.add_argument("--dataset-dir", default=DEFAULT_OUTPUT)

    evaluate = sub.add_parser("evaluate", help="Enqueue one evaluation arm")
    evaluate.add_argument("--strategy", required=True)
    evaluate.add_argument("--split", default="tuning")
    evaluate.add_argument("--snapshot-id", default=None)
    evaluate.add_argument("--db", default=DEFAULT_DB)
    evaluate.add_argument("--idempotency-key", default=None)
    evaluate.add_argument("--question-ids", default=None, help="Comma-separated question ids")

    get = sub.add_parser("job-get", help="Show one job with progress/error/retryability")
    get.add_argument("--db", default=DEFAULT_DB)
    get.add_argument("--job-id", required=True)

    lst = sub.add_parser("job-list", help="List jobs by state/kind")
    lst.add_argument("--db", default=DEFAULT_DB)
    lst.add_argument("--state", default=None)
    lst.add_argument("--kind", default=None)
    lst.add_argument("--limit", type=int, default=20)

    cancel = sub.add_parser("job-cancel", help="Request cancellation (honored at item boundary)")
    cancel.add_argument("--db", default=DEFAULT_DB)
    cancel.add_argument("--job-id", required=True)

    resume = sub.add_parser("job-resume", help="Resume interrupted/cancelled/failed/paused jobs")
    resume.add_argument("--db", default=DEFAULT_DB)
    resume.add_argument("--job-id", required=True)
    return parser


def cmd_prepare(args: argparse.Namespace) -> int:
    load_config(args.config)  # validate local configuration early; secrets stay in env
    source = Path(args.source)
    if not source.is_file():
        if args.no_download:
            raise SystemExit(f"source file not found: {source}")
        ensure_source(source, args.source_url)
    sample = SampleConfig(
        sample_size=args.sample_size if args.sample_size is not None else 150,
        tuning_size=args.tuning_size if args.tuning_size is not None else 30,
        seed=args.seed if args.seed is not None else 42,
    )
    manifest = prepare_dataset(source, Path(args.output), sample, source_url=args.source_url)
    print(json.dumps(manifest.model_dump(), indent=2, sort_keys=True))
    return 0


def _parse_kinds(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    kinds = [part.strip() for part in raw.split(",") if part.strip()]
    return kinds or None


def cmd_worker(args: argparse.Namespace) -> int:
    from graphrag_lab.index_tasks import make_index_vector_handler
    from graphrag_lab.jobs import JobStore
    from graphrag_lab.worker import Worker, demo_handler_factory

    if args.lease_timeout_s <= 0 or args.poll_interval_s <= 0:
        raise SystemExit("lease-timeout-s and poll-interval-s must be positive")
    config = load_config(args.config)
    store = JobStore(args.db)  # ensure schema before worker starts
    worker = Worker(args.db, worker_id=args.worker_id, lease_timeout_s=float(args.lease_timeout_s))
    worker.register_handler("demo", demo_handler_factory())
    worker.register_handler(
        "index_vector",
        make_index_vector_handler(store, embedding_model=config.embedding_model),
    )
    kinds = _parse_kinds(args.kinds)
    recovered = worker.recover_stale()
    if recovered:
        print(json.dumps({"recovered": recovered}, sort_keys=True))
    if args.once:
        job_id = worker.run_once(kinds=kinds)
        print(json.dumps({"executed": job_id, "worker_id": worker.worker_id}, sort_keys=True))
        return 0
    done = worker.run_forever(
        kinds=kinds,
        poll_interval_s=float(args.poll_interval_s),
        max_jobs=args.max_jobs,
    )
    print(json.dumps({"completed_jobs": done, "worker_id": worker.worker_id}, sort_keys=True))
    return 0


def cmd_enqueue_demo(args: argparse.Namespace) -> int:
    from graphrag_lab.jobs import JobStore

    if args.count < 0:
        raise SystemExit("count must be non-negative")
    store = JobStore(args.db)
    items = [{"id": f"{args.prefix}_{i:02d}"} for i in range(int(args.count))]
    ref = store.enqueue_job(
        args.kind,
        {"items": items},
        idempotency_key=args.idempotency_key,
        job_id=args.job_id,
    )
    print(json.dumps(ref.model_dump(), indent=2, sort_keys=True))
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from graphrag_lab.jobs import JobStore, enqueue_index

    config = load_config(args.config)
    store = JobStore(args.db)
    items: list[dict[str, str]] | None = None
    if args.stage == "vector":
        # One control item: the handler builds chunks → embeddings → Neo4j
        # with batched progress inside a single durable job attempt.
        items = [{"id": "build", "dataset_dir": args.dataset_dir}]
    ref = enqueue_index(
        store,
        snapshot_id=args.snapshot_id,
        stage=args.stage,
        budget=config.budget,
        idempotency_key=args.idempotency_key,
        items=items,
    )
    print(json.dumps(ref.model_dump(), indent=2, sort_keys=True))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from graphrag_lab.jobs import JobStore, enqueue_evaluation

    store = JobStore(args.db)
    question_ids = (
        [part.strip() for part in args.question_ids.split(",") if part.strip()]
        if args.question_ids
        else []
    )
    items = [{"id": qid} for qid in question_ids]
    ref = enqueue_evaluation(
        store,
        snapshot_id=args.snapshot_id,
        strategy=args.strategy,
        split=args.split,
        config={},
        items=items,
        idempotency_key=args.idempotency_key,
    )
    print(json.dumps(ref.model_dump(), indent=2, sort_keys=True))
    return 0


def cmd_job_get(args: argparse.Namespace) -> int:
    from graphrag_lab.jobs import JobStore, get_job_status

    store = JobStore(args.db)
    print(json.dumps(get_job_status(store, args.job_id), indent=2, sort_keys=True))
    return 0


def cmd_job_list(args: argparse.Namespace) -> int:
    from graphrag_lab.jobs import JobStore

    store = JobStore(args.db)
    jobs = store.list_jobs(state=args.state, kind=args.kind, limit=int(args.limit))
    print(json.dumps([j.model_dump() for j in jobs], indent=2, sort_keys=True))
    return 0


def cmd_job_cancel(args: argparse.Namespace) -> int:
    from graphrag_lab.jobs import JobStore

    store = JobStore(args.db)
    job = store.cancel_job(args.job_id)
    print(json.dumps(job.model_dump(), indent=2, sort_keys=True))
    return 0


def cmd_job_resume(args: argparse.Namespace) -> int:
    from graphrag_lab.jobs import JobStore

    store = JobStore(args.db)
    job = store.resume_job(args.job_id)
    print(json.dumps(job.model_dump(), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "worker":
        return cmd_worker(args)
    if args.command == "enqueue-demo":
        return cmd_enqueue_demo(args)
    if args.command == "index":
        return cmd_index(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    if args.command == "job-get":
        return cmd_job_get(args)
    if args.command == "job-list":
        return cmd_job_list(args)
    if args.command == "job-cancel":
        return cmd_job_cancel(args)
    if args.command == "job-resume":
        return cmd_job_resume(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
