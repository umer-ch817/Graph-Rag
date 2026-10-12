# GraphRAG Lab

A local research application for comparing vector retrieval with
evidence-backed graph retrieval on multi-hop QA. The core app is complete
through T15 at fixture scale: deterministic sampling, sentence-preserving
chunks, vector search, fixed-schema extraction with conservative resolution,
bounded hybrid retrieval with reranker ablations (A–D), frozen evaluation
scaffolding, durable jobs, and a Streamlit dashboard with guarded exports.

Locked 120-question multi-hop evaluation and human-verified 30-question
single-hop controls were not executed; `docs/RESULTS.md` reports only
measured fixture outcomes. Work proceeds one ticket at a time; see
[tickets](tasks/todo.md) and the [planning package](docs/README.md).

## Quickstart (clean environment)

Use **Python 3.12** and Docker Desktop with Linux containers. Run commands
from this folder.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

If the Python launcher does not recognize `-3.12`, use the full path to a
Python 3.12 interpreter. The lock is tested on Windows x64 / Python 3.12;
other platforms require verification before claiming support.

## Start Neo4j

Start Docker Desktop. Copy `.env.example` to `.env` **only if `.env` does
not exist**, and fill in a unique password of at least eight characters.
The real `.env` is ignored. Preserve the existing password because Neo4j
stores authentication in its data volume.

```powershell
docker compose config --quiet
docker compose up -d --wait --wait-timeout 180 neo4j
docker compose ps
```

Neo4j Browser: <http://127.0.0.1:7474>. Bolt: `bolt://127.0.0.1:7687`.
Username: `neo4j`. Ports bind only to loopback. Compose pins a
registry-verified Community image digest and persists the database in a
named volume. Changing the `.env` password does not change an existing
database password.

Stop without deleting data:

```powershell
docker compose stop neo4j
```

Do not use `docker compose down --volumes` unless you intend to delete the
database. Do not publish `docker compose config` output without `--quiet`,
because expanded configuration contains the local password.

## Configuration

`configs/default.yaml` contains non-secret settings. Process environment
variables prefixed `GRAPHRAG_` override YAML; nested fields use `__`, for
example `GRAPHRAG_BUDGET__MAX_CALLS=10`. Python deliberately does not
auto-load `.env`; Docker Compose loads it for the local database. Export
credentials explicitly in the Python process environment when needed.

Supported secret environment variables: `GRAPHRAG_NEO4J_PASSWORD` and
`GRAPHRAG_LLM_API_KEY`. They are excluded from configuration dumps. Do not
put secrets in YAML. The LLM model is unset by default, so paid calls are
not ready. A call or token cap is mandatory even when a monetary cap
exists.

```powershell
.\.venv\Scripts\python.exe -c "from graphrag_lab.config import load_config; print(load_config('configs/default.yaml').model_dump_json(indent=2))"
.\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py -q
.\.venv\Scripts\python.exe -m ruff check .
```

Update dependency locks intentionally, after editing `pyproject.toml`:

```powershell
uv pip compile pyproject.toml --extra dev --generate-hashes --python .venv/Scripts/python.exe --output-file requirements-dev.lock
```

The lock update command requires uv; ordinary installation from the lock
uses pip and does not require uv. Keep the lockfile in version control.
Never commit `.env`, virtual environments, generated artifacts or
credentials. [Environment record](docs/ENVIRONMENT.md) captures the tested
hardware.

## Fixture demo (no downloads, no paid calls)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_full_pipeline.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q src app
```

The fixture flow prepares a synthetic 8-question source twice
(byte-identical), chunks with provenance checks, answers with
`FakeLlmBackend`, scores with a fixed denominator (failures stay in),
compares compatible arms with bootstrap intervals, exports guarded
artifacts with unknown costs labeled, persists jobs across restarts, and
exercises the dashboard view-models and control freeze gates. Full steps
and the clean-environment checklist live in [docs/DEMO.md](docs/DEMO.md).

Run the dashboard after Neo4j is healthy:

```powershell
$env:GRAPHRAG_NEO4J_PASSWORD = "<password-from-.env>"
.\.venv\Scripts\python.exe -m graphrag_lab.cli worker --once
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Walk Overview → Dataset & Index → Ask & Compare → Evidence → Evaluations.
The worker claims SQLite jobs independently of Streamlit reruns; the UI
enqueues work and polls persisted status.

## Results and limitations

[docs/RESULTS.md](docs/RESULTS.md) is the honest handoff report: models,
hardware, pooled-corpus caveats, unknown costs, and negative results. Key
points:

- Defaults: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine),
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, LLM unset, budgets 20 calls /
  20k tokens, final top-8 / 3000 tokens, graph caps 25/100/40/2s.
- Corpus design is a pooled 150-question HotpotQA distractor sample (30
  tuning / 120 locked) plus 30 draft single-hop controls — not an official
  benchmark split. `data/raw/` is empty in this handoff, so no real sample
  was built.
- Citation validity means resolution to packed evidence, not entailment.
  Token budgets use the conservative `char-word-max-v1` estimator with
  model-tokenizer re-validation.
- No frozen 120+30 evaluation was run and controls lack human review, so no
  hybrid uplift (or regression) is claimed. Resume wording uses only the
  measured fixture outcomes quoted in `docs/RESULTS.md`.

## Architecture

Streamlit dashboard + CLI share typed Python services. Neo4j owns chunks,
entities, and relations; SQLite owns jobs/runs; JSONL artifacts own corpus,
extraction, and evaluation payloads with content hashes. One local worker
executes jobs; only one active index build is supported. See
[TRD](docs/TRD.md), [app flow](docs/APP-FLOW.md), and
[backend schema](docs/BACKEND-SCHEMA.md).

Configuration and container setup follow [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) and [Neo4j Docker Compose guidance](https://neo4j.com/docs/operations-manual/current/docker/docker-compose-standalone/).
