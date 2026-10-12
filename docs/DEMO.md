# Demo and setup guide — GraphRAG Lab (T15)

Clean-environment path from clone to a fixture-scale run. All commands run
from the repository root on Windows PowerShell. Linux/macOS equivalents use
`python3.12 -m venv .venv` and `.venv/bin/python`.

## 1. Prerequisites

- Python 3.12 (project pins `>=3.12,<3.13`; tested on CPython 3.12.14).
- Docker Desktop with Linux containers (tested Docker 29.8.0).
- `uv` only for regenerating the hash lock; ordinary installs use `pip`.
- No HotpotQA download, model download, or LLM credentials are needed for
  the offline fixture demo. Paid steps are explicitly marked and skipped by
  default.

## 2. Fresh setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

Copy `.env.example` to `.env` only if `.env` does not exist, then set a
unique local password of at least eight characters. The real `.env` is
ignored by Git and is read by Docker Compose only; Python reads process
environment variables, never `.env` automatically.

```powershell
docker compose config --quiet
docker compose up -d --wait --wait-timeout 180 neo4j
docker compose ps
```

Neo4j Browser: http://127.0.0.1:7474. Bolt: `bolt://127.0.0.1:7687`.
Username `neo4j`. Ports bind to loopback only. Do not run
`docker compose down --volumes` unless you intend to delete the database.

Validate configuration (no secrets printed):

```powershell
.\.venv\Scripts\python.exe -c "from graphrag_lab.config import load_config; print(load_config('configs/default.yaml').model_dump_json(indent=2))"
.\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py -q
.\.venv\Scripts\python.exe -m ruff check .
```

## 3. Offline fixture demo (no downloads, no paid calls)

This is the T15 acceptance flow. It uses a synthetic 8-question source,
`FakeLlmBackend`, and `FakeRerankerBackend`; nothing reaches the network.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/e2e/test_full_pipeline.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q src app
```

What the fixture flow proves:

1. `prepare_dataset` twice produces byte-identical artifacts and the same
   corpus id; `questions.jsonl` carries no answers or supporting facts.
2. `chunk_paragraphs` preserves sentence provenance; every gold fact
   resolves to a stored sentence and to at least one chunk.
3. `answer_question` with `FakeLlmBackend` produces supported answers with
   resolving citations; empty evidence abstains without an LLM call.
4. `evaluate_predictions` keeps the fixed denominator (failures score zero);
   `compare_runs` produces paired deltas with 95% bootstrap intervals only
   for compatible runs and refuses unlabeled deltas otherwise.
5. `export_baseline` / `export_comparison` write guarded atomic artifacts
   (`results.jsonl`, `aggregate.json`, `manifest.json`, `summary.md` /
   `comparison.json`, `regression.csv`, `report.md`) with content hashes;
   unknown money is labeled `unknown`, never `$0`.
6. SQLite jobs survive restart; duplicate Streamlit submissions reuse the
   same active job; `summarize_setup` walks no-corpus → queued → running →
   ready without losing checkpoints.
7. `compare_flow` citations resolve to exact source sentences; the tuning
   picker excludes locked examples; matched budgets are enforced.
8. Single-hop controls stay gated: `require_verified=True` refuses the
   shipped draft (`human_verified=false`) until independent review.

## 4. Full local app (requires Neo4j + optional paid LLM)

Start the worker in one terminal (it claims SQLite jobs independently of
Streamlit reruns):

```powershell
$env:GRAPHRAG_NEO4J_PASSWORD = "<password-from-.env>"
.\.venv\Scripts\python.exe -m graphrag_lab.cli worker --once
# or long-running:
.\.venv\Scripts\python.exe -m graphrag_lab.cli worker
```

Prepare the real 150-question sample (downloads HotpotQA once to
`data/raw/`; skipped in the offline demo):

```powershell
.\.venv\Scripts\python.exe -m graphrag_lab.cli prepare --config configs/default.yaml
```

Enqueue a vector build, then run the worker to execute it:

```powershell
.\.venv\Scripts\python.exe -m graphrag_lab.cli index --stage vector --snapshot-id snap_local
.\.venv\Scripts\python.exe -m graphrag_lab.cli worker --once
.\.venv\Scripts\python.exe -m graphrag_lab.cli job-get --job-id <job-id>
```

Launch the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Walk Overview → Dataset & Index → Ask & Compare → Evidence → Evaluations.
Overview shows one state-dependent next action. Dataset & Index has four
guided sections with one progress panel each. Ask & Compare submits through
an explicit form; examples edit the draft only. Evidence opens exact cited
sentences; the relation table is keyboard-operable. Evaluations reads saved
runs only and never reruns a question implicitly.

## 5. Paid steps (not run in T15)

- Graph extraction pilot (20 chunks) and full build require
  `GRAPHRAG_LLM_MODEL`, `GRAPHRAG_LLM_API_KEY`, and a call/token cap in
  `configs/default.yaml` (`budget.max_calls`, `budget.max_total_tokens`;
  `max_cost_usd` is optional and requires pricing metadata).
- Frozen evaluation on 120 multi-hop + 30 single-hop controls requires the
  dataset, both indexes, human-verified controls, and an evaluation budget.
  See `docs/RESULTS.md` for why these were not executed in T15.
- Never commit `.env`, `.venv`, `artifacts/`, or credentials.

## 6. Demo checklist (clean-environment validation)

- [ ] Fresh `venv` installs from `requirements-dev.lock` with hashes.
- [ ] `docker compose config --quiet` passes; `neo4j` is healthy on loopback.
- [ ] `pytest tests/unit/test_config.py -q` passes (31 tests).
- [ ] `pytest tests/e2e/test_full_pipeline.py -q` passes (9 tests, offline).
- [ ] `pytest -q` passes (full suite, Neo4j integration included when up).
- [ ] `ruff check .` passes; `compileall -q src app` passes.
- [ ] Overview renders with zero exceptions; Dataset page shows four guided
      sections; Ask & Compare keeps question/settings after completion;
      Evidence restores prior run on return; Evaluations shows
      "No evaluation results yet" before any run.
- [ ] `docs/RESULTS.md` limitations match observed behavior (no invented
      scores, unknown costs labeled, single-hop draft flagged).
- [ ] Keyboard-only walk (submit, strategy select, citation open, return,
      evaluation filter, export) works; narrow-window layout stacks without
      horizontal prose overflow.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `source file not found` on `prepare` | Pass `--no-download` to fail fast, or allow one download to `data/raw/` |
| `GRAPHRAG_NEO4J_PASSWORD is not set` | Export it in the Python/worker process environment |
| `Neo4j unavailable ... Queued work is kept` | `docker compose up -d --wait neo4j`, then resume the job |
| `Worker has not reported yet` | Start `python -m graphrag_lab.cli worker` in another terminal |
| `BUDGET_EXCEEDED` / `paused_budget` | Revise `budget` caps, then `job-resume`; completed calls are kept |
| `Incompatible runs ... No unlabeled paired delta` | Compare only arms sharing corpus, index, question set, models, budgets |
| `lack human review ... refusing frozen use` | Expected until `data/controls/single_hop.jsonl` is audited |
| `Existing artifact ... differs` | Choose a new output directory; exports never overwrite silently |
