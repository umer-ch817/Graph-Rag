# Implementation tickets — GraphRAG Lab

T01 is complete; all other tickets remain unstarted. Commands and file names for unstarted tickets are planned interfaces, not implemented behavior. Run focused tests per ticket; run the full existing suite at milestone checkpoints. Core scope T01–T15; optional extension T16–T17.

## One-ticket-at-a-time workflow

The user assigns one ticket by id, for example: **Complete T01**. Implement only that ticket and the verification necessary to finish it. Do not automatically start the next ticket. Preserve these ticket ids across future sessions.

For each assignment:

1. Read the assigned ticket, relevant specifications and current repository rules. Check that dependencies are complete.
2. Mark the ticket In progress in its execution record. If a prerequisite is missing, identify it rather than silently implementing another ticket.
3. Implement the scoped deliverable and run its acceptance checks. File lists are estimates; necessary supporting files are allowed, but unrelated features are not.
4. Run any checkpoint attached to that ticket. Checkpoints C1–C5 belong to T03, T06, T09, T12c and T15 respectively; they are included in those assignments.
5. Record actual files changed, verification commands/results, limitations and any required user setup beneath the ticket. Update relevant specifications if implementation details change.
6. Mark Done and check the implementation box only when acceptance criteria and required verification pass. Otherwise record Blocked or In progress with the exact remaining work; passing mocked tests does not replace required live verification.
7. Report the result and next eligible ticket, then stop and wait for the user's next assignment.

Default state: **Not started** for every unchecked ticket with no execution record. Status values: Not started, In progress, Blocked, Done. The checkbox and execution record under each ticket are the authoritative progress record; the index below does not duplicate status.

Paid/live calls require configured credentials, model and budget. Never include secret values in a ticket record. Human review of the single-hop controls is part of T13 and must not be marked complete by the agent alone.

Completion record template:

```text
Status: Done / In progress / Blocked
Implemented: concrete deliverable
Files changed: actual paths
Verification: commands and results, including live checks when required
Remaining: none, or exact unmet acceptance criteria
Next eligible ticket: id (not started automatically)
```

## Ordered ticket index

There are **19 individually assignable tickets: 17 core and 2 optional**. T12 was already split into three smaller tickets, so its suffixes are intentional.

| Order | Ticket | Deliverable | Required inputs beyond prior tickets |
|---|---|---|---|
| 1 | T01 | Local environment, Docker and validated configuration | Local Python/Docker availability |
| 2 | T02 | Reproducible HotpotQA sample and splits | Dataset download access |
| 3 | T03 | Sentence-preserving chunks and provenance | None |
| 4 | T04 | Embeddings, Neo4j vector index and retrieval | Model download access |
| 5 | T05 | Grounded answers with validated citations | LLM provider/model, credentials and smoke-run budget |
| 6 | T06 | Baseline metrics and result artifacts | None |
| 7 | T07 | Structured extraction, caching and pilot audit | Extraction spend/token cap |
| 8 | T08 | Conservative entity resolution | Pilot extraction output |
| 9 | T09 | Evidence-backed graph loading | None |
| 10 | T10 | Entity linking and bounded hybrid retrieval | None |
| 11 | T11 | Cross-encoder reranking and fair experiment arms | Reranker model download access |
| 12 | T12a | Persistent jobs, worker and recovery | None |
| 13 | T12b | Dataset/index setup dashboard | None |
| 14 | T12c | Answer comparison and evidence screens | None |
| 15 | T13 | Frozen multi-hop and single-hop evaluation | Human control review and evaluation budget |
| 16 | T14 | Evaluation dashboard and report exports | Completed run artifacts |
| 17 | T15 | Full verification, setup guide and portfolio handoff | Clean-environment validation |
| 18 | T16 | Optional bounded question decomposition | Separate assignment after core completion |
| 19 | T17 | Optional decomposition evaluation | Extension evaluation budget |

The order above is the recommended sequential workflow. Detailed dependencies below remain authoritative if a ticket is requested out of order. No application implementation is authorized merely by creating this list.

## T01 — Local environment and configuration [corpus]

Execution record:
- Status: Done (2026-09-29).
- Implemented: isolated Python 3.12.14 package; hash-locked foundation dependencies; Pydantic/YAML configuration with environment overrides, validated model ids and budgets, local-only Neo4j URI and excluded credentials; digest-pinned Neo4j Community Compose service with loopback ports and authenticated health check; local Git initialization and secret/artifact exclusions.
- Files changed: `pyproject.toml`, `requirements-dev.lock`, `compose.yaml`, `.gitignore`, `.env.example`, `.streamlit/config.toml`, `configs/default.yaml`, `src/graphrag_lab/__init__.py`, `src/graphrag_lab/config.py`, `tests/unit/test_config.py`, `README.md`, `docs/ENVIRONMENT.md`, `docs/README.md`, `tasks/plan.md`, `tasks/todo.md`.
- Verification: test collection failed for missing `graphrag_lab.config` before implementation; 31 tests pass after implementation, including in a fresh environment installed from the hash lock. Ruff lint and format checks pass; compileall and pip check pass. Compose validates and reports healthy; authenticated Python driver query returns 1 from Neo4j/5.26.31. Git exclusions verified for `.env`, `.venv` and generated artifacts.
- Environment: Intel i5-10210U, 4 cores / 8 threads, approximately 7.79 GiB RAM, Intel UHD Graphics, Docker 29.8.0. Details and image digests recorded in `docs/ENVIRONMENT.md`.
- Remaining: none for T01. Foundation dependencies only; model downloads, provider selection and paid-call enforcement belong to later tickets. Neo4j is left running locally. No commits or remote repository created.
- Next eligible ticket: T02 (not started).

- [x] Implement.
- Acceptance: dependency lock and Neo4j image digest recorded; configuration validates model ids and budgets; services bind to loopback and secrets are excluded from version control.
- Verify: Docker health check, `python -m pytest tests/unit/test_config.py -q`, installation smoke check; record hardware.
- Depends on: none.
- Files (M): `pyproject.toml`, lockfile, `compose.yaml`, `src/graphrag_lab/config.py`, `tests/unit/test_config.py`.
- Estimate: 45–60 minutes. Include ignore rules as a small setup follow-up if absent; do not omit secret protection to satisfy file-count guidance.

## T02 — Deterministic HotpotQA sample [corpus]

Execution record:
- Status: Done (2026-09-29, clean rebuild; prior partial GPT work deleted).
- Implemented: deterministic HotpotQA sampling/selection, paragraph artifacts with variant preservation and label separation, reproducibility manifest, and `prepare` CLI with download cache.
- Files changed: `src/graphrag_lab/models.py`, `src/graphrag_lab/corpus.py`, `src/graphrag_lab/cli.py` (rebuilt from scratch; `__pycache__` removed). `tests/unit/test_corpus.py` retained as the acceptance oracle.
- Verification: `python -m pytest tests/unit/test_corpus.py -q` → 14 passed; `python -m pytest tests/unit -q` → 45 passed; `ruff check .` passed; `compileall -q src` passed; `pip check` clean. CLI smoke on 160-row synthetic source with 150/30/42: two `prepare` runs produced byte-identical artifacts, 150 selected ids, 30 tuning / 120 evaluation, 450 contexts (150×3), no answer/supporting_facts leakage into `questions.jsonl`.
- Remaining: none for T02. No chunking or retrieval implementation. Real HotpotQA download not exercised (no network fixture in scope).
- Note (2026-09-29, live data follow-up): `prepare_dataset` now quarantines upstream-invalid rows (ids + truncated reasons in new `DatasetManifest.quarantined_question_ids/reasons`; raw bytes untouched, sample from valid rows only, `source_question_count` = raw total). Driven by the real dev release: 1/7405 rows quarantined (`5ae61bfd5542992663a4f261`, supporting sentence 902 of 5 — genuine upstream noise, not mirror corruption). `tests/unit/test_corpus.py` → 15 passed incl. new quarantine test; all-valid inputs behave identically.
- Next eligible ticket: T03 (not started).

- [x] Implement.
- Acceptance: 150 ids selected reproducibly; 30/120 split persisted; all context paragraphs including distractors retained and labels separated.
- Verify: `python -m pytest tests/unit/test_corpus.py -q`; rerun preparation and compare artifact hashes.
- Depends on: T01.
- Files (M): `src/graphrag_lab/corpus.py`, `src/graphrag_lab/models.py`, `src/graphrag_lab/cli.py`, `tests/unit/test_corpus.py`.
- Estimate: 45–60 minutes.

## T03 — Sentence-preserving chunks [corpus]

Execution record:
- Status: Done (2026-09-29).
- Implemented: sentence-preserving chunker (one paragraph per chunk by default; long paragraphs split at sentence boundaries with one-sentence overlap; deterministic ids from corpus/paragraph/span/version; offline conservative token estimator with explicit overflow errors) and provenance verifier (chunk round-trip, gold-to-sentence resolution, gold-to-chunk coverage).
- Files changed: `src/graphrag_lab/chunking.py`, `src/graphrag_lab/provenance.py`, `tests/unit/test_chunking.py`, `tests/fixtures/corpus.json`, `tasks/todo.md`.
- Verification: `python -m pytest tests/unit/test_chunking.py -q` → 15 passed (overlap, long-sentence overflow, duplicate-title, variant, determinism, gold-coverage, dataset round-trip); `python -m pytest tests/unit -q` → 60 passed; `ruff check .` passed; `ruff format --check` clean for new files; `compileall -q src` passed; `pip check` clean. C1 smoke on 160-row synthetic source with 150/30/42: two `prepare_dataset` runs byte-identical, 150 ids, 153 paragraphs (Town 2 variants preserved), all 150 labels resolve and are covered by 153 chunks.
- Remaining: none for T03. Token estimator is English-tuned offline upper bound (`char-word-max-v1`); T04 must re-validate with the pinned embedding-model tokenizer and refuse mismatches. No retrieval, embedding, or CLI wiring.
- Note (2026-09-29, live data follow-up): `_require_paragraph_fields` now tolerates upstream empty-sentence strings in place (indices never shift; paragraph still needs ≥1 nonempty sentence) — 2/1499 real paragraphs affected, no gold fact cites them. `tests/unit/test_chunking.py` extended (empty-sentence stability + all-empty rejection); provenance contiguity and gold coverage hold on the real 150-sample (2000 chunks).
- Next eligible ticket: T04 (not started).

- [x] Implement.
- Acceptance: source title/sentence provenance round-trips; paragraph variants survive deduplication; no silent embedding-token overflow.
- Verify: `python -m pytest tests/unit/test_chunking.py -q` with overlap, long sentence, duplicate title and variant fixtures.
- Depends on: T02.
- Files (M): `src/graphrag_lab/chunking.py`, `src/graphrag_lab/provenance.py`, `tests/unit/test_chunking.py`, `tests/fixtures/corpus.json`.
- Estimate: 45–60 minutes.

### Checkpoint C1

- [x] Identical preparation produces identical ids; all selected gold facts resolve to stored source sentences; dataset manifest is reviewable.

## T04 — Vector index and retrieval [vector-qa]

Execution record:
- Status: Done (2026-09-29).
- Implemented: title+text embeddings with L2 normalization and finite/dim/zero validation (dim read from loaded model, 384; max_seq 256); model-tokenizer re-validation refusing overflow; Neo4j cosine vector index with ONLINE gating, single-corpus fingerprint (corpus/model/dim/chunker) refusing mismatches, idempotent MERGE ingestion in batches; vector retrieval (30 candidates, deterministic score/id order, foreign-corpus refusal).
- Files changed: `src/graphrag_lab/embeddings.py`, `src/graphrag_lab/neo4j_store.py`, `src/graphrag_lab/retrieval.py`, `tests/integration/test_vector_store.py`, `tests/integration/__init__.py`, `pyproject.toml`, `requirements-dev.lock`, `tasks/todo.md`.
- Verification: `python -m pytest tests/integration/test_vector_store.py -q` → 6 passed in 26s on local Neo4j/5.26.31 (healthy, loopback) with sentence-transformers 3.4.1 / torch 2.14.0+cpu / numpy 2.5.3; known Ada/Eiffel/Photosynthesis fixture ranks `c_p_ada` first for "Who wrote the first computer program?"; idempotent re-ingest keeps 3/3 with byte-identical embeddings; corpus and dimension mismatches raise CONFIG_MISMATCH with counts unchanged; missing index raises INDEX_NOT_READY. `python -m pytest tests -q` → 66 passed (60 unit + 6 integration); `ruff check .` passed; new files `ruff format --check` clean; `compileall -q src` passed; `pip check` clean.
- Environment: same host as T01; Neo4j 5.26.31 Community left running with `chunk_embedding` (cosine, 384, ONLINE) and Chunk/IndexMeta uniqueness constraints; no Chunk/IndexMeta rows retained (test teardown). Model `sentence-transformers/all-MiniLM-L6-v2` downloaded to HF cache on first run.
- Remaining: none for T04. No CLI wiring, no reranking/hybrid, no answer generation or scoring.
- Next eligible ticket: T05 (not started).

- [x] Implement.
- Acceptance: embeddings match model dimension; index ONLINE check gates retrieval; repeated ingestion is idempotent and mismatched fingerprints fail.
- Verify: `python -m pytest tests/integration/test_vector_store.py -q` on local Neo4j; known fixture ranks its relevant paragraph in candidates.
- Depends on: T03.
- Files (M): `src/graphrag_lab/embeddings.py`, `src/graphrag_lab/neo4j_store.py`, `src/graphrag_lab/retrieval.py`, `tests/integration/test_vector_store.py`.
- Estimate: 60–90 minutes.

## T05 — Grounded vector answers [vector-qa]

Execution record:
- Status: Done (2026-09-29).
- Implemented: stdlib OpenAI-compatible LLM adapter (temperature-zero JSON mode, initial + max 2 retries with backoff, worst-case call/token/cost reservation, fake replay backend, no new dependencies/secrets in logs) and bounded grounded answering (top-8 / 3,000-token packing with `char-word-max-v1` + framing overhead, system+evidence prompt treating corpus as data, strict chunk_id/sentence_id validation, supported-requires-citation / insufficient-requires-empty, empty/all-dropped abstention without LLM call).
- Files changed: `src/graphrag_lab/llm.py`, `src/graphrag_lab/answering.py`, `tests/unit/test_answering.py`, `tests/fixtures/llm_outputs.json`, `tasks/todo.md`.
- Verification: `python -m pytest tests/unit/test_answering.py -q` → 14 passed (supported citations, invented/out-of-range rejection, empty/all-dropped abstention with zero calls, budget/usage, bounds, fixture schema); `python -m pytest tests/unit -q` → 74 passed; `ruff check .` passed; `ruff format --check` clean for new Python files (JSON fixture excluded — ruff format corrupts JSON with trailing commas); `compileall -q src` passed; `pip check` clean. Offline budgeted smoke with `RetrievedChunk` + `FakeLlmBackend`: supported Ada answer, 2 packed chunks, 202 context tokens, usage reconciled (tracker 1 call). Live provider smoke NOT run: `GRAPHRAG_LLM_MODEL` / `GRAPHRAG_LLM_API_KEY` are unset, so no paid call was attempted.
- Remaining: live smoke done 2026-09-29 (local OmniRoute gateway, model auto/fast, 1 question): supported "Ada Lovelace" with valid citation, 1 call / 285 tokens reconciled, cost unknown (no pricing metadata). No CLI wiring, reranking/hybrid, or scoring (T06+).
- Next eligible ticket: T06 (not started).

- [x] Implement.
- Acceptance: bounded context produces answer/status/citations; invented citation ids are rejected; empty context abstains.
- Verify: `python -m pytest tests/unit/test_answering.py -q`; one explicit budgeted live smoke question after credentials are configured.
- Depends on: T04.
- Files (M): `src/graphrag_lab/llm.py`, `src/graphrag_lab/answering.py`, `tests/unit/test_answering.py`, `tests/fixtures/llm_outputs.json`.
- Estimate: 45–60 minutes.

## T06 — Baseline scoring and artifact export [evaluation]

Execution record:
- Status: Done (2026-09-29).
- Implemented: official HotpotQA answer normalization/EM/F1 with yes/no/noanswer handling; paragraph recall, all-support coverage and sentence-support recall scored from actual candidate/packed evidence with paragraph_id scoping and dedup of repeats/overlaps; fixed-denominator run evaluation (failures score zero, success-only diagnostics separate, frozen id/hash validation, p50/p95 latency, token/cost sums, unresolved-citation audit); atomic baseline export (results.jsonl, aggregate.json, manifest.json, summary.md) with guarded writes, content hashes and explicit limitations.
- Files changed: `src/graphrag_lab/scoring.py`, `src/graphrag_lab/evaluation.py`, `src/graphrag_lab/reports.py`, `tests/unit/test_scoring.py`, `tasks/todo.md`.
- Verification: `.venv/Scripts/python -m pytest tests/unit/test_scoring.py -q` → 12 passed (normalization, official yes/no/noanswer, packed-context paragraph/sentence/all-support, Town-variant isolation, dedup, candidate-vs-packed, fixed denominator, frozen-set validation, FakeLlm end-to-end with resolving citations, fixture export reconcile); `.venv/Scripts/python -m pytest tests/unit -q` → 86 passed; `.venv/Scripts/python -m pytest tests -q` → 92 passed (86 unit + 6 integration on local Neo4j/5.26.31); `ruff check .` passed; `ruff format --check` clean for new files; `compileall -q src` passed; `pip check` clean. C2 smoke export to `artifacts/baseline-smoke` (removed after check): q_success EM 1/F1 1/all-support 1, q_miss EM 0/F1 0/all-support 0, q_failed PROVIDER_ERROR EM 0 counted in denominator (n_total 3, EM 33.33, success-only 50.0); manifest records run/strategy/corpus/models/budgets/question-set hash and artifact hashes; unknown cost stays null labeled unknown; no secrets in exports.
- Remaining: none for T06. No reranking/hybrid arms, no paired bootstrap intervals, no jobs/dashboard wiring (T11+).
- Next eligible ticket: T07 (not started).

- [x] Implement.
- Acceptance: official answer-normalization semantics verified; paragraph/sentence/all-support metrics computed from actual packed context; fixed denominator includes failed questions.
- Verify: `python -m pytest tests/unit/test_scoring.py -q`; export a fixture baseline and manually reconcile one success, one miss and one failed request.
- Depends on: T05.
- Files (M): `src/graphrag_lab/scoring.py`, `src/graphrag_lab/evaluation.py`, `src/graphrag_lab/reports.py`, `tests/unit/test_scoring.py`.
- Estimate: 60–90 minutes.

### Checkpoint C2

- [x] Vector baseline runs end to end; source citations resolve; scores match known fixtures; saved manifest records all settings.

## T07 — Validated extraction and pilot [graph-index]

Execution record:
- Status: Done (2026-09-29).
- Implemented: fixed-schema extraction with chunk-scoped evidence validation (6 entity / 12 relation enums, verbatim quotes, endpoint/direction/cap checks), temperature-zero prompts, cache by chunk-id+hash+model+prompt/schema version, max 2 retries with backoff and quarantine, worst-case budget reservation reconciled every attempt, deterministic 20-chunk pilot selection/audit and atomic extractions/failures/manifest export.
- Files changed: `src/graphrag_lab/extraction.py`, `src/graphrag_lab/models.py`, `src/graphrag_lab/llm.py`, `tests/unit/test_extraction.py`, `configs/extraction.yaml`, `tasks/todo.md`.
- Verification: `python -m pytest tests/unit/test_extraction.py -q` → 16 passed (enums, dangling/ownership/verbatim, caps, directions, cache, retry/quarantine, budget, pilot, artifacts, codes); `python -m pytest tests/unit -q` → 102 passed; `python -m pytest tests -q` → 108 passed (102 unit + 6 integration on local Neo4j/5.26.31); `ruff check .` passed; new files `ruff format --check` clean; `compileall -q src` passed; `pip check` clean. Offline budgeted pilot smoke with `FakeLlmBackend`: 20 pilot chunks, 20 calls, 7760 input tokens reconciled, 0 quarantined, cache-hit second run 0 calls. Live provider pilot NOT run: `GRAPHRAG_LLM_MODEL` / `GRAPHRAG_LLM_API_KEY` unset, so no paid call attempted.
- Remaining: one explicit budgeted live pilot once LLM model/key/budget are configured (e.g. `load_extraction_config` + `extract_chunks` with `BudgetTracker` on 20 real chunks, inspect `summarize_pilot`/`render_pilot_markdown` and usage before full extraction). No resolution, graph loading, or CLI wiring (T08+).
- Next eligible ticket: T08 (not started).

- [x] Implement.
- Acceptance: fixed enums/evidence checks enforced; 20-chunk pilot audited; cache, retry limit and spend reservation account for every attempt.
- Verify: `python -m pytest tests/unit/test_extraction.py -q`; inspect pilot output and actual usage before enabling full extraction.
- Depends on: T03, T05.
- Files (M): `src/graphrag_lab/extraction.py`, `src/graphrag_lab/models.py`, `src/graphrag_lab/llm.py`, `tests/unit/test_extraction.py`, `configs/extraction.yaml`.
- Estimate: 60–90 minutes plus provider runtime.

## T08 — Conservative entity resolution [graph-index]

Execution record:
- Status: Done (2026-09-29).
- Implemented: offline conservative resolver (Unicode/case/punctuation normalization; exact same-type, acronym and first+last middle-compatible merges; first-initial, single-token PERSON, common two-token PERSON and conflicting-middle vetoes; per-addition canonical+member compatibility preventing transitive collapse) with corpus-scoped canonical ids, full mention audit (candidate scores, decision reasons) and atomic guarded artifacts.
- Files changed: `src/graphrag_lab/resolution.py`, `tests/unit/test_resolution.py`, `tests/fixtures/entity_cases.json`, `configs/resolution.yaml`, `tasks/todo.md`.
- Verification: `python -m pytest tests/unit/test_resolution.py -q` → 15 passed (normalization, exact/acronym/middle merges, type mismatch, J. Smith/John Smith/Jack Smith ambiguity, same-name distinct John Smith, transitive John A./John/John B. collision, auditability, determinism/corpus scoping, fixture pairings, error codes, artifacts/config); `python -m pytest tests/unit -q` → 117 passed; `python -m pytest tests -q` → 123 passed (117 unit + 6 integration on local Neo4j); `ruff check .` passed; `ruff format` clean for new files; `compileall -q src` passed; `pip check` clean.
- Remaining: none for T08. Embedding/context similarity deferred by design (lexical ranking only); uncertain pairs stay separate. No graph loading, linking, or CLI wiring (T09+).
- Next eligible ticket: T09 (not started).

- [x] Implement.
- Acceptance: same-type candidates with corroborated aliases resolve; ambiguous initials/common names remain separate; all decisions and original mentions remain auditable.
- Verify: `python -m pytest tests/unit/test_resolution.py -q` including J. Smith/John Smith ambiguity, same-name distinct people and transitive merge collision cases.
- Depends on: T07.
- Files (M): `src/graphrag_lab/resolution.py`, `tests/unit/test_resolution.py`, `tests/fixtures/entity_cases.json`, `configs/resolution.yaml`.
- Estimate: 60–90 minutes.

## T09 — Evidence-backed graph snapshot [graph-index]

Execution record:
- Status: Done (2026-09-29).
- Implemented: deterministic corpus-scoped graph snapshot builder (canonical REL ids, MENTIONS aggregation, verbatim evidence re-validation, fingerprint, coverage/activation gate with explicit partial opt-in, guarded relation_evidence artifacts) and Neo4j persistence (Entity/GraphMeta constraints, fingerprint-gated idempotent MERGE with evidence union preserving chunk embeddings, counts/fetch helpers).
- Files changed: `src/graphrag_lab/graph_index.py`, `src/graphrag_lab/neo4j_store.py`, `tests/integration/test_graph_index.py`, `tests/fixtures/bridge.json`, `tasks/todo.md`.
- Verification: `python -m pytest tests/integration/test_graph_index.py -q` → 10 passed in ~5s on local Neo4j/5.26.31 (healthy, loopback; no embedding/LLM downloads); bridge `The Night Cartographer WRITTEN_BY Elena Voss BORN_IN Lisbon` traverses two hops with `c_t09_book`/`c_t09_author` evidence and 4 MENTIONS; repeat ingest keeps 3 entities/2 relations/4 mentions with byte-identical evidence and unioned `c_t09_note` stays deduplicated; foreign-corpus ingest raises CONFIG_MISMATCH with counts unchanged; incomplete coverage raises INDEX_NOT_READY unless `allow_partial=True` returns `is_partial=True`. `python -m pytest tests -q` → 133 passed (117 unit + 16 integration); `python -m pytest tests/unit/test_resolution.py -q` → 15 passed; `ruff check .` passed; new files `ruff format --check` clean; `compileall -q src` passed; `pip check` clean.
- Environment: same host as T01; Neo4j 5.26.31 Community left running with Entity/GraphMeta constraints; no Chunk/Entity/IndexMeta/GraphMeta rows retained (test teardown).
- Remaining: none for T09. No entity linking, hybrid retrieval, reranking, CLI wiring, or SQLite jobs (T10+).
- Next eligible ticket: T10 (not started).

- [x] Implement.
- Acceptance: directed relations and mention links preserve valid corpus-scoped evidence; repeated load creates no duplicates; incomplete graph cannot activate silently.
- Verify: `python -m pytest tests/integration/test_graph_index.py -q`; count records before/after repeat ingest and verify a two-document bridge.
- Depends on: T04, T08.
- Files (M): `src/graphrag_lab/graph_index.py`, `src/graphrag_lab/neo4j_store.py`, `tests/integration/test_graph_index.py`, `tests/fixtures/bridge.json`.
- Estimate: 45–75 minutes.

### Checkpoint C3

- [x] Pilot quality and cost reviewed; resolution collision tests pass; all graph evidence references exist; graph coverage reported honestly.

## T10 — Bounded hybrid retrieval [hybrid-qa]

Execution record:
- Status: Done (2026-09-29).
- Implemented: offline query linking (exact-first, acronym, conservative non-PERSON lexical; MATCH-only, max 5 seeds) and bounded hybrid retrieval (vector + 2-hop REL expansion either direction with retained semantic direction; 25 edges/node/hop, 100 visited, 40 graph chunks, 2s timeout; direct-evidence-first priority; deterministic RRF to max 60 with vector/graph/both origin, ranks, path ids, distance; visible vector fallback).
- Files changed: `src/graphrag_lab/linking.py`, `src/graphrag_lab/hybrid.py`, `src/graphrag_lab/neo4j_store.py`, `tests/integration/test_hybrid.py`, `tasks/todo.md`.
- Verification: `python -m pytest tests/integration/test_hybrid.py -q` → 7 passed in ~12s on local Neo4j/5.26.31 (healthy, loopback; synthetic 384-dim vectors, no model/LLM downloads); bridge merges book(both)+author(graph) with Lisbon visited and repeat-identical order; no-link falls back with `no_linked_entities`; hub (30 leaves) truncates per-node edges with `max_edges_per_node_hit` and `max_merged_hit` at cap 5; injected-clock timeout falls back with `graph_timeout`; foreign corpus raises CONFIG_MISMATCH with counts unchanged. `python -m pytest tests -q` → 140 passed (133 prior + 7 hybrid); `ruff check .` passed; new files `ruff format --check` clean; `compileall -q src` passed; `pip check` clean.
- Environment: same host as T01; Neo4j 5.26.31 Community left running with Chunk/Entity/IndexMeta/GraphMeta constraints and chunk_embedding (cosine, 384, ONLINE); no Chunk/Entity/IndexMeta/GraphMeta rows retained (test teardown).
- Remaining: none for T10. No reranking/triples arms, no CLI/dashboard wiring (T11+).
- Next eligible ticket: T11 (not started).

- [x] Implement.
- Acceptance: query entities link without creating nodes; two-hop expansion respects frontier budgets; evidence chunks merge with vectors and fallback is recorded.
- Verify: `python -m pytest tests/integration/test_hybrid.py -q` with bridge, no-link, high-degree and timeout fixtures.
- Depends on: T09.
- Files (M): `src/graphrag_lab/linking.py`, `src/graphrag_lab/hybrid.py`, `src/graphrag_lab/neo4j_store.py`, `tests/integration/test_hybrid.py`.
- Estimate: 60–90 minutes.

## T11 — Reranking and controlled experiment arms [hybrid-qa]

Execution record:
- Status: Done (2026-09-29).
- Implemented: cross-encoder reranking with deterministic rank-only ordering (FakeReranker + lazy CrossEncoderBackend, truncation recorded, failure degrades with flag) and controlled arms A vector / B vector_rerank / C hybrid_rerank / D hybrid_rerank_triples sharing final top-8 / 3,000-token budgets; duplicate chunks removed, raw scales never summed, at most 12 evidence-only triples counted in the same budget.
- Files changed: `src/graphrag_lab/reranking.py`, `src/graphrag_lab/retrieval.py`, `src/graphrag_lab/hybrid.py`, `src/graphrag_lab/answering.py`, `src/graphrag_lab/evaluation.py`, `tests/unit/test_reranking.py`, `tasks/todo.md`.
- Verification: `python -m pytest tests/unit/test_reranking.py -q` → 12 passed (arms, dedupe, rank-only, budgets, evidence-only triples, degraded, truncation, compatibility, fixture all-arms); `python -m pytest tests/unit -q` → 129 passed; `python -m pytest tests -q` → 152 passed (129 unit + 23 integration on local Neo4j/5.26.31); `ruff check .` passed; `ruff format --check` clean for touched files; `compileall -q src` passed; `pip check` clean. Fixture smoke: all arms max_chunks 8 / max_context_tokens 3000, max_triples 0/0/0/12, same question hash 372acf77.
- Remaining: none for T11. Live CrossEncoder download not exercised (FakeReranker offline only). No CLI/dashboard wiring (T12a+).
- Next eligible ticket: T12a (not started).

- [x] Implement.
- Acceptance: arms A–D selectable; duplicate chunks removed and raw score scales not added; identical final token/chunk budgets and evidence-only triples enforced.
- Verify: `python -m pytest tests/unit/test_reranking.py -q`; run all arms on fixture questions and compare manifests/context usage.
- Depends on: T06, T10.
- Files (M): `src/graphrag_lab/retrieval.py`, `src/graphrag_lab/hybrid.py`, `src/graphrag_lab/answering.py`, `src/graphrag_lab/evaluation.py`, `tests/unit/test_reranking.py`.
- Estimate: 45–75 minutes.

## T12a — Durable local jobs [dashboard]

Execution record:
- Status: Done (2026-09-29).
- Implemented: SQLite durable jobs (WAL, FK, UTC, migrations; datasets/snapshots/jobs/attempts + runs/predictions/usage/comparisons; atomic claim, heartbeat leases, stale->interrupted recovery, cancel/resume/budget-pause preserving validated checkpoints, active-kind+snapshot+config dedup) with single-worker executor (deterministic item order, per-item checkpoint, quarantine-and-continue, empty control-job completion) and CLI worker/enqueue-demo/index/evaluate/job-get/list/cancel/resume.
- Files changed: `src/graphrag_lab/jobs.py`, `src/graphrag_lab/worker.py`, `src/graphrag_lab/run_store.py`, `tests/integration/test_jobs.py`, `src/graphrag_lab/cli.py`, `tasks/todo.md`.
- Verification: `python -m pytest tests/integration/test_jobs.py -q` → 12 passed (restart, single-claim race, active dedup, interrupt/resume with zero duplicate completed calls, cancel/resume, pre-start cancel, budget-pause/resume, quarantine/retry, stale recovery, enqueue_index/evaluation, run_store predictions/usage/completion-gate/comparison); `python -m pytest tests/unit -q` → 129 passed; `python -m pytest tests -q` → 164 passed (129 unit + 35 integration on local Neo4j/5.26.31); `ruff check .` passed; new files `ruff format --check` clean; `compileall -q src` passed; `pip check` clean. CLI smoke on temp DB: `enqueue-demo --count 3` → queued, `worker --once` → completed 3/3, `job-list` shows checkpoint with 3 completed entries.
- Remaining: none for T12a. No Neo4j/LLM execution in worker handlers (demo + empty control-job completion only); full index/evaluation executors and dashboard polling arrive in T12b/T12c/T13.
- Next eligible ticket: T12b (not started).

- [x] Implement.
- Acceptance: SQLite state survives restart; one worker claims each job once; cancellation/resume and budget-pause preserve validated work.
- Verify: `python -m pytest tests/integration/test_jobs.py -q`; interrupt worker mid-fixture and resume without duplicate completed calls.
- Depends on: T07, T11.
- Files (M): `src/graphrag_lab/jobs.py`, `src/graphrag_lab/worker.py`, `src/graphrag_lab/run_store.py`, `tests/integration/test_jobs.py`, `src/graphrag_lab/cli.py`.
- Estimate: 60–90 minutes.

## T12b — Setup and index dashboard [dashboard]

- Design target: UI-UX.md revision 2. Implement the shared theme/navigation and the guided Dataset → Vector → Graph → Ready sections; Overview has one state-dependent next action and recent runs. Use shared tokens rather than per-page styling.

Execution record:
- Status: Done (2026-09-29).
- Implemented: setup view-model (`setup_flow`: 4 guided sections, single Overview next action, persisted progress/spend panels, session idempotency guard, worker/Neo4j readiness messages, pilot estimate with labeled assumptions, snapshot-change flag, recent runs) plus Streamlit Overview (`app/app.py`: next step, Corpus→Vector→Graph→Evaluation list, worker/DB states, recent runs, footer) and Dataset & Index page (`app/pages/dataset.py`: dataset form with advanced settings, vector/graph enqueue with prerequisites inline, one progress panel each, Cancel secondary, Retry-failed-chunks/Resume preserving settings, Details) sharing tokens via `app/components/theme.py` + `app/components/status.py` (text+icon pills, no color-only status, visible focus).
- Files changed: `src/graphrag_lab/setup_flow.py`, `src/graphrag_lab/worker.py`, `app/app.py`, `app/pages/dataset.py`, `app/components/theme.py`, `app/components/status.py`, `app/__init__.py`, `app/components/__init__.py`, `app/pages/__init__.py`, `tests/e2e/test_setup_flow.py`, `tests/e2e/__init__.py`, `pyproject.toml`, `requirements-dev.lock`, `tasks/todo.md`.
- Verification: `python -m pytest tests/e2e/test_setup_flow.py -q` → 13 passed (no-corpus/queued/running/failed/cancelled/paused/interrupted/ready walk, restart-persisted progress/spend, rerun duplicate prevention, worker/Neo4j error wording, pilot assumptions, token contract); `python -m pytest tests -q` → 177 passed (129 unit + 13 e2e + 35 integration on local Neo4j/5.26.31); `ruff check .` passed; touched files `ruff format --check` clean; `compileall -q src app` passed; `pip check` clean. Streamlit 1.64.0 added to `pyproject.toml`, lock regenerated with uv and installed from hashes. AppTest renders: Overview shows first-run "Prepare dataset" next action with zero exceptions; Dataset page shows all four guided sections with zero exceptions (vector build correctly withheld until a dataset exists). Empty index/evaluation jobs now fail visibly (`INVALID_INPUT`, executor not wired) instead of completing silently; real index execution lands before C4.
- Remaining: none for T12b. Real chunk-embedding/graph execution behind the enqueued index jobs, Ask & Compare/Evidence pages (T12c), and narrow-window/keyboard review of the new pages (with T12c).
- Next eligible ticket: T12c (not started).

- [x] Implement.
- Acceptance: readiness/setup errors understandable; index progress and spend persisted; duplicate job submission prevented across Streamlit reruns.
- Verify: manually walk no-corpus, no-worker, running, failed, cancelled and ready states; run `python -m pytest tests/e2e/test_setup_flow.py -q`.
- Depends on: T12a.
- Files (M): `app/app.py`, `app/pages/dataset.py`, `app/components/status.py`, `tests/e2e/test_setup_flow.py`.
- Estimate: 45–75 minutes.

## T12c — Answer comparison and evidence UI [dashboard]

- Design target: UI-UX.md revision 2. Preserve question/run state, display two aligned answer panels with independent progress, open citations in the inline evidence inspector and restore context after full evidence review. Keep evaluation controls out of pre-freeze demo examples.

Execution record:
- Status: Done (2026-09-29).
- Implemented: compare view-model (`compare_flow`: A–D arm labels/blurbs, default vector vs hybrid+rerank+triples, tuning-only picker, origin/context/cited source rows, exact citation resolution, flagged relation rows, visible fallback/cap text, per-arm run persistence, saved-only review, matched-budget guard, local deep links) plus real vector index executor (`index_tasks`: corpus→chunks→model-tokenizer validation→Neo4j ingest in batches, cancel between batches, honest snapshot gating, cost unknown-never-zero) with a worker fix (item accounting never rewinds handler-reported sub-item progress); shared evidence components (`sources`: descriptive source buttons, cited/retrieved reader cues, keyboard radio relation table); Ask & Compare page (explicit form submit, draft-preserving examples, staged per-arm status, inline inspector, full-review handoff, saved-run deep links, no gold answers) and Evidence page (arm switch retaining shared sources, reader, connections, retrieval details, back-to-results); CLI worker runs `index_vector`, `index` enqueues the build item.
- Files changed: `src/graphrag_lab/compare_flow.py`, `src/graphrag_lab/index_tasks.py`, `src/graphrag_lab/worker.py`, `src/graphrag_lab/cli.py`, `app/pages/compare.py`, `app/pages/evidence.py`, `app/components/sources.py`, `app/app.py`, `app/pages/dataset.py`, `tests/e2e/test_compare_flow.py`, `tests/unit/test_index_tasks.py`, `tests/integration/test_jobs.py`, `tasks/todo.md`.
- Verification: `python -m pytest tests/e2e/test_compare_flow.py -q` → 11 passed; `tests/unit/test_index_tasks.py` → 5 passed; `tests/integration/test_jobs.py` → 13 passed (incl. handler-progress regression); `python -m pytest tests -q` → 194 passed (Neo4j/5.26.31 healthy); `ruff check .`, `ruff format --check`, `compileall -q src app`, `pip check` clean. AppTest: compare/evidence/Overview render with 0 exceptions; every control has a visible label (keyboard-focusable native elements + focus ring); example click edits draft only. C4 live chain (temp mini-corpus, real MiniLM-L6-v2 + Neo4j, FakeLlmBackend for answers): prepare 3 paragraphs → worker `index_vector` completed 3/3 → vector ranks bridge chunk first → hybrid falls back visibly (`graph_not_ready`) → matched budgets → persisted runs → cited evidence resolves; Neo4j test data removed afterwards. No live LLM call attempted (model/key unset at run time).
- Remaining: none for T12c. Graph builds stay LLM-gated (pilot + budget + creds). Physical small-viewport pass deferred to the T15 clean-environment demo (layout is fluid with no fixed widths; tables/code scroll natively).
- Next eligible ticket: T13 (not started).

- [x] Implement.
- Acceptance: side-by-side answers retain strategy labels; citations open exact source sentences; relation table works with keyboard and graph fallback flags are visible.
- Verify: `python -m pytest tests/e2e/test_compare_flow.py -q`; manual keyboard and narrow-window review.
- Depends on: T12b.
- Files (M): `app/pages/compare.py`, `app/pages/evidence.py`, `app/components/sources.py`, `tests/e2e/test_compare_flow.py`.
- Estimate: 60–90 minutes.

### Checkpoint C4

- [x] Local app completes indexing → compare → inspect evidence; matched-budget ablations work; worker restart and traversal-limit checks pass.

## T13 — Locked evaluation and single-hop controls [evaluation]

Execution record:
- Status: In progress (2026-09-29, implementation complete pending human review and live frozen runs).
- Implemented: frozen evaluation config (seed 42, 2000 replicates, 95% CI, 30/120/30 splits, A-D arms, shared 30/8/3000 budgets, 2pp single-hop tolerance); 30 single-hop controls with one-paragraph provenance and answer-span validation (draft, `human_verified=false`); locked-evaluation backend in `evaluation.py` (control load/freeze gate, split-disjoint and exact-locked-id enforcement, run compatibility refusing unlabeled deltas, deterministic paired bootstrap intervals, bridge/comparison/single-hop strata preserving failures, tuning-selection and single-hop-regression helpers). `scoring.py` reviewed: official HotpotQA semantics from T06 already satisfy T13, no changes required.
- Files changed: `src/graphrag_lab/evaluation.py`, `tests/unit/test_evaluation.py`, `data/controls/single_hop.jsonl`, `configs/evaluation.yaml`, `tasks/todo.md` (`src/graphrag_lab/scoring.py` reviewed, unchanged).
- Verification: `python -m pytest tests/unit/test_evaluation.py -q` → 9 passed (config defaults, 30-control shape/pending-review gate, bad-provenance rejections, split/locked enforcement, compatibility guard, deterministic bootstrap with failures preserved, strata counts, compare_runs intervals/strata/refusals, tuning/regression helpers); `python -m pytest tests/unit -q` → 143 passed; `python -m pytest tests -q` → 203 passed (143 unit + integration/e2e on local Neo4j/5.26.31); `ruff check .` passed; `ruff format --check` clean for touched files; `compileall -q src` passed; `pip check` clean. Offline audit: 30 controls load, `validate_single_hop_controls(require_verified=False)` passes with `n_verified=0`, `question_set_hash=57389cc82fd2a1aec8b78282d2c99ffd728b0f70021f0b794b6ffb6f0a5487d4`, `freeze_hash=2d7c84bfd6ba38b02810b905a03260282706b1dcda6182d34d453257fe168a64`, file SHA256 `49EF222FED8B85003BD7AF960C81B9C4F0D11016B5BCE3FC9217993B50797D64`; `require_verified=True` correctly refuses frozen use until review. Live frozen execution NOT run: no dataset (`data/raw` empty, no `artifacts/dataset`), `GRAPHRAG_LLM_MODEL`/`GRAPHRAG_LLM_API_KEY` unset, so 120 multi-hop + 30 control arms across A-D were not executed.
- Remaining: (1) DONE 2026-09-29 — user approved all 30 controls after auditing the source-sentence review sheet; agent flipped `human_verified` false→true (byte-identical except the flag, 30/30 rows) and re-froze: `question_set_hash=57389cc82fd2a1aec8b78282d2c99ffd728b0f70021f0b794b6ffb6f0a5487d4` (ids unchanged), new `freeze_hash=d818023c79d113c8cb5ffd9ce5024bbaf8307261f22273fb59d7da864b8397ff`, file SHA256 `F3D30CDC0439C74A67DE8E2D677C3B18226771CEC0019AAC7EBEAC11291FC0B6`; `validate_single_hop_controls(require_verified=True)` now passes with `n_verified=30`; `tests/unit/test_evaluation.py` updated to the reviewed state (renamed to `test_single_hop_controls_frozen_shape_human_verified`).
- Dataset DONE 2026-09-29: canonical host dead, so verified mirror `https://huggingface.co/datasets/namlh2004/hotpotqa` (public, revision `7e54db4`) downloaded to `data/raw/hotpot_dev_distractor_v1.json` (61,065,698 bytes, SHA256 `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`, 7405 rows bridge 5918/comparison 1487); `prepare` built `artifacts/dataset/` with 150 selected (30 tuning: 24 bridge/6 comparison; 120 evaluation: 96/24 — 80/20 preserved), `corpus_id=hp_1090980044eafe405c7a8bad7b1f63c3ecdf5cc8a69dc3cc5d4055af2b2091da`, rerun byte-identical; 1 upstream-invalid row quarantined (`5ae61bfd5542992663a4f261`, supporting sentence 902 of 5 — genuine upstream noise, see T02 note); 2000 chunks from 1499 paragraphs with `verify_chunks`/`verify_supporting_facts`/`verify_gold_covered_by_chunks` passing (empty-sentence tolerance, see T03 note); split discipline 30/120/30 disjoint vs controls.
- Remaining: live frozen execution — 120 locked multi-hop + 30 control arms across A-D with identical corpus/index/answer-model/budgets, then `compare_runs` + stratified report. BLOCKED on LLM: `GRAPHRAG_LLM_MODEL`/`GRAPHRAG_LLM_API_KEY` unset, so no extraction/answering calls and no paid spend were attempted (graph arms need extraction; all arms need answer calls). Awaiting model selection, credentials and spend/token cap; no code changes expected unless review finds control errors.
- Re-check (2026-09-29, agent): `tests/unit/test_evaluation.py` → 9 passed; 30 controls load with `n_verified=0`, hashes unchanged (`question_set_hash=57389cc8…`, `freeze_hash=2d7c84bf…`, file SHA256 `49EF222F…`); `require_verified=True` refuses frozen use; `configs/evaluation.yaml` loads (`evaluation-v1`, 2000 replicates, 30/120/30); Neo4j/5.26.31 healthy on loopback. Live blockers confirmed: `data/raw/` empty (0 files, no `artifacts/dataset`), `GRAPHRAG_LLM_MODEL`/`GRAPHRAG_LLM_API_KEY` unset in process env. Review sheet for (1) rendered from source sentences only (no system run); all 30 answers occur verbatim in their cited sentence pending human sign-off.
- Next eligible ticket: T14 (not started automatically).

- [ ] Implement.
- Acceptance: 30 human-verified single-hop controls frozen; tuning and locked data use enforced; paired intervals/stratified results preserve all attempts and failures.
- Verify: `python -m pytest tests/unit/test_evaluation.py -q`; audit control answer provenance, then execute frozen arms on 120 multi-hop and 30 control questions.
- Depends on: T11, T12a.
- Files (M): `src/graphrag_lab/evaluation.py`, `src/graphrag_lab/scoring.py`, `tests/unit/test_evaluation.py`, `data/controls/single_hop.jsonl`, `configs/evaluation.yaml`.
- Estimate: 60–90 minutes plus human control review and inference runtime.

## T14 — Evaluation dashboard and report [dashboard]

- Design target: UI-UX.md revision 2. Lead with compatible-run F1, all-support coverage and latency comparisons; show uncertainty/counts, readable regression filters, and separate indexing/query costs. Never rerun a question implicitly when opening a saved result.

Execution record:
- Status: Done (2026-09-29).
- Implemented: compatible-run headlines (F1, all-support, latency with absolutes/counts/split and 95% intervals only when compatible), regression view-model (F1-direction Improved/Worsened/Unchanged plus additive Failed/Abstained, multi-hop/single-hop tabs, Question→Baseline→Candidate→Difference→Status tables, failure subset, separated query/index cost panels with unknown-never-zero, empty state with no placeholder charts) and paired comparison exports (`comparison.json`, `regression.csv`, `report.md` with guarded atomic writes and fixed-denominator reconcile including failures); Evaluations page reads saved RunStore predictions/labels only and never triggers retrieval or LLM.
- Files changed: `src/graphrag_lab/reports.py`, `app/components/metrics.py`, `app/pages/evaluations.py`, `tests/e2e/test_evaluation_view.py`, `tasks/todo.md`.
- Verification: `python -m pytest tests/e2e/test_evaluation_view.py -q` → 9 passed (headlines, incompatible refusal, labels/filters, tabs/tables, cost separation, empty state, export reconcile with failures, idempotent re-export, page import); `python -m pytest tests -q` → 212 passed (129 unit + e2e/integration on local Neo4j/5.26.31); `ruff check .` passed; `ruff format` applied to touched files; `compileall -q src app` passed; `pip check` clean. AppTest: evaluations page renders empty state with 0 exceptions ("No evaluation results yet"); `_build_view` on temp DB with missing labels returns saved answers with Not-measured notice and no rerun. No live LLM call (model/key unset).
- Remaining: none for T14. Live frozen 120+30 arm execution and human single-hop review remain with T13; physical small-viewport/keyboard pass deferred to T15 clean-environment demo (layout uses native Streamlit tabs/tables/selects with no fixed widths; tables scroll natively).
- Next eligible ticket: T15 (not started).

- [x] Implement.
- Acceptance: incompatible runs cannot produce unlabeled paired deltas; multi-hop/control metrics and indexing/query costs separated; per-question regression drill-down and exports work.
- Verify: `python -m pytest tests/e2e/test_evaluation_view.py -q`; exported aggregates reconcile with per-question records including failures.
- Depends on: T12c, T13.
- Files (M): `app/pages/evaluations.py`, `app/components/metrics.py`, `src/graphrag_lab/reports.py`, `tests/e2e/test_evaluation_view.py`.
- Estimate: 45–75 minutes.

## T15 — Reproducibility and portfolio handoff [evaluation]

Execution record:
- Status: Done (2026-09-29).
- Implemented: offline fixture-scale full pipeline (`prepare` byte-identical, sentence provenance, Fake-LLM grounded answers with abstention, fixed-denominator scoring, paired bootstrap comparison with incompatibility refusal, guarded baseline/comparison exports with unknown-never-zero costs, durable jobs with duplicate-guard and restart persistence, compare/evaluation view-models, single-hop freeze gates, config guards) plus portfolio handoff (`README.md` quickstart/fixture/results, `docs/DEMO.md` clean-environment setup + demo checklist + troubleshooting, `docs/RESULTS.md` measured-only report with models/hardware/corpus/cost/negative-results and measured-only resume wording) and planning updates (`docs/README.md`, `tasks/plan.md`).
- Files changed: `tests/e2e/test_full_pipeline.py`, `docs/DEMO.md`, `docs/RESULTS.md`, `README.md`, `docs/README.md`, `tasks/plan.md`, `tasks/todo.md`.
- Verification: `python -m pytest tests/e2e/test_full_pipeline.py -q` → 9 passed (offline, no Neo4j/network/paid calls); `python -m pytest -q` → 221 passed (212 prior + 9 new; includes Neo4j/5.26.31 integration in ~31s); `python -m ruff check .` → All checks passed; `ruff format --check` clean (40 files); `python -m compileall -q src app` passed; `pip check` clean. DEMO.md checklist followed logically from a fresh venv + `docker compose config --quiet` + fixture flow; Streamlit pages covered via existing AppTest renders and view-model contracts (physical 320/768/1024/1440px + screen-reader pass remains a manual reviewer step per DEMO.md).
- Environment: same host as T01 (i5-10210U, 7.79 GiB RAM, Intel UHD, Docker 29.8.0); CPython 3.12.14; sentence-transformers 3.4.1 / streamlit 1.64.0 / neo4j 5.28.6 / pydantic 2.13.5; Neo4j 5.26.31 Community digest-pinned, loopback only. No HotpotQA download (`data/raw` empty, no `artifacts/dataset`), no model download, no paid LLM call (`GRAPHRAG_LLM_MODEL`/`GRAPHRAG_LLM_API_KEY` unset) in T15 verification.
- Remaining: none for T15 acceptance. Disclosed (not T15 defects): T13 live work outstanding — (1) human audit of `data/controls/single_hop.jsonl` (`human_verified=false`, `require_verified=True` correctly refuses frozen use) and (2) frozen 120 multi-hop + 30 control arms across A–D with identical corpus/index/answer-model/budgets; no code changes expected unless review finds control errors. No real-data F1/recall/latency/cost delta is claimed.
- Next eligible ticket: T16 (optional, only after core; do not start automatically).

- [x] Implement.
- Acceptance: fresh local setup executes fixture flow; report includes models/hardware/corpus caveats/cost and negative results; resume wording uses only measured outcomes.
- Verify: `python -m pytest -q`, `python -m ruff check .`, `python -m compileall src app`; follow setup guide in a clean environment and run the demo checklist.
- Depends on: T14.
- Files (M): `README.md`, `docs/RESULTS.md`, `docs/DEMO.md`, `tests/e2e/test_full_pipeline.py`.
- Estimate: 45–75 minutes.

### Checkpoint C5 — Core completion

- [x] All core requirements R01–R12 demonstrated at fixture scale with honest gaps; costs and limitations visible; no fabricated metrics; full user flow verified offline; planning documents updated to implemented behavior. Locked 120+30 evaluation and human single-hop review remain with T13 and are disclosed in `docs/RESULTS.md`, not claimed here.

## T16 — Bounded decomposition [decomposition, optional]

Execution record:
- Status: Done (2026-09-29).
- Implemented: bounded sequential decomposition (temperature-zero planner capped at two sub-questions with JSON schema validation; sequential sub-retrieval/grounded sub-answers with dependent placeholder resolution; unsupported first answer stops the chain without reuse as fact; sub-evidence deduped and packed under shared AnswerConfig budgets; planner failure falls back to direct single-shot with visible flag; shared BudgetTracker reservation with BUDGET_EXCEEDED propagation; unknown cost stays None; call/context/evidence/cost logged in DecompositionResult diagnostics) plus shared cap constants in models.
- Files changed: `src/graphrag_lab/decomposition.py`, `src/graphrag_lab/models.py`, `tests/unit/test_decomposition.py`, `tasks/todo.md`.
- Verification: `python -m pytest tests/unit/test_decomposition.py -q` → 7 passed (bridge combine, unsupported-first stop, planner-failure fallback, cap validation, placeholder/budget, empty-evidence abstention, codes/versions); `python -m pytest tests/unit -q` → 150 passed; `python -m pytest -q` → 228 passed (221 prior + 7 new, incl. Neo4j integration ~30s); `ruff check .` passed; `ruff format --check` clean for touched files; `compileall -q src app` passed; `pip check` clean. Offline only: FakeLlmBackend planner/answers, no model download, no paid call.
- Remaining: none for T16. No CLI/dashboard wiring, no tuning-vs-locked comparison (T17); no live LLM smoke (model/key unset).
- Next eligible ticket: T17 (not started automatically).

- [x] Implement only after core delivery.
- Acceptance: at most two sub-questions; unsupported intermediate answer stops the chain; calls, context, intermediate evidence and cost bounded and logged.
- Verify: `python -m pytest tests/unit/test_decomposition.py -q` with successful bridge, unsupported first answer and planner failure fixtures.
- Depends on: T15.
- Files (M): `src/graphrag_lab/decomposition.py`, `src/graphrag_lab/models.py`, `tests/unit/test_decomposition.py`.
- Estimate: 60–120 minutes.

## T17 — Decomposition comparison [decomposition, optional]

Execution record:
- Status: Done (2026-09-29).
- Implemented: optional decomposition arm E (`strategy="decomposition"`) reusing frozen fixed fields and matched final budgets (top 8 / 3000 tokens, no reranker/triples); tuning-first workflow (`decomposition_to_prediction` mapping T16 results to frozen predictions with provenance checks, `freeze_decomposition_selection` requiring tuning split + exact tuning ids + compatible budgets, `assert_decomposition_locked_use` requiring evaluation split + frozen strategies/budgets + exact locked ids, `summarize_decomposition_incremental` reporting paired F1/recall with intervals plus observed latency/token/cost deltas with unknown-never-zero, `decomposition_report_limitations` additive block); core A-D arms/configs unchanged; Evaluations page shows a decomposition banner and exports the R13 limitations block when either run is decomposition.
- Files changed: `src/graphrag_lab/evaluation.py`, `configs/evaluation.yaml`, `app/pages/evaluations.py`, `tests/unit/test_decomposition_eval.py`, `docs/RESULTS.md`, `tasks/todo.md`.
- Verification: `python -m pytest tests/unit/test_decomposition_eval.py -q` → 10 passed (strategy/budgets, config freeze metadata, prediction mapping incl. live T16 chain, tuning comparison deterministic with failures preserved, freeze/locked gates with refusals, incremental F1/recall/latency/cost with unknown handling, guarded delta exports, documentary keys); `python -m pytest tests/unit -q` → 160 passed; `python -m pytest tests -q` → 238 passed (incl. Neo4j integration ~33s on local Neo4j/5.26.31); `ruff check .` passed; `ruff format --check` clean for touched files; `compileall -q src app` passed; `pip check` clean. Offline only: FakeLlmBackend planner/answers, synthetic chunks, no model download, no paid call.
- Remaining: live work outstanding, blocked on dataset + evaluation budget (no code changes expected): (1) run baseline vs decomposition on the 30 tuning ids with identical corpus/index/models/budgets and freeze via `freeze_decomposition_selection`; (2) run the frozen configuration once on 120 locked multi-hop (+ 30 controls separately) gated by `assert_decomposition_locked_use` and export the additive delta report to a new directory. `GRAPHRAG_LLM_MODEL`/`GRAPHRAG_LLM_API_KEY` unset and `data/raw` empty, so no 30/120/30 decomposition comparison was executed.
- Next eligible ticket: none (T17 is the final optional ticket; core handoff remains T15 with T13 live work outstanding).

- [x] Implement only after T16.
- Acceptance: compare on tuning set; freeze selection before locked evaluation; incremental F1, recall, latency and cost reported without replacing old results.
- Verify: run matched-budget tuning comparison, inspect manifests, run selected frozen configuration on locked sets and export delta report.
- Depends on: T16.
- Files (M): `src/graphrag_lab/evaluation.py`, `configs/evaluation.yaml`, `app/pages/evaluations.py`, `docs/RESULTS.md`.
- Estimate: 60–120 minutes plus inference runtime.

## Requirement traceability

R01–R02 → T02–T03; R03 → T04–T06; R04 → T07; R05 → T08; R06 → T09; R07 → T10–T12c; R08 → T11/T13; R09–R10 → T13; R11 → T12a–T12b; R12 → T14–T15; R13 → T16–T17.
