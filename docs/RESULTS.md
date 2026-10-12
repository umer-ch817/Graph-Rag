# Measured results — GraphRAG Lab (T15 core handoff; T13 dataset built 2026-09-29, frozen arms await LLM budget)

Status: core application complete through T15 at fixture scale. No frozen
120-question multi-hop evaluation and no 30-question single-hop control run
were executed in T15. This report contains only measured offline outcomes;
every unmeasured claim is labeled as such. See `docs/DEMO.md` for the
reproducible fixture flow.

## 1. What was actually measured

- Full test suite offline + local Neo4j integration where available:
  `tests/unit`, `tests/integration`, `tests/e2e` including the new
  `tests/e2e/test_full_pipeline.py` (9 tests). Fixture flow uses a
  synthetic 8-question Hotpot-style source (6 sampled, 2 tuning / 4
  evaluation, seed 42), `FakeLlmBackend`, and `FakeRerankerBackend`.
- No network download was performed for T15 verification beyond the
  pre-installed dependencies. No paid LLM call was attempted:
  `GRAPHRAG_LLM_MODEL` and `GRAPHRAG_LLM_API_KEY` are unset.
- Real HotpotQA corpus built 2026-09-29: `data/raw/hotpot_dev_distractor_v1.json`
  (61,065,698 bytes, SHA256 `e3da074d…caa8`, 7405 rows — bridge 5918 /
  comparison 1487) from verified mirror `namlh2004/hotpotqa` revision `7e54db4`
  (canonical `curtis.ml.cmu.edu` host dead; HF API listing + ranged GET verified
  before download). Seeded sample 150 (30 tuning: 24 bridge / 6 comparison;
  120 evaluation: 96 / 24) in `artifacts/dataset/`
  (`corpus_id=hp_10909800…2091da`); rerun byte-identical; one upstream-invalid
  row quarantined in-manifest (`5ae61bfd…`, supporting sentence 902 of 5 —
  genuine upstream noise); 2000 chunks from 1499 paragraphs with provenance
  and gold coverage passing; tuning/locked/control splits disjoint.
- Vector index (MiniLM-L6-v2, 384-dim, cosine) and graph snapshot were
  exercised only through prior ticket integration fixtures on local
  Neo4j/5.26.31, not through a fresh end-to-end build in T15.

### Fixture-scale observations (synthetic, not benchmark claims)

- `prepare_dataset` twice on the synthetic source is byte-identical:
  same corpus id, same 6 selected ids, same paragraph/context hashes.
  `questions.jsonl` carries no answers or supporting facts; every label
  resolves to a stored paragraph sentence.
- `chunk_paragraphs` (one paragraph per chunk for short paragraphs) is
  deterministic; `verify_chunks`, `verify_supporting_facts`, and
  `verify_gold_covered_by_chunks` all pass. Town title variants (2 versions)
  survive as distinct paragraphs; the shared Distractor collapses to one.
- Grounded answering with the fake backend: supported answers carry
  citations that resolve to packed evidence; invented chunk ids and
  out-of-range sentences are rejected; empty evidence abstains without
  consuming a queued LLM output.
- Fixed-denominator scoring on 4 evaluation fixture questions (1 success,
  1 miss, 1 failed `PROVIDER_ERROR`, 1 abstention): failures score zero
  inside the denominator (`n_total=4`, `n_failed=1`, `n_abstained=1`,
  `n_supported=2`); success-only diagnostics are reported separately and
  never replace the primary aggregate.
- Paired comparison on the same fixture set (`replicates=200` in tests;
  frozen default is 2000, seed 42, 95% CI): compatible arms produce F1 and
  all-support deltas with intervals, bridge/comparison strata, and failure
  counts. Incompatible arms (changed corpus id) correctly return
  `compatible=False` and `export_comparison` refuses with
  "Incompatible runs cannot produce an unlabeled paired delta".
- Guarded exports reconcile: per-question `results.jsonl` /
  `regression.csv` rows average exactly to the aggregate F1 and all-support
  values, including failed rows at zero. Re-export with identical bytes is
  idempotent; differing bytes are refused, never overwritten.
- Durable jobs: 4-item demo job persists 2/4 progress across a simulated
  restart and completes 4/4 on resume with no duplicate completed calls;
  duplicate Streamlit submissions reuse the same active job.
- Dashboard view-models: no-corpus → dataset-ready → vector-queued →
  vector-ready transitions verified; `build_sources` / `resolve_citation`
  open exact source sentences; tuning picker excludes locked examples;
  matched budgets (`max_chunks=8`, `max_context_tokens=3000`) enforced;
  headlines show absolutes with counts/split and intervals only when
  compatible; empty Evaluations state shows "No evaluation results yet"
  with no placeholder charts.
- Controls gate: `data/controls/single_hop.jsonl` holds 30 human-verified
  rows (`human_verified=true`, user-approved 2026-09-29);
  `validate_single_hop_controls(require_verified=True)` passes with
  `n_verified=30` (`freeze_hash=d818023c79d113c8cb5ffd9ce5024bbaf8307261f22273fb59d7da864b8397ff`).
  Split discipline (`validate_split_disjoint`,
  `assert_locked_question_use`) rejects tuning/locked/control overlaps and
  non-exact locked sets.

No hybrid-vs-vector F1, recall, latency, or cost delta is reported here
because the frozen experiment was not run. In particular, there is no
evidence in this report that graph traversal improves (or worsens)
multi-hop quality, and no single-hop non-inferiority claim is made.

## 2. Configuration under test

| Item | Value |
|---|---|
| Embedding model (default) | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine; installed 3.4.1) |
| Reranker model (default) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (installed via `sentence-transformers`; live download not exercised in T15) |
| LLM model | unset (`llm_model: null`); no provider, no prompt run beyond `FakeLlmBackend` |
| Budgets (default) | `max_calls=20`, `max_total_tokens=20000`, `max_cost_usd=null` (call/token caps only) |
| Chunking | `sent-overlap-v1`, `target 180 / max 256` tokens, estimator `char-word-max-v1` + 3 special tokens |
| Retrieval budgets | 30 vector candidates → RRF ≤60 → rerank → final top 8 / 3000 tokens; graph caps 25 edges/node/hop, 100 visited, 40 graph chunks, 2s timeout; triples ≤12 evidence-only |
| Evaluation freeze | `evaluation-v1`, `hotpot-scoring-v1`, seed 42, 2000 replicates, 95% CI, 30 tuning / 120 locked / 30 controls, arms A–D |
| Single-hop controls | `data/controls/single_hop.jsonl`, 30 rows, human-verified 2026-09-29 (`human_verified=true`), freeze hash `d818023c…b8397ff` |

## 3. Hardware and software

| Item | Observed |
|---|---|
| Host | Windows x64, Intel Core i5-10210U (4 cores / 8 threads), ~7.79 GiB RAM, Intel UHD Graphics (CPU-first inference) |
| Interpreter | CPython 3.12.14, isolated `.venv` |
| Docker | 29.8.0, Linux containers via Docker Desktop / WSL2 |
| Neo4j | 5.26.31 Community (`5.26-community` digest-pinned in `compose.yaml`), 1536 MiB container limit, loopback ports 7474/7687 |
| Key packages | `sentence-transformers 3.4.1`, `streamlit 1.64.0`, `neo4j 5.28.6`, `pydantic 2.13.5` (see `requirements-dev.lock` for full hashes) |

Warm/cold latency and indexing wall time are not reported: no timed
end-to-end build was run in T15, and local compute is measured time, not
priced money.

## 4. Corpus caveats

- The experiment design is a pooled 150-question sample from the HotpotQA
  dev distractor release (30 tuning / 120 locked), not the official
  per-question distractor or full-Wikipedia benchmark. Source file verified:
  7405 rows with the documented `_id/question/answer/context/supporting_facts`
  schema (first id `5a8b57f25542995d1e6f1371`). Do not present any fixture
  number as an official HotpotQA score.
- One upstream row (`5ae61bfd…`, Jimmy Butler bridge) ships a supporting
  sentence index (902) outside its 5-sentence paragraph in the official
  release itself. It is quarantined by id + reason in the dataset manifest
  and excluded from sampling; raw bytes are untouched so the exclusion is
  auditable. Two sampled paragraphs contain upstream empty-sentence strings;
  they are preserved in place (indices never shift) and no gold fact cites
  them.
- The shared corpus pools every context paragraph including distractors;
  locked-question paragraphs may be present as retrieval text. This is
  disclosed retrieval-corpus construction; tuning uses no locked answers or
  evidence labels.
- Title variants (e.g. Town v0/v1) are preserved as distinct paragraphs;
  paragraph-id scoping prevents cross-variant false recall.
- Citation validity means the citation resolves to packed evidence, not
  entailment. Syntactically valid extraction claims may still be false;
  the 20-chunk pilot audit is required before any full graph build.
- Token budgets use the offline `char-word-max-v1` upper bound; T04
  re-validates with the pinned model tokenizer and refuses mismatches.
  Reranker pairs are truncated only with a recorded flag.

## 5. Costs

- Query LLM cost for all T15 fixture runs: `unknown` (fake backend, no
  pricing metadata). Exports label this `unknown`, never `$0.000000`.
- Indexing cost (embeddings, extraction calls/tokens, retries): `unknown`
  (no build executed in T15; local compute is unpriced time, not free).
- One live-budgeted smoke from T05 history (local gateway, 1 question) is
  not a T15 result and is not repeated here; no new spend occurred.
- Do not run full extraction without a configured spend/token cap and a
  reviewed 20-chunk pilot estimate; cancellation cannot undo in-flight
  provider charges.

## 6. Negative results and limitations

- Hybrid value is unproven in this handoff: the graph arm was not executed
  on locked questions, so no uplift, parity, or regression can be claimed.
- Single-hop requirement (R10) is half complete: controls are human-verified
  (2026-09-29), but no control arm was run yet — frozen control evaluation
  awaits the dataset build and LLM budget.
- Locked evaluation (R09) is incomplete: no 120-question run, no paired
  intervals on real data, no bridge/comparison strata on real data.
- Dashboard accessibility and narrow-window review were done via component
  contracts and AppTest renders (labels, focus ring, stacked layout, text
  cues), not a physical device pass; the T12c note defers the physical
  320/768/1024/1440px + screen-reader pass to the demo in `docs/DEMO.md`.
- Small samples are noisy: paired intervals below n=30 carry an explicit
  warning; shared evidence limits inference to observed sets, not
  population significance.
- Partial graphs never activate silently: coverage gates require explicit
  `allow_partial=True` and then carry `is_partial=True` with coverage
  statistics.

## 7. Requirement traceability (observed state)

R01 sample — deterministic `prepare_dataset` proven at fixture scale and on
the real dev release (150-sample built 2026-09-29, rerun byte-identical, 1
upstream row quarantined). R02 provenance — chunk round-trip and gold coverage
proven at fixture scale and on the real 150-sample (2000 chunks, empty
sentences tolerated in place). R03 vector baseline — grounded answering +
abstention proven with fake backend; live vector index not rebuilt in T15.
R04 extraction — fixed enums/evidence/caching verified in T07 unit tests,
not re-run live. R05 resolution — collision fixtures pass (T08), not
re-tuned. R06 graph load — idempotent evidence-backed ingest proven in T09
integration, not rebuilt. R07 hybrid QA — bounded merge + fallback proven
in T10 integration and fixture view-models. R08 fair comparison — matched
budgets and compatibility refusal proven at fixture scale; no locked
comparison executed. R09 honest evaluation — fixed denominator, intervals,
cost separation proven at fixture scale; no locked report produced. R10
single-hop — controls human-verified 2026-09-29, gate passes; control runs
outstanding. R11 recovery — persist/resume/duplicate-guard
proven at fixture scale. R12 export — guarded baseline/comparison exports
with hashes and limitations proven at fixture scale. R13 decomposition
(optional, T17) — bounded planner plus tuning-first freeze and additive delta
proven offline in unit tests; no live tuning/locked decomposition comparison
executed (see §9).

## 8. Resume wording (measured outcomes only)

> Built a local GraphRAG lab (Python/Streamlit/Neo4j/SQLite) comparing
> vector retrieval with evidence-backed graph retrieval on a seeded HotpotQA
> sample design (150 questions, 30 tuning / 120 locked, plus 30 human-verified
> single-hop controls). Implemented sentence-preserving chunking with
> provenance checks, cosine vector search, fixed-schema extraction with
> conservative resolution, bounded hybrid retrieval with cross-encoder
> ablations (A–D), paired bootstrap evaluation, and guarded JSON/CSV/
> Markdown exports. Verified offline fixture flow end to end (deterministic
> preparation, resolving citations, fixed-denominator scoring, compatible-run
> intervals, persisted jobs) with unknown costs labeled and failures
> preserved; locked 120+30 evaluation remains outstanding, so no quality
> uplift is claimed.

If hybrid later underperforms on locked data, replace the final sentence
with: "Locked evaluation showed no improvement for the tested graph
configuration; the report preserves all attempts, costs, and limitations."

## 9. Optional decomposition comparison — T17 appendix (R13, E)

Status: implementation complete and verified offline at fixture scale. No
live tuning (30) or locked (120 + 30 controls) decomposition comparison was
executed: `data/raw/` is empty so no 150/30/120 split exists, and
`GRAPHRAG_LLM_MODEL` / `GRAPHRAG_LLM_API_KEY` are unset so no paid planner
or sub-answer call was attempted. No decomposition quality uplift (or
regression) is claimed here, and no frozen A–D result was replaced.

### What was actually measured (offline, `tests/unit/test_decomposition_eval.py`, 10 tests)

- `RunConfig(strategy="decomposition")` builds with matched final budgets
  (top 8 / 3000 tokens, no reranker, no triples); decomposition with a
  reranker is rejected, as are unknown strategies. Compatibility
  (`check_run_compatibility`) holds for baseline-vs-decomposition pairs with
  identical corpus/index, models and budgets, and refuses budget drift.
- `decomposition_to_prediction` maps one T16 `DecompositionResult` to a
  frozen `QuestionPrediction`: candidates from `combined_chunk_ids`,
  packed evidence from `packed_chunk_ids` (subset of combined), citations
  preserved, tokens summed, cost `None` (unknown, never zero), latency from
  `diagnostics.elapsed_ms` unless the caller supplies an observed value.
  Unknown chunks, missing paragraph/title provenance and packed-outside-
  combined raise instead of scoring silently. A live T16 chain
  (`decompose_and_answer` with `FakeLlmBackend` planner/answers over the
  Night Cartographer bridge) maps to a scorable decomposition prediction
  with both bridge chunks combined.
- Tuning-first workflow on a synthetic 4-question set (2 bridge / 2
  comparison, 1 failed `PROVIDER_ERROR` in both arms): compatible paired
  comparison is deterministic across repeated `compare_runs` calls
  (`replicates=200` in tests; frozen default 2000, seed 42, 95% CI);
  failures stay in the fixed denominator; mismatched budgets refuse with
  "refusing unlabeled paired delta".
- `freeze_decomposition_selection` requires both runs on `split="tuning"`,
  a `decomposition` candidate, exact tuning-id coverage and compatible fixed
  fields, then records the shared PRD decision (F1 improvement without
  all-support worsening; cost acceptability stays an operator judgement).
  Non-tuning splits, non-decomposition candidates, id mismatches and
  incompatible budgets are refused with explicit reasons.
- `assert_decomposition_locked_use` requires `split="evaluation"`, the
  frozen strategies, unchanged corpus/index/models/budgets (only the
  question-set hash and split may change) and exact locked-id coverage;
  strategy substitution, budget drift and id drops/adds are refused.
- `summarize_decomposition_incremental` reports the additive delta only:
  paired F1 plus packed paragraph/all-support/sentence recall with
  intervals, observed latency p50/p95 differences (no interval), observed
  query-token deltas and cost deltas with unknown money staying unknown
  (never zero). Inputs are not mutated; warnings state the delta never
  replaces A–D results and that extra planner/sub-answer spend is part of
  the result under matched final budgets.
- `export_comparison` with `decomposition_report_limitations()` reconciles
  on the decomposition pair (per-question `regression.csv` rows average
  exactly to the aggregates, failures at zero); identical re-export is
  idempotent while differing bytes are refused, never overwritten. The
  Evaluations page shows a decomposition banner and exports the same R13
  limitations block when either run uses the decomposition arm.

### How to run the frozen comparison once inputs exist

1. Tuning: run baseline and decomposition arms on the 30 tuning ids with
   identical corpus/index fingerprint, embedding/answer models and final
   budgets (`max_chunks=8`, `max_context_tokens=3000`); score with
   `evaluate_predictions`, compare with `compare_runs`, and inspect
   manifests plus `summarize_decomposition_incremental`.
2. Freeze: call `freeze_decomposition_selection(baseline_eval,
   candidate_eval, tuning_ids, selection_id=...)` and persist the returned
   `DecompositionSelection` (run ids, tuning hash, fixed fields, versions,
   `frozen_at`). Do not proceed to locked data without this record.
3. Locked: run the frozen configuration once on the 120 locked multi-hop
   (plus 30 controls separately) with `assert_decomposition_locked_use`
   gating exact locked ids, then `compare_runs` and `export_comparison`
   with the R13 limitations block into a new output directory. Never
   overwrite an existing A–D artifact directory with differing bytes.

Costs: all T17 fixture runs report query cost `unknown` (fake backends, no
pricing metadata) and indexing cost `unknown` (no build executed); exports
label this `unknown`, never `$0.000000`. Extra decomposition calls/tokens/
latency are logged per question (`total_llm_calls`, input/output tokens,
`elapsed_ms`) and surfaced in the incremental summary.
