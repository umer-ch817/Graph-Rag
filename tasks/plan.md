# Master plan — GraphRAG Lab

## Outcome

Build a local research and portfolio app demonstrating whether evidence-backed graph traversal improves multi-hop QA over vector retrieval. Users can prepare a dataset, build an index, compare answers and inspect the evidence, then export reproducible results.

This document and its linked specifications define the project plan. Core tickets T01–T12c and T14–T15 are implemented and verified at fixture scale; T13 code is complete pending human single-hop review and live frozen runs (see todo.md). The detailed execution record lives in todo.md.

## Capability map

| Module | Responsibility | Depends on |
|---|---|---|
| corpus | Sampling, paragraph deduplication, sentence-preserving chunks, manifests | — |
| vector-qa | Embeddings, vector retrieval, grounded answer generation | corpus |
| graph-index | Structured extraction, conservative entity resolution, provenance | corpus |
| hybrid-qa | Entity linking, bounded traversal, merged retrieval, reranking | vector-qa, graph-index |
| evaluation | Frozen runs, scoring, comparisons, cost reports | vector-qa; hybrid-qa for graph runs |
| dashboard | Setup, questions, evidence inspection, experiment views | corpus, vector-qa, graph-index, hybrid-qa, evaluation |
| decomposition | Optional sequential sub-question retrieval | hybrid-qa, evaluation |

Provider contracts and schemas live in BACKEND-SCHEMA.md. Detailed module behavior lives in TRD.md. Stable module ids above also identify the ticket ownership boundaries.

Build order: corpus → vector-qa → baseline evaluation → graph-index → hybrid-qa → paired evaluation → dashboard completion → optional decomposition. Add small usable dashboard views alongside the relevant slices.

## Architecture decisions

1. Streamlit is the proposed interface because it keeps the local project in Python. A separate JavaScript frontend and HTTP API would expand the scope without improving the research experiment.
2. Use Neo4j for graph and chunk vectors, SQLite for local durable job/run metadata, and JSONL artifacts for reproducibility. Do not add Redis or a distributed task queue.
3. A single CLI worker claims jobs from SQLite. The dashboard enqueues work and polls persisted status; browser reruns must not launch duplicate indexing jobs.
4. Preserve paragraph and sentence provenance through chunking, extraction, resolution, retrieval and answer citations.
5. Treat reranking as a confounder: measure vector, vector+rerank, hybrid+rerank, and hybrid+rerank+triples separately.
6. Conservative entity linking is preferred to merging unrelated people with common names. Similar embeddings alone cannot establish identity.
7. Build new immutable index snapshots and activate only after validation. Do not overwrite the usable baseline during graph extraction.

## Milestones and effort

Estimates are engineering effort, not runtime guarantees. Downloads, extraction calls, retries and CPU inference add wall-clock time.

| Milestone | Deliverable | Planning allowance |
|---|---|---|
| M1 | Reproducible corpus, vector QA and baseline scoring | 4–6 hours |
| M2 | Validated extraction, entity resolution, graph hybrid retrieval | 5–8 hours |
| M3 | Complete dashboard, evidence inspection, paired evaluation/export | 5–8 hours |
| M4 | Optional decomposition experiment | 2–4 hours |

Core local app: approximately 14–22 hours plus model runtime. The original 6–8 hours is a prototype target: a small smoke corpus, basic CLI, baseline and first hybrid answer. Do not describe that milestone as a completed 150-question study.

## Delivery order and checkpoints

- T01–T03: foundation, sample and chunks. Checkpoint: deterministic corpus and verified gold mappings.
- T04–T06: vector QA and baseline score export. Checkpoint: every prediction is traceable and scoring fixtures pass.
- T07–T09: extraction, resolution and graph load. Checkpoint: bridge fixture works; unrelated names stay separate; graph evidence valid.
- T10–T12: hybrid retrieval, ablations and comparison UI. Checkpoint: budgets equal across arms; failure fallback visible.
- T13–T15: controls, evaluation dashboard, reproducibility handoff. Checkpoint: locked results and honest report.
- T16–T17: optional decomposition and its evaluation. Keep only on tuning evidence, then evaluate the frozen selection once.

Tickets and checkboxes are maintained only in [todo.md](todo.md).

## User-directed execution

The project is divided into 19 individually assignable tickets: 17 core tickets (including T12a, T12b and T12c) and 2 optional extension tickets. The user assigns one id at a time. Complete and verify that ticket, persist its execution record in todo.md, report the result, and stop before starting another ticket. Checkpoints are included in the ticket immediately preceding them. See the ordered index and execution protocol in todo.md.

## Risks and mitigations

| Risk | Effect | Planned control |
|---|---|---|
| False entity merges | Plausible but incorrect bridges | Same-type matching, lexical/context confirmation, reversible mapping, collision fixtures |
| LLM unsupported relations | Misleading evidence | Evidence sentence ids and text validation; invalid output quarantined |
| Generic/high-degree nodes | Large irrelevant candidate sets | Per-seed/per-hop caps, fan-out filtering, timeout, logged truncation |
| Missing relations | Graph arm adds no value | Report extraction coverage and unresolved links; retain vector candidates |
| Reranker removes bridge chunks | Missing support in final context | Measure recall before and after reranking; tune evidence diversity on tuning set only |
| Gold leakage | Invalid score improvements | Separate label artifacts; retrieval receives question and corpus only |
| Small sample | Noisy improvement | Paired intervals, counts, per-type results, restrained claims |
| API cost grows | Incomplete indexing | Pilot estimate, cached chunks, bounded retries, user-configured hard cap |
| Schema too narrow | Facts cannot be represented | Audit 20 pilot chunks; freeze revised enums before locked evaluation |
| CPU latency | Sluggish demo | Cache models and completed runs; display durations, measure warm/cold separately |

## Decisions deferred to implementation setup

- LLM provider/model and a hard spend limit. Pilot 20 chunks before any full extraction run.
- Available RAM/GPU and exact Docker/Python/model revisions; record a tested lockfile and image digest.
- Final entity-resolution thresholds calibrated on reviewed tuning examples.

These do not block writing the plan. They must be resolved before paid calls or performance claims. Deployment and API credentials are not assumed available.

## Definition of done

All core tickets pass their acceptance checks; the local app completes the dataset → index → answer → evaluation flow; saved run manifests reproduce settings; multi-hop and independently labeled single-hop results are reported; actual indexing cost is disclosed; results are never invented. A measured lack of improvement does not block delivery, but it changes the conclusion and resume wording.
