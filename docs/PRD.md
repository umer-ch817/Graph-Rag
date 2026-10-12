# Product requirements — GraphRAG Lab

## Problem and users

Answers to bridge and comparison questions may depend on facts spread across paragraphs. A top-k vector search can retrieve one relevant paragraph while missing another. This project tests whether an extracted knowledge graph closes that gap and makes its evidence inspectable.

Primary user: the project author running and evaluating experiments locally. Secondary user: a reviewer watching a portfolio demo or reading the exported report.

## Core experience

Prepare a reproducible corpus, run a vector baseline, build the graph, and compare the same question across retrieval strategies. Inspect source paragraphs, entity connections and costs. Run a frozen evaluation and export both aggregate and per-question results.

## Requirements

| ID | Requirement | Acceptance condition |
|---|---|---|
| R01 | Reproducible sample | Seed, source checksum, question ids, split and corpus fingerprint saved; repeated preparation returns identical ids |
| R02 | Provenance-preserving chunks | Every sentence maps back to source title and sentence index; distractors retained |
| R03 | Vector baseline | Answer, retrieved context, citations, timing and usage persisted without graph access |
| R04 | Fixed-schema extraction | Unknown entity/relation types and dangling references rejected; failures visible and resumable |
| R05 | Entity resolution | Same-type candidates evaluated; merge decisions and original mentions retained; ambiguous cases remain separate |
| R06 | Graph indexing | Chunk mentions and directed relations retain valid evidence; repeat ingest creates no duplicates |
| R07 | Hybrid QA | Vector and bounded graph candidates merged and reranked; final sources and triples inspectable |
| R08 | Fair comparison | Identical question ids, corpus, answer model and context budget; reranker ablation included |
| R09 | Honest evaluation | EM, answer F1, evidence recall, all-support coverage, latency, failures and cost exported |
| R10 | Single-hop control | Separate human-reviewed questions and supporting facts; reported independently |
| R11 | Recovery | Interrupted jobs resume from validated artifacts; incomplete indexes never shown as ready |
| R12 | Export | Machine-readable results plus a Markdown summary include configurations and limitations |
| R13 | Decomposition, optional | Bounded sequential retrieval compared on tuning data before final selection |

## Measurement and success

Engineering success: all core requirements function and results can be audited.

Research hypotheses, not guarantees: hybrid retrieval increases multi-hop answer F1 and all-support coverage; single-hop answer quality remains comparable. Report observed paired deltas and uncertainty. Do not set a favorable result as a completion criterion.

Proposed decision rule: choose the more complex configuration only if its tuning-set F1 improves, its all-support coverage does not worsen, and its costs are acceptable. Freeze the choice before locked evaluation. For the single-hop control, flag an F1 decrease greater than 2 percentage points as a regression to investigate, with sample-size caveats; this is a provisional tolerance, not evidence of non-inferiority.

Report EM/F1 on a 0–100 scale and deltas in percentage points. Always show sample size. Retrieval metrics must state k and whether measured at candidate or final-context stage.

## Scope

Core: one local operator, HotpotQA sample, experiment setup, baseline and hybrid answers, graph evidence inspection, run comparison, durable artifacts and exports.

Optional: two-step decomposition; second dataset adapter; richer graph visualization after core evidence tables work.

Excluded: authentication, collaborative workspaces, arbitrary document uploads, internet search during answers, public hosting, mobile app, production scaling, model training and claims of official benchmark performance.

## Operational expectations

- Credentials loaded from environment variables; never stored in exports or logs.
- Local services bind to loopback by default.
- No full extraction run without a configured spend limit and pilot estimate.
- Controls show unavailable dependencies, failed jobs and unsupported answers explicitly.
- Index and experiment identifiers are visible in results to prevent accidental comparisons across incompatible data.
- Keyboard navigation, text alternatives for graph information and readable contrast are part of the dashboard acceptance criteria.

## Portfolio output

Deliver a runnable repository, setup guide, architecture diagram, short demo walkthrough and a measured evaluation report. The resume sentence is generated only after results exist and names the sampled setting. If hybrid underperforms, describe what was built and measured without claiming an uplift.
