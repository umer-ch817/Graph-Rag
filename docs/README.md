# GraphRAG Lab — planning package

Status: core implementation complete through T15 at fixture scale (T01–T12c, T14 done; T13 code done pending human control review and live frozen runs). See [environment verification](ENVIRONMENT.md), [measured results](RESULTS.md), [setup guide](DEMO.md) and [ticket status](../tasks/todo.md).

Confirmed scope: a local portfolio application with a simple dashboard and reproducible evaluations. Prepared 2026-09-29; handoff updated 2026-09-29 for T15.

## Read in this order

1. [Master plan](../tasks/plan.md): scope, capability map, sequencing, estimates, risks and decisions.
2. [PRD](PRD.md): users, requirements, success criteria and exclusions.
3. [TRD](TRD.md): architecture, retrieval, extraction, evaluation and operational design.
4. [UI/UX design](UI-UX.md): screens, wireframes, states and accessibility.
5. [App flow](APP-FLOW.md): user journeys and execution state transitions.
6. [Backend schema](BACKEND-SCHEMA.md): graph, application records and service contracts.
7. [Implementation tickets](../tasks/todo.md): ordered work with acceptance criteria and verification.
8. [Setup guide](DEMO.md): clean-environment setup and demo checklist (T15).
9. [Measured results](RESULTS.md): models/hardware/corpus caveats/costs with no fabricated metrics (T15).

## Proposed defaults

- Python service layer, Streamlit dashboard, CLI, Neo4j Community in Docker.
- HotpotQA development distractor data: 150 sampled multi-hop questions, split into 30 tuning and 120 locked evaluation questions.
- Separate 30-question human-verified single-hop control set drawn from the same corpus; never present it as an official HotpotQA single-hop split.
- A shared corpus containing all context paragraphs, including distractors, from the sample. This is a pooled sample experiment, not the official per-question distractor or full-Wikipedia benchmark.
- CPU-compatible embedding and reranking models; GPU optional. Exact model revisions and dependency versions pinned during setup.
- Hosted structured-output LLM through a configurable adapter; provider, model, credentials and monetary budget remain runtime choices.
- No authentication, public deployment, arbitrary uploads, multi-user features or managed cloud services in MVP.

## Evidence and claims

HotpotQA provides multi-hop questions and sentence-level supporting facts; the evaluation repository defines supporting-fact identifiers as title and sentence index. Preserve that mapping instead of reducing all evidence to document titles. [Dataset](https://hotpotqa.github.io/), [official evaluator](https://github.com/hotpotqa/hotpot).

Neo4j supports vector indexes; Community embeddings should use numeric list properties. Pin a tested server version and use that version's index syntax. [Vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/), [vector property types](https://neo4j.com/docs/cypher-manual/current/values-and-types/vector/).

Sentence Transformers documents the retrieve-then-rerank approach. The graph experiment adds a candidate source to this pipeline. [Retrieve and rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html).

All quality improvements and latency/cost figures are hypotheses until measured. A negative result is a valid completed experiment.
