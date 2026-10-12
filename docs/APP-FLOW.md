# App flow

UI navigation follows UI-UX.md revision 2: first-run users start on Overview; returning users resume their last page/run once vectors are ready. Citation clicks open an inline evidence inspector without discarding compared answers. Full Evidence review restores the prior question/run on return. Example selection edits the draft only; Run comparison explicitly submits it. Opening an evaluation row reads saved results and never triggers a new LLM request.

## First successful experiment

```mermaid
flowchart TD
  A[Open local app] --> B[Check Neo4j, worker and model configuration]
  B --> C[Prepare seeded sample and freeze manifest]
  C --> D[Build chunk vectors]
  D --> E[Ask a baseline question]
  E --> F[Run baseline tuning evaluation]
  F --> G[Extract 20 pilot chunks and audit]
  G --> H[Set budget and build graph]
  H --> I[Compare baseline and hybrid evidence]
  I --> J[Tune only on tuning questions]
  J --> K[Freeze configurations and single-hop controls]
  K --> L[Run locked evaluation arms]
  L --> M[Inspect regressions and export report]
```

The graph can fail to improve results. The terminal outcome is an auditable report, not necessarily a positive delta.

## Question request

1. Select active ready snapshot, strategies and question.
2. Validate question length and configuration; persist a request/job id.
3. Load immutable strategy settings; retrieve vector candidates.
4. For graph modes, link question entities and expand bounded graph frontiers.
5. Merge candidates, optionally rerank, pack context and eligible triples.
6. Generate structured answer; validate citations and persist timings/usage.
7. Display answer cards. Citation selection opens Evidence for the same run id.

If graph linking returns no seeds, record vector fallback. If answer generation fails, persist failed status rather than a fabricated answer. If evidence is insufficient, persist an abstention as a valid response.

## Index job state machine

```mermaid
stateDiagram-v2
  [*] --> queued
  queued --> running: worker claims job
  queued --> cancelled: cancel before start
  running --> completed: validation passes
  running --> failed: retries exhausted
  running --> paused_budget: cap reached
  running --> cancelling: user requests cancellation
  cancelling --> cancelled: safe boundary
  running --> interrupted: stale worker lease
  interrupted --> queued: explicit resume
  failed --> queued: retry failed items
  paused_budget --> queued: revised budget
  cancelled --> queued: resume checkpoint
```

Resume reuses validated cache entries and records a new job attempt. Ready snapshot activation occurs only after index validation. Existing ready snapshots remain selectable while another build is incomplete. Cancellation stops new calls; it cannot reverse completed calls or their charges.

## Evaluation flow

Choose frozen question set and compatible arms → persist run manifest → enqueue each arm → persist per-question results → compute metrics over the fixed denominator → compute paired comparisons → export immutable report.

Report prerequisites: same question ids, corpus, answer model and context budget; any mismatches disable direct paired comparison and explain why. A later prompt or resolver change requires a new versioned experiment. Do not overwrite old scores.

## Recovery

- Neo4j unavailable: keep queued work and show connectivity error; retry once service returns.
- Model download unavailable: setup fails clearly; no extraction calls begin.
- Malformed extraction: bounded retry, then quarantine with validation reason.
- Browser closes: worker continues; reopening shows persisted job state.
- Worker stops: heartbeat expires; mark interrupted and offer checkpoint resume.
- Budget reached: stop scheduling calls, preserve completed artifacts and show revised estimate.
