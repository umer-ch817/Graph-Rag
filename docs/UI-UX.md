# UI/UX design — GraphRAG Lab

Design revision 2. This revision defines the implementation target for T12b, T12c and T14. The screen contracts below refine the original outline; no application UI has been implemented. Direction is based on this project's evidence-comparison workflow, without an external visual reference or copied product layout.

## Design contract

The central experience is **ask → compare → inspect evidence**. Setup should guide a first-time user to that experience; returning users should reach it immediately. Prioritize the question, answer and evidence over system statistics. Use a restrained editorial style: warm white surfaces, dark ink headings, thin borders, teal actions and compact metadata. Reserve strong color for actions and meaningful status.

Five destinations remain: Overview, Dataset & Index, Ask & Compare, Evidence, Evaluations. Overview is the first-run landing page until vectors are ready. Afterward, return to the last visited page and run; provide an obvious Ask & Compare action. Evidence is both an inline inspector and a full page for deeper review, using the same selected run/source state.

## Visual system

| Token | Value | Use |
|---|---|---|
| background | #F6F7F5 | Application canvas |
| surface | #FFFFFF | Forms, results and source reader |
| surface-muted | #EDF1EE | Metadata strips and secondary sections |
| text-primary | #18251F | Headings and body |
| text-secondary | #4C5E54 | Descriptions, labels and timestamps |
| border | #D6DED8 | Decorative separators; not sole control boundary |
| control-border | #728579 | Input outlines where needed |
| action | #176348 | Primary buttons and links |
| action-hover | #104B36 | Pointer/active state |
| focus | #245FC7 | Visible focus outline |
| vector | #295D91 | Vector label and chart series |
| graph | #176348 | Graph label and chart series |
| warning | #865500 | Warning text on pale amber |
| danger | #A32932 | Error text and destructive controls |

White text on action buttons. Validate actual rendered foreground/background pairs before acceptance, including disabled, hover and focus states. Chart series always carry text labels or different markers.

Typography: system sans-serif for prose; system monospace only for ids and technical metadata. Page title 28/36px, section title 20/28px, answer text 24/32px, body 16/24px, supporting metadata 14/20px. Use tabular numerals for metrics and costs. Avoid all-caps body labels.

Spacing scale: 4, 8, 12, 16, 24, 32, 48px. Form controls at least 40px high, with 44px target size preferred for primary actions. Panel radius 8px, input/button radius 6px, pill radius reserved for short status labels. Use borders rather than pervasive shadows. Comfortable source reading width: approximately 65–80 characters per line.

Layout at 1440px: approximately 224px navigation, 32px main padding, 24px panel gaps. Main content maximum width 1440px; prose sections remain narrower. Use native Streamlit layout primitives where possible; exact sidebar width is a visual target, not a requirement for brittle DOM overrides.

## Revised screen contracts

### Overview: one next step

Show a welcome heading, one sentence explaining the current state, and a single primary action. Below it, show a numbered progress list: Corpus → Vector baseline → Graph → Evaluation. Each step includes status and the reason it is incomplete. Secondary region contains recent runs, each with question excerpt, strategy, time and result status. Dataset statistics stay in a compact details row.

First run copy: “Start with a reproducible question set.” Action: “Prepare dataset.” Returning copy: “Your vector index is ready.” Action: “Ask a question.” Do not show fake scores or empty metric tiles.

### Dataset & Index: guided preparation

Present four numbered sections: Dataset, Vector index, Graph pilot & build, Ready to compare. Keep completed sections summarized but expandable. Each section has one main action and prerequisites stated inline.

Dataset defaults are visible; advanced sampling and model settings sit in an expander. Preview a few actual paragraphs and the resulting split before preparation. For indexing, show progress, processed/total chunks, elapsed time, failed chunks and budget in a single progress panel. Put Cancel as a secondary action. After a failure, preserve settings and show “Retry failed chunks”; do not send the user back to the beginning.

The graph pilot summary separates estimated full-build cost from actual spend. Budget labels include currency or token/call units. Changing a completed configuration shows “Creates a new snapshot” next to the action. Snapshot ids and schema revisions stay under Details.

### Ask & Compare: the primary workspace

```text
┌──────────────────┬─────────────────────────────────────────────────────┐
│ GraphRAG Lab     │ Ask & Compare                     Dataset · Ready   │
│                  │ See what each retrieval method found.               │
│ Overview         │                                                     │
│ Dataset & Index  │ QUESTION                                            │
│ Ask & Compare  ◀ │ [ Where was the director of … born?              ]  │
│ Evidence         │ Examples: Bridge · Comparison                       │
│ Evaluations      │ [Vector vs Hybrid ▾] [Run comparison] [Settings ▾]   │
│                  │                                                     │
│                  │ Vector                     Hybrid                   │
│                  │ ─────────────────────      ─────────────────────    │
│                  │ Answer or abstention       Answer or abstention     │
│                  │ [1] [2] Sources            [1] [2] Sources          │
│                  │ Duration · source count    Duration · source count  │
│                  │                                                     │
│                  │ EVIDENCE                                            │
│                  │ [Sources] [Connections] [Retrieval details]         │
│ Active dataset   │ Source list          │ Selected paragraph           │
│ Worker · Ready   │ Title · origin       │ Highlighted cited sentences  │
└──────────────────┴─────────────────────────────────────────────────────┘
```

Wireframe contains no measured results. Default comparison is vector versus hybrid with reranking/triples; the precise arm names appear in each result header. Advanced selector exposes all A–D configurations with plain-language explanations. Users can display one arm or compare two at a time; batch evaluation handles wider comparisons.

The question input is multiline with a visible label. Submit through an explicit form action to prevent requests on every rerun. Keep the question and chosen settings after completion. Selecting another example changes the draft but does not start a paid request.

Result panels share the same structure: strategy, result status, short answer, source citations, duration/source count, Details. Align panel tops; let answer length wrap naturally. Show an answer when that arm finishes without waiting for the other. The unfinished arm displays its actual stage: Retrieving evidence, Reranking, or Writing answer. Never show a fabricated progress percentage.

The first citation opens the inline source inspector below the answers. Preserve both answers, selected strategy, question and scroll context. Full-screen evidence review is a secondary link. Returning from it restores the prior run. Use links/buttons with descriptive accessible names, such as “Open source 2: Example title, sentence 4.”

An identical answer is not labeled a tie or a success without a known reference. A different answer is not automatically a graph improvement. Single-hop controls and locked evaluation examples stay out of the default demo picker before evaluation is frozen; use tuning examples for interactive exploration.

### Evidence: follow a claim to its source

At wide widths, allocate roughly one-third to the source list and two-thirds to the reader. Each source row includes title, paragraph/sentence range, Vector / Graph / Both label, and whether it was used in final context. Rank and raw scores are available under Details, not dominant text.

Reader header contains title and provenance. Highlight the exact cited sentences while leaving surrounding text readable. Label a sentence as cited or retrieved without implying it is verified true. The user can switch between arms while retaining the source selection if that source exists in both.

Connections view shows a readable sequence or relation table: entity → relation → entity → evidence. Each evidence link selects its source text. Graph visualization remains optional and secondary to this table. State “No linked entities; vector evidence shown” when graph linking fails. Show explicit truncation summaries, including traversal caps and hidden graph nodes.

### Evaluations: answer the research question first

Start with run selection and compatibility status. Display three headline comparisons: Answer F1, All-support coverage, and per-question latency. Each shows both absolute values; deltas appear only for compatible runs. Counts and dataset split sit next to the headline, not in a tooltip. Show uncertainty beside quality deltas; do not use celebratory coloring when the result is inconclusive.

Below: Multi-hop / Single-hop control tabs, full metrics table, and a failure-analysis table. Column priority is Question → Baseline score → Hybrid score → Difference → Status. Offer filters Improved, Worsened, Unchanged, Failed and Abstained; allow multiple relevant labels per result. A row opens paired answers and evidence from the saved run, never reruns the question implicitly.

Separate Indexing cost from Query cost. Clearly label estimated monetary values and unpriced local compute. Export action stays visible beside run selection and offers CSV, JSON and Markdown. Before any run: “No evaluation results yet” with “Configure evaluation”; no placeholder charts.

## Interaction persistence and accessibility acceptance

- Persist active dataset, run, strategy pair and selected source in session/navigation state. Durable job and run records remain authoritative after refresh. Local deep links may identify run/source ids; they are not publicly shareable URLs.
- Dataset switches preserve the draft question but clear incompatible selected runs with an explanation. Filters affect views, never the saved evaluation denominator.
- Error messages identify what failed, what remains saved and the next action. Retry is explicit and creates a traceable attempt.
- Success/cancel/error updates are available to assistive technology; routine polling must not repeatedly steal focus or announce every counter change.
- At 1024px collapse secondary metadata; below 900px stack answer panels and source list/reader. At 768px use compact navigation; at 320px controls wrap and prose has no horizontal overflow. Wide metric tables may scroll inside a labeled region.
- Keyboard users can submit, select a strategy, open a citation, return to results, filter evaluations and export. Hover is never required. Source highlights have a text cue.
- Respect reduced motion. Optional transitions are 120–180ms; there is no auto-moving graph or animated background.
- During implementation, test 320, 768, 1024 and 1440px, actual contrast, screen-reader labels, focus order and all failure states. Record issues honestly; this document does not claim a rendered accessibility pass.

## Streamlit implementation boundary

Build this direction using shared theme tokens, forms, columns, containers, tables and existing service contracts. Prefer an inline inspector over a custom sliding drawer. Do not add a React frontend, custom JavaScript focus system or new graph component merely to match the wireframe. If a required interaction cannot be delivered with supported components, document the limitation and propose the smallest design adjustment in the assigned ticket.

The screen inventory below provides supporting requirements for revision 2.

## Product character

A calm experiment workbench emphasizing answer quality, source evidence and reproducibility. Working title: GraphRAG Lab. Avoid a chat-only interface: the primary task is comparing retrieval approaches, not maintaining conversations.

Desktop-first, usable at 1280px; panels stack on narrow windows. Apply the revision 2 visual tokens and typography above. Never communicate strategy or status using color alone. Prefer native Streamlit components; custom CSS is limited to stable layout/spacing needs.

## Navigation

Sidebar: Overview → Dataset & Index → Ask & Compare → Evidence → Evaluations. Footer shows active corpus, index status and model configuration. Settings appear in Dataset & Index rather than a separate administration area.

## 1. Overview

Purpose: tell the user what is ready and the next useful action.

- Dataset count, chunks, graph entities/relations and last completed evaluation.
- Checklist: prepare dataset, build vectors, run baseline, build graph, compare, evaluate.
- Primary action changes with readiness; no empty dashboard full of zero-valued metrics.
- Missing Docker, models or credentials appears as a clear setup state with a remedy.

## 2. Dataset & Index

Controls: sample size 100–200 (default 150), seed, tuning/evaluation split, model selections, spend/token cap and worker status. Show a manifest preview and sample paragraphs before indexing.

Two stages: Build vector index, then Build graph. Graph pilot estimates cost for the full corpus and shows the assumptions. Progress uses completed/total chunks, elapsed time, retry count, failures and spend; no fabricated ETA. Offer cancel, resume and inspect failures. Changing a fingerprinted setting creates a new snapshot rather than mutating a completed run.

## 3. Ask & Compare

```text
GraphRAG Lab                         Corpus: hp-150   Index: Ready

Question [Where was the director of ... born?                ]
Examples [Bridge | Comparison]  (tuning questions only)
Strategies [Vector] [Vector + rerank] [Hybrid]     [Compare]

+---------------------------+--------------------------------+
| Vector                    | Hybrid                         |
| Answer / insufficient     | Answer / insufficient          |
| Source citations          | Source citations               |
| Duration · retrieved docs | Duration · graph additions     |
+---------------------------+--------------------------------+
Evidence [Sources] [Connections] [Retrieval details]
Source title · sentence range · vector / graph / both
Text with cited sentences highlighted
```

Compare defaults to vector and hybrid; advanced controls expose ablation arms. Question and run settings remain visible while waiting. Do not show a percentage confidence score. Show insufficient evidence when returned; failures display a retry action and trace id.

Citations open the exact source sentence range in Evidence. Graph connections are labeled “Retrieved connections,” not private model reasoning. Gold answers remain hidden during ordinary Q&A and appear only in evaluation review.

## 4. Evidence

Source list with title, chunk id, original sentence ids, retrieval origin and rank. Selected item shows text with preserved sentence boundaries. A relation table shows source entity, relation, target and evidence citation; this is the required accessible representation.

An optional bounded node-link view shows only the selected run's retrieved subgraph, capped at 30 visible nodes. Selecting a relation opens its evidence. Overflow is explicit; do not render the whole knowledge graph. Unresolved question entities and traversal truncation appear in the details panel.

## 5. Evaluations

Select compatible completed runs. Columns: strategy, sample size, answer EM, answer F1, paragraph recall, all-support coverage, sentence-support recall, p50/p95 latency and query cost. Multi-hop and single-hop controls use separate tabs. One-time indexing cost has its own panel.

Show paired deltas and intervals beside absolute scores. Unrun metrics display “Not measured.” Partial/failed runs have visible status and denominators. Per-question table can filter improved, worsened, unchanged, failed and abstained cases. Selecting a row opens the paired answers and sources. Export JSON, CSV and Markdown report.

## Interaction and accessibility

- Every control has a visible label; keyboard navigation follows reading order with visible focus.
- Status messages use text and icons, not color alone. Verify normal text contrast at least 4.5:1.
- Source tables and relation lists provide the same information as visual graphs.
- Loading states preserve the submitted question; duplicate submission is disabled for an active job.
- Below 900px, answer cards stack; comparison tables allow horizontal scrolling.
- Exports and source selections must work without hover interactions.

## State coverage

| State | Presentation | Next action |
|---|---|---|
| No corpus | Explanation and prepare button | Prepare sample |
| Vector ready, graph absent | Vector modes enabled; graph unavailable with reason | Build graph |
| Worker unavailable | Job remains queued; worker command shown | Start worker |
| Indexing | Durable progress and budget | Cancel or wait |
| Partial extraction | Failed chunk count and partial-index warning | Retry failures |
| Entity not linked | Vector fallback badge | Inspect linked entities |
| No evidence | Insufficient evidence message | Try another question |
| Provider failure | Sanitized error and retained request | Retry |
| Completed evaluation | Metrics and artifact links | Compare/export |

Acceptance: manually walk each state using fixtures, verify keyboard-only source navigation, and confirm charts/tables never imply unmeasured improvement. This is a design specification and wireframe, not a built interface.
