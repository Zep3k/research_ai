# Evidence-driven roadmap

## V0.1 — retained compatibility layer

- Local SQLite project state, ideas, and PDF import
- OpenAlex discovery client
- OpenAI and Anthropic provider adapters
- Per-call cost ledger and explicit local monthly budget
- One adversarial `investigate` workflow

V0.1 remains usable during migration, but its report JSON and recent-history prompt no longer define the architecture.

## V0.2 — typed research memory

- Typed entities with generic attributes
- Explicit validated relations in SQLite
- Precise provenance locators and claim-to-source links
- Separate lifecycle status and epistemic trust state
- Deterministic trust/write gate
- Durable workstream lifecycle with retained failure artifacts and history
- Workstream artifacts and bounded reviews
- Deterministic theorem delta
- Deterministic entity/workstream graph-neighborhood context
- Ordered schema-version ledger and V0.1 backfill
- Workstream-aware API cost ledger

## First graph-backed attack workflow

- Identifiable-origin requirement for sourced state
- Unambiguous workstream execution lifecycle
- Explicit model-facing epistemic context partitions
- Syntax-only attribute-key normalization
- Exactly one graph-scoped provider call per attack
- Strict structured output and reference validation
- Quarantined typed artifacts through the deterministic write gate
- Bounded review, lifecycle completion/error, and workstream-linked cost
- No automatic retrieval, provider debate, recursion, or new infrastructure

## First graph-backed develop workflow

- Dedicated durable `develop` workstream type
- Exactly one graph-scoped provider call per development pass
- Strict coverage of consequences, intermediate lemmas/components, parameter analysis, proof obligations, and distinct branches
- Explicit promising/blocked/failed/unresolved branch outcomes
- Failed branches retained as quarantined `FailedApproach` objects
- All generated artifacts quarantined through the deterministic write gate
- Workstream-linked cost accounting and completion/error lifecycle
- No automatic relations, proof claims, retrieval, debate, recursion, or new infrastructure

## Bounded adaptive research controller

- Deterministic graph-state operation selection among develop, attack, synthesize, and prove
- Exactly one provider call per persisted iteration; no model-planner call
- Durable operation rationale, target, progress, duplicate count, outcome, and stop reason
- Strict graph-scoped structured output with quarantined artifacts and relations
- Synthesis consumes at least two linked artifacts for one specific obligation
- Prove/attack eligibility gates require precise candidates
- Failed/blocked branches and new proof obligations remain durable graph objects
- Deterministic material-key and lexical duplicate rejection
- Early stops for bounded attack success, terminal branches, stagnation, human judgment, budget, and call limit
- No retrieval, provider debate, embeddings, or unbounded autonomy

## Development freeze: evaluation gate

Freeze implementation. Reconstruct multiple historical states from one completed theoretical-research project, including a direction before it failed, a proof before its flaw, a direction before a decisive theorem was found, and a successful direction before its key insight. Compare ordinary frontier-model interaction with the graph-backed single-step workflows and bounded controller using the same model where possible. Measure actual-issue recovery, assumption recovery, decisive counterexamples, recovered intermediate insights, useful false alarms, changed next actions, stagnation, and cost per useful result. Do not choose another implementation milestone until these results exist.
