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

## Development freeze: evaluation gate

Freeze implementation. Reconstruct multiple historical states from one completed theoretical-research project, including a direction before it failed, a proof before its flaw, a direction before a decisive theorem was found, and a successful direction before its key insight. Compare ordinary frontier-model interaction with this graph-backed attack using the same model where possible. Measure actual-issue recovery, assumption recovery, decisive counterexamples, useful false alarms, changed next actions, and cost per useful result. Do not choose another implementation milestone until these results exist.
