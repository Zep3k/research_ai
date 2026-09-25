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
- Durable workstreams with retained failure states
- Workstream artifacts and bounded reviews
- Deterministic theorem delta
- Deterministic entity/workstream graph-neighborhood context
- Ordered schema-version ledger and V0.1 backfill
- Workstream-aware API cost ledger

## V0.2 evaluation gate

Evaluate the system on at least three old research directions with known outcomes, including one false or flawed conjecture. For each case, record whether the graph and delta exposed a hidden assumption, located a decisive known result, prevented a repeated failed approach, or changed the next action. Compare against the retained one-shot-style `investigate` result. Do not infer success from response polish.

## One next milestone (requires approval)

Build one graph-backed `attack` workflow that consumes an explicitly selected conjecture/workstream context, emits only quarantined or speculative candidate artifacts through the deterministic write gate, and is evaluated on the known cases above. Do not add embeddings, autonomous loops, reviewer swarms, or new infrastructure as part of that milestone.
