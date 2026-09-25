# Coding-agent instructions

Goal: build a personal theoretical-research workbench that improves research decisions and mathematical reliability. It is not an autonomous scientist.

Constraints:
1. Keep durable research state provider-independent.
2. Treat the typed SQLite research graph, not chat history, as the architectural center.
3. Separate research lifecycle status from epistemic trust state.
4. Never promote an LLM-generated claim to `sourced` without persisted provenance.
5. Log every model call with provider, model, purpose, tokens, estimated cost, timestamp, run ID, and workstream ID when applicable.
6. Keep monthly budget enforcement explicit and local.
7. Preserve precise source metadata for literature-derived claims.
8. Preserve failed approaches, counterexamples, obstructions, and failed/abandoned workstreams.
9. Prefer small graph-scoped workflows over giant prompts.
10. Keep model IDs and prices centralized.
11. Never commit `.env` or `.theory/`.
12. Do not add autonomous agent swarms, a vector database, or a graph database without evaluation evidence.
13. Never label a theorem or proof "verified" merely because an LLM found no flaw.

Current milestone: evaluate whether typed objects, theorem delta, durable workstreams, and deterministic graph context prevent wasted research effort on known cases. The V0.1 `investigate` workflow remains only as a compatibility workflow; it no longer defines the architecture.
