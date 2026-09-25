# Coding-agent instructions

Goal: build a personal theoretical-research workbench that improves research decisions and mathematical reliability. It is not an autonomous scientist.

Constraints:
1. Keep durable research state provider-independent.
2. Treat the typed SQLite research graph, not chat history, as the architectural center.
3. Separate research lifecycle status from epistemic trust state.
4. `sourced` requires provenance with an identifiable paper or external origin; it never means theorem-verified.
5. Treat incomplete locators as notes, not sufficient support for `sourced` state.
6. Log every model call with provider, model, purpose, tokens, estimated cost, timestamp, run ID, and workstream ID when applicable.
7. Keep monthly budget enforcement explicit and local.
8. Preserve precise source metadata for literature-derived claims.
9. Preserve failed approaches, counterexamples, obstructions, and abandoned/blocked workstreams.
10. Workstream status describes execution lifecycle only; scientific outcome belongs in artifacts, relations, and reviews.
11. Prefer small graph-scoped workflows over giant prompts.
12. Keep model IDs and prices centralized.
13. Never commit `.env` or `.theory/`.
14. Do not add autonomous agent swarms, a vector database, or a graph database without evaluation evidence.
15. Never label a theorem or proof "verified" merely because an LLM found no flaw.

Current phase: development is frozen after the first graph-backed `attack` workflow. Evaluate reconstructed states from a completed research project before adding another workflow or major feature. The V0.1 `investigate` workflow remains only as compatibility functionality.
