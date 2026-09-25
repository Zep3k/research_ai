# Coding-agent instructions

Goal: build a personal theoretical-research workbench that improves research decisions and mathematical reliability.

Constraints:
1. Keep research state provider-independent.
2. Log every model call with model, purpose, tokens, and estimated cost.
3. Keep monthly budget enforcement explicit and local.
4. Preserve source metadata for literature-derived claims.
5. Prefer small workflows over giant prompts.
6. Keep model IDs/prices centralized.
7. Never commit `.env` or `.theory/`.
8. Do not add autonomous agent swarms or a vector DB without evaluation evidence.
9. Never label a theorem/proof "verified" merely because an LLM found no flaw.

Current milestone: make `theory investigate IDEA_ID` beat ordinary one-shot chat on known research cases.
