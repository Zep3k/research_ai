# theory-research v0.2

A local, durable workbench for serious theoretical-STEM research. It is designed to help a human decide what deserves attention, preserve evidence and dead ends, compare results precisely, and make both the researcher and language models harder to fool.

It is not an autonomous scientist, an automatic theorem prover, or a chat-history archive.

## Install

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
# PowerShell: .\\.venv\\Scripts\\Activate.ps1
# cmd.exe:    .venv\\Scripts\\activate.bat
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Initialize a workspace in the directory where its research state should live:

```bash
theory init "Master thesis" --monthly-budget 100
```

This creates `.theory/research.db`, `.theory/config.json`, and `.theory/papers/`. Both `.theory/` and `.env` are ignored by Git. The graph, sources, workstreams, and model-call ledger are local SQLite state.

API keys are unnecessary for all graph commands and tests. To use the retained V0.1 `investigate` compatibility workflow, copy `.env.example` to `.env` and configure a provider key. A free OpenAlex key is strongly recommended for that workflow. Never commit `.env`.

## Conceptual architecture

```text
typed research graph
    entities + generic attributes + explicit relations
        |
        +-- precise provenance locators
        +-- lifecycle status and epistemic trust state
        +-- durable workstreams and retained failures
        +-- bounded review records
        +-- deterministic graph-neighborhood context
        |
        +--> small graph-backed workflows + bounded controller
                 + per-call tokens/cost ledger
                 + local monthly-budget gate
```

The architecture is centered on provider-independent research objects, not model transcripts. OpenAI or Anthropic may reason over a selected graph neighborhood, but no provider owns the durable research state.

### Entities

V0.2 supports these strongly validated types:

```text
Paper, Theorem, Definition, Assumption, Model, Technique,
ResearchIdea, Conjecture, OpenQuestion,
ProofAttempt, Lemma, Counterexample, Obstruction, FailedApproach, Finding
```

They share one representation: title, body, lifecycle status, trust state, confidence, and timestamps. The schema deliberately avoids an elaborate class hierarchy.

### Relations

Relations are directed, validated, and must reference existing entities:

```text
USES, DEPENDS_ON, EXTENDS, IMPROVES, CONTRADICTS, SUPPORTS,
REFUTES, BLOCKS, ATTEMPTS, FAILS_AT, SOURCED_FROM
```

Deleting an entity used by a relation is restricted. Entity-owned attributes cascade when an otherwise unreferenced entity is deleted. The CLI intentionally offers no delete command in V0.2.

### Provenance and trust

`sources` stores a locator: page, section, theorem number, excerpt, paper entity, and/or external URL. `entity_sources` attaches one or more locators to a claim. This separate join is necessary because a paper is not itself evidence for every statement attributed to it.

A page- or section-only locator may be retained as an incomplete research note. It cannot support `sourced` state until it has an identifiable origin: either a `Paper` entity or a non-empty stable external URL. The link means only that the location is claimed as evidence for the object; it does not establish the claim's truth.

Lifecycle status and epistemic state answer different questions:

- `status` (`active`, `abandoned`, `resolved`) says what happened to the research object.
- `trust_state` says what kind of warrant it has: `unverified`, `sourced`, `inferred`, `speculative`, `contradicted`, or `quarantined`.

`sourced` means a persisted locator with an identifiable origin is claimed to support the statement. It does not mean the statement or proof has been mathematically verified. Adding a source does not silently promote a claim; promotion is an explicit command, and application and database gates reject promotion without identifiable provenance. Contradicted or quarantined objects must pass through `unverified` before they can be reconsidered as sourced.

### Workstreams and reviews

A workstream is a durable focused effort (`literature`, `explore`, `attack`, `develop`, `research`, or `proof`), not an agent persona. Its lifecycle status is one of:

- `active`: execution may proceed.
- `completed`: the bounded workflow execution finished normally, regardless of scientific outcome.
- `blocked`: execution cannot currently proceed because a recorded dependency is unresolved.
- `abandoned`: the human intentionally stopped this effort without completing it.
- `error`: execution failed technically, for example due to provider or output-validation failure.
- `legacy_failed`: migration-only compatibility state for ambiguous V0.2 `failed`; no scientific or execution conclusion may be inferred until a human reclassifies it.

Entities attach as `input`, `created`, `modified`, `evidence`, or `blocked_by`.

Scientific outcomes do not live in lifecycle status. A completed attack that found no refutation remains `completed`; its `no_flaw_found` or `inconclusive` review records the bounded scientific outcome. Failed approaches, obstructions, and counterexamples remain queryable so the same dead end is not rediscovered months later.

Reviews persist a bounded observation such as `proof_critique` or `source_verification`. Allowed results are only:

```text
no_flaw_found, issue_found, inconclusive
```

There is deliberately no `verified` result.

## Core CLI

Create and inspect objects:

```bash
theory entity add theorem "Known result"
theory entity add conjecture "Possible improvement"
theory entity list
theory entity show 1
```

Attach structured, domain-independent attributes:

```bash
theory attr set 1 synchrony asynchronous
theory attr set 1 fault_model Byzantine
theory attr set 1 resilience "t < n/3"
theory attr list 1
```

Attribute keys receive syntax-only normalization: lowercase, surrounding whitespace removed, spaces and hyphens converted to underscores, and repeated underscores collapsed. This intentionally does not merge semantic variants such as `communication` and `communication_complexity`.

Add and inspect relations:

```bash
theory relation add 2 EXTENDS 1
theory relation list 2
```

Add precise provenance. `--paper` takes a `Paper` entity ID, not necessarily the legacy `paper` table ID in a migrated workspace:

```bash
theory source add 2 \
  --paper 3 \
  --page 7 \
  --section "Main results" \
  --theorem "Theorem 3.2" \
  --external-url "https://doi.org/10.1000/example"
theory entity trust 2 sourced
```

Create and retain focused efforts:

```bash
theory workstream create attack "Try to refute conjecture #2"
theory workstream link 1 2 input
theory workstream status 1 completed --summary "Attack pass finished; scientific result is in its review."
theory workstream show 1
```

Store a bounded review:

```bash
theory review add proof_critique no_flaw_found \
  --entity 2 \
  --issues "Checked the induction step only"
```

## Deterministic theorem delta

`theory delta A B` compares generic attributes without calling a model. It exists because theoretical novelty often lies in one changed assumption, guarantee, or complexity bound—not in a fluent summary.

Toy example:

```bash
theory entity add theorem "Known cubic protocol"
theory entity add conjecture "Quadratic protocol under the same model"

theory attr set 1 synchrony asynchronous
theory attr set 2 synchrony asynchronous
theory attr set 1 fault_model Byzantine
theory attr set 2 fault_model Byzantine
theory attr set 1 resilience "t < n/3"
theory attr set 2 resilience "t < n/3"
theory attr set 1 communication "O(n^3)"
theory attr set 2 communication "O(n^2)"

theory relation add 2 EXTENDS 1
theory delta 1 2
```

Output:

```text
UNCHANGED
fault_model = Byzantine
resilience = t < n/3
synchrony = asynchronous

CHANGED
communication:
  A = O(n^3)
  B = O(n^2)

ONLY IN A

ONLY IN B
```

The attributes are generic key/value records. Distributed-computing terminology is an example, not a schema commitment; theoretical ML or another field can use its own keys.

## Deterministic model context

`theory.research_context` exposes:

```python
from theory import research_context

entity_context = research_context.for_entity(entity_id)
workstream_context = research_context.for_workstream(workstream_id)
```

The builder selects a deterministic one-hop neighborhood: the target or linked workstream artifacts, active direct relations, assumptions, nearby theorems/lemmas, proof attempts, counterexamples, blockers, sourced findings, attributes, and provenance. Retired relations are excluded.

Every context has explicit `sourced`, `inferred`, `speculative`, `unverified`, `contradicted`, and `quarantined` partitions for both entities and active relations. The model-facing serialization foregrounds these instructions:

- sourced is source-backed, not theorem-verified;
- inferred is provisional;
- speculative is a hypothesis;
- contradicted is counterevidence/history;
- quarantined must never be assumed true.

## Graph-backed attack workflow

An attack workstream must be `active`, have type `attack`, and contain exactly one eligible primary `input`: `Conjecture`, `ResearchIdea`, `OpenQuestion`, or `Theorem`. Supporting inputs of other types are allowed. Multiple plausible targets are rejected rather than guessed.

```bash
theory workstream create attack "Try to refute the proposed improvement"
theory workstream link 1 2 input
theory attack 1 --provider openai
theory workstream show 1
```

The workflow makes exactly one provider call using only `research_context.for_workstream(1)`. It performs no literature retrieval, web search, recursion, provider comparison, or extra display call. The strict result contains a target ID, concise summary, candidate artifacts, and explicit unresolved points. Candidate kinds are:

```text
counterexample, obstruction, hidden_assumption, boundary_case,
conflict, ambiguity, failed_strategy, open_question
```

The model may label a candidate only `inference`, `speculation`, or `unresolved`; `sourced` is not in the output schema. Every candidate becomes a small typed entity through the deterministic write gate with `generated_by_llm=True`, initial trust `quarantined`, and workstream role `created`. Existing source IDs may be cited and attached, but do not lift quarantine. The workflow creates no `REFUTES` relation automatically.

A successful call always sets lifecycle to `completed`. A concrete candidate produces review result `issue_found`; only questions/unresolved output is `inconclusive`; an empty pass with no reported unknowns is `no_flaw_found`. This last phrase means only that this one pass found no concrete issue. Provider or validation failure sets lifecycle to `error`. Budget refusal occurs before execution and leaves it `active`.

## Graph-backed develop workflow

A develop workstream must be `active`, have type `develop`, and contain exactly one eligible primary `input`. Eligible targets are a `ResearchIdea`, `Conjecture`, `OpenQuestion`, `Theorem`, `Lemma`, `Technique`, `ProofAttempt`, or `Finding`. Other inputs can supply context, but multiple eligible targets are rejected rather than guessed.

```bash
theory workstream create develop "Advance the promising construction"
theory workstream link 2 2 input
theory develop 2 --provider openai
theory workstream show 2
```

Like `attack`, `develop` makes exactly one provider call over `research_context.for_workstream(...)`, performs no retrieval, and records the workstream-linked call and cost before invoking the provider. Its strict result must contain:

- a derived consequence;
- a parameter or counting analysis;
- a proof obligation;
- at least one intermediate lemma or protocol component;
- two to four materially different branches; and
- explicit unresolved points.

Branches have a scientific-development label of `promising`, `blocked`, `failed`, or `unresolved`; this does not alter the workstream lifecycle meaning. A failed branch is retained as a `FailedApproach`, a blocked branch as an `Obstruction`, and promising/unresolved branches as `Technique` candidates. Other development items map to the existing `Finding`, `Lemma`, `Technique`, and `OpenQuestion` entity types.

All model-created development entities pass through the normal write gate with `generated_by_llm=True`, start `quarantined`, and attach to the workstream as `created`. Model output can say only `inference`, `speculation`, or `unresolved`; those labels are retained as metadata and do not lift quarantine. Existing in-context source IDs may be cited, but do not make the generated object sourced. No graph relation, novelty claim, proof claim, or verification review is created automatically.

A valid provider response sets lifecycle to `completed`, including when a branch failed. Provider or structured-output failure sets lifecycle to `error`; budget refusal occurs before execution and leaves the workstream `active`. `workstream show` reads the stored summary, artifacts, trust states, and cost without another model call.

## Bounded iterative research controller

The `research` controller is deliberately bounded, graph-scoped, and auditable. It requires an active `research` workstream with exactly one eligible primary input:

```bash
theory workstream create research "Advance and stress-test the candidate"
theory workstream link 3 2 input
theory research 3 --provider openai --max-calls 4
theory workstream show 3
```

`--max-calls` is limited to 1–20. Every completed iteration makes exactly one provider call. A deterministic controller—not another model call—chooses the next operation from current linked graph state:

- `develop` when the target is still vague or lacks an actionable proof/synthesis frontier;
- `synthesize` when a specific open proof obligation has at least two usable linked artifacts;
- `prove` only when a precise theorem, lemma, protocol, conjecture, or proof candidate exists;
- `attack` only for a concrete theorem/lemma/protocol/proof candidate, prioritizing new proof attempts.

The choice, target, and rationale are persisted in `research_iterations` before the operation call. The model must echo that choice; it cannot redirect the controller. Synthesis must echo and reference every required consumed entity. All prompts contain only `research_context.for_workstream(...)`, the deterministic decision, and no retrieval or chat history.

Each strict response contains typed artifacts, a stable `material_key`, in-context entity/source references, provisional epistemic status, addressed-obligation candidates, attack outcome, unresolved points, and any request for human judgment. New objects and generated `ATTEMPTS` relations pass through the write gate as `quarantined` and attach to the workstream. An addressed obligation means only that a concrete candidate lemma/proof attempt was recorded; it does not mean the obligation or theorem is verified.

Failed or refuted branches persist as `FailedApproach`; blocked branches persist as `Obstruction`; new proof obligations persist as `OpenQuestion`. A deterministic duplicate gate rejects repeated material keys and high-overlap normalized statements. Rejected rephrasing is not material progress.

The controller stops when:

- addressed obligations are followed by an attack reporting `no_critical_issue`—recorded explicitly as a bounded, non-verifying result;
- every recorded branch is blocked, failed, or refuted;
- two consecutive iterations create no substantive non-duplicate object;
- the response says human scientific judgment is required;
- the call limit is reached; or
- the local budget guard refuses the next call.

Success or call-limit completion sets lifecycle `completed`; no live branch, stagnation, or required human judgment sets it `blocked`; provider/output failure sets it `error`. Budget refusal happens before an iteration or API-call record is created and leaves the workstream active. `workstream show` displays decisions, rationales, progress, duplicates, stop reasons, artifacts, reviews, and costs without a model call.

## V0.1 compatibility and migration

The first open of an older database migrates it in place to schema version 6. Back up important `.theory/` directories before any upgrade.

Migration behavior is explicit:

1. Missing V0.1 ledger/source-hardening columns are added for early databases.
2. V0.2 graph tables and a `schema_migrations` ledger are created.
3. Every legacy `ideas` row becomes an unverified `ResearchIdea` entity.
4. Every legacy `papers` row becomes an unverified `Paper` entity.
5. `legacy_entity_links` records the old-to-new ID mapping; graph IDs may differ from legacy IDs.
6. Existing runs, reports, raw provider responses, PDF paths, hashes, and cost records are retained.
7. `api_calls.workstream_id` is added as nullable. Existing rows remain unattributed; new references are guarded even on migrated tables.
8. V0.2 `failed` workstreams become `legacy_failed`; the migration does not guess whether execution or research failed.
9. Existing sourced entities/relations backed only by incomplete locators are conservatively downgraded to `unverified`.
10. Schema version 5 broadens only the workstream-type constraint to add `develop`; existing workstream rows, links, and model-call references retain their IDs and values.
11. Schema version 6 adds the `research` workstream type and durable `research_iterations`; existing workstream IDs, links, reviews, and model-call references are preserved.

Migration does not reinterpret old investigation JSON as sourced graph knowledge. Doing so would manufacture trust that V0.1 did not record. The legacy `idea`, `paper`, `investigate`, and `run show` commands remain available; new `idea add` and `paper add` operations also create linked graph entities.

## Retained V0.1 provider layer

OpenAI and Anthropic adapters remain behind the same provider-independent interface. Every attempted paid call is entered in `api_calls` before execution and completed or failed with provider, model, purpose, token counts, estimated cost, timestamp, raw response when available, run ID, and optional workstream ID.

Before a call, a conservative local bound is checked against the configured monthly budget. Unknown models are refused until pricing is added centrally in `theory/providers.py`. Pricing and model availability are external state and must be verified against official provider documentation before being changed.

The retained `investigate` workflow still operates on legacy ideas and OpenAlex discovery metadata. It is compatibility functionality, not the graph-backed attack architecture, and its JSON report is not automatically promoted into the research graph.

## Verify locally

```bash
source .venv/bin/activate       # macOS/Linux
python -m pytest -q
theory --help
```

Tests require no API keys and make no network or paid model calls. OpenAlex is mocked where used.

## Deliberate omissions

This release has no workflow-time retrieval, second-model review, unbounded autonomous loop, vector database, embeddings, Neo4j, giant-corpus RAG, web UI, cloud infrastructure, Zotero integration, agent swarm, automatic paper generation, automatic novelty claim, or automatic theorem-verification claim.

The product test remains: **did this prevent the researcher from wasting two weeks?**
