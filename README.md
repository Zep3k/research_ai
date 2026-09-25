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
        +--> optional provider-independent model workflows
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

`sources` stores a locator into a paper or external source: page, section, theorem number, excerpt, DOI/URL, with fields optional when inapplicable. `entity_sources` attaches one or more locators to a claim. This separate join is necessary because a paper is not itself evidence for every statement attributed to it.

Lifecycle status and epistemic state answer different questions:

- `status` (`active`, `abandoned`, `resolved`) says what happened to the research object.
- `trust_state` says what kind of warrant it has: `unverified`, `sourced`, `inferred`, `speculative`, `contradicted`, or `quarantined`.

`sourced` means a concrete persisted locator supports the statement. It does not mean the statement or proof has been mathematically verified. Adding a source does not silently promote a claim; promotion is an explicit command, and the database rejects promotion without provenance. Contradicted or quarantined objects must pass through `unverified` before they can be reconsidered as sourced.

### Workstreams and reviews

A workstream is a durable focused effort (`literature`, `explore`, `attack`, or `proof`), not an agent persona. Its status can be `active`, `completed`, `failed`, `abandoned`, or `blocked`. Entities attach as `input`, `created`, `modified`, `evidence`, or `blocked_by`.

Failure is permanent research memory. A failed proof attack, its obstruction, and its counterexample remain queryable so the same dead end is not rediscovered months later.

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
theory workstream status 1 failed --summary "No refutation found; adaptive schedule remains unresolved."
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

The builder selects a deterministic one-hop neighborhood: the target or linked workstream artifacts, direct relations, assumptions, nearby theorems/lemmas, proof attempts, counterexamples, blockers, sourced findings, attributes, and provenance. It performs no embeddings, vector search, or model call.

## V0.1 compatibility and migration

The first open of a V0.1 database migrates it in place to schema version 3. Back up important `.theory/` directories before any upgrade.

Migration behavior is explicit:

1. Missing V0.1 ledger/source-hardening columns are added for early databases.
2. V0.2 graph tables and a `schema_migrations` ledger are created.
3. Every legacy `ideas` row becomes an unverified `ResearchIdea` entity.
4. Every legacy `papers` row becomes an unverified `Paper` entity.
5. `legacy_entity_links` records the old-to-new ID mapping; graph IDs may differ from legacy IDs.
6. Existing runs, reports, raw provider responses, PDF paths, hashes, and cost records are retained.
7. `api_calls.workstream_id` is added as nullable. Existing rows remain unattributed; new references are guarded even on migrated tables.

Migration does not reinterpret old investigation JSON as sourced graph knowledge. Doing so would manufacture trust that V0.1 did not record. The legacy `idea`, `paper`, `investigate`, and `run show` commands remain available; new `idea add` and `paper add` operations also create linked graph entities.

## Retained V0.1 provider layer

OpenAI and Anthropic adapters remain behind the same provider-independent interface. Every attempted paid call is entered in `api_calls` before execution and completed or failed with provider, model, purpose, token counts, estimated cost, timestamp, raw response when available, run ID, and optional workstream ID.

Before a call, a conservative local bound is checked against the configured monthly budget. Unknown models are refused until pricing is added centrally in `theory/providers.py`. Pricing and model availability are external state and must be verified against official provider documentation before being changed.

The retained `investigate` workflow still operates on legacy ideas and OpenAlex discovery metadata. It is compatibility functionality, not the V0.2 architecture, and its JSON report is not automatically promoted into the research graph.

## Verify locally

```bash
source .venv/bin/activate       # macOS/Linux
python -m pytest -q
theory --help
```

Tests require no API keys and make no network or paid model calls. OpenAlex is mocked where used.

## Deliberate omissions

V0.2 has no vector database, embeddings, Neo4j, giant-corpus RAG, web UI, cloud infrastructure, Zotero integration, autonomous loop, agent swarm, automatic paper generation, automatic novelty claim, or automatic theorem-verification claim.

The product test remains: **did this prevent the researcher from wasting two weeks?**
