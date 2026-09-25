# theory-research v0.1

A deliberately small local CLI for theoretical research.

The purpose of v0.1 is to test whether persistent project state + literature retrieval + adversarial LLM analysis beats ordinary chat for deciding what theoretical research deserves more time.

## Install

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
# PowerShell: .\\.venv\\Scripts\\Activate.ps1
# cmd.exe:    .venv\\Scripts\\activate.bat
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
# PowerShell: Copy-Item .env.example .env
# cmd.exe:    copy .env.example .env
# macOS/Linux: cp .env.example .env
```

Add at least one LLM API key to `.env`. A free `OPENALEX_API_KEY` is strongly
recommended for literature searches; anonymous OpenAlex access has a very small shared
budget. `OPENALEX_EMAIL` remains optional. The `.env` file and all workspace state are
ignored by Git.

## First run

```bash
theory init "Master thesis" --monthly-budget 100
theory idea add "Can the cubic multishot warm-up be reduced under the same model?"
theory idea list
theory investigate 1 --provider openai
theory run show 1
theory budget
```

Anthropic works the same way:

```bash
theory investigate 1 --provider anthropic
```

You can also import a PDF locally:

```bash
theory paper add path/to/paper.pdf
theory paper list
```

PDF import validates the document, copies it under `.theory/papers`, extracts page-marked
text, and records a SHA-256 digest. In v0.1, imported paper titles appear in project context,
but PDF contents are not automatically treated as evidence during `investigate`.

## V0.1 architecture

```text
local SQLite research state
        |
        +-- ideas
        +-- papers
        +-- investigation history
        +-- API-cost ledger
        |
        +--> LLM formalizes the question
        +--> OpenAlex retrieves nearby literature
        +--> LLM attacks/reframes the direction
        +--> typed report + retrieved-source metadata + cost saved locally
```

An investigation is created before its first provider call. Every call attempt is entered in
the local ledger before the request starts and then marked completed or failed with provider,
model, purpose, token counts, estimated cost, timestamp, and raw response when available.
Failed parsing therefore does not erase paid work. OpenAlex query outcomes and exact metadata
are saved with the run, and sourced findings must cite one of that run's `S1`, `S2`, ... IDs.

Before each paid call, the CLI computes a conservative worst-case price from the prompt size
and provider-side output-token cap. It refuses the call if that upper bound would cross the
configured monthly budget. Unknown models are refused until their pricing is added centrally
in `theory/providers.py`.

The default IDs and standard uncached rates were verified on 2026-09-25 against the official
[OpenAI model documentation](https://platform.openai.com/docs/models) and
[Anthropic Opus page](https://www.anthropic.com/claude/opus). Re-check them before changing
models; provider pricing and availability are external state.

Existing v0.1 databases are migrated in place when opened. Keep a backup of `.theory` before
upgrading important long-lived workspaces.

OpenAlex searches use bounded retry/backoff and save per-query failures in the investigation
instead of silently discarding them. OpenAlex now uses credit-based limits; review its current
[authentication and rate-limit documentation](https://help.openalex.org/api/authentication/)
when configuring a key. OpenAlex charges, quotas, and availability are not part of the local
LLM monthly-budget ledger in v0.1.

## Local verification

```bash
python -m pytest -q
theory --help
```

The tests use fake providers and do not make paid API calls.

## Deliberate omissions

No vector DB, no multi-agent swarm, no cloud deployment, no automatic novelty claims, and no claim that a proof is "verified." Those can be reconsidered after evaluation.

## How to evaluate it

Test it on old research cases where you already know what happened. A useful run should do at least one of these:

- find relevant literature you missed,
- expose a hidden assumption,
- kill/reframe a weak direction early,
- identify a serious counterexample target,
- identify a concrete question whose answer changes what you do next.

If it mainly produces polished brainstorming, improve the workflow before adding features.
