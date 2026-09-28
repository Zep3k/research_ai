# Offline protocol-construction evaluation

Run from the repository root, without provider credentials:

```sh
.venv/bin/python -m eval.protocol_construction.runner
.venv/bin/python -m pytest tests/test_protocol_construction.py
```

`cases.py` defines seven immutable contracts, staged executor responses, and
structural expectations. The tiny JSON DSL inside ordinary artifact statements
names concepts, state variables, message types, transition rules, invariants,
explicit dependencies, and assumptions. The checker reads **persisted artifacts**
and requires an assembled candidate with the required structure and graph links;
it does not compare natural-language wording, trust a summary, or equate a
completed workstream with scientific success. Transition strings are symbolic
rules (event/guard/action), not expected prose.

`runner.py` runs the production controller, routing, validators, graph writes,
duplicate checks, budget admission, progress aggregation, and stopping logic.
Only provider completion is scripted. A call-through completion hook captures
traces; it never changes decisions or output. Scripts can see only the actual
focused prompt. A missing component cannot be read directly from the database
by the executor. The strategy script chooses the requested stage if offered,
otherwise the first legal move. Most cases use deterministic strategy-off mode;
the combination case exercises the actual strategy-call path. Additional tests
exercise continuation with strategy enabled.

Each run creates and removes an isolated temporary workspace. API-call ledger
entries are real local records for **mock calls**, with zero synthetic usage and
cost. There are no network requests, extra probing calls, or writes to an existing
research workspace. Runs are sequential because workspace selection uses cwd.

## Baseline failures and root causes

`baseline.json` was captured against the unmodified `main` revision `b7b200f`,
then reproduced from an archive of that revision with the same harness.
`after.json` is the post-fix deterministic snapshot. Both include legal/selected
move IDs, actual focused IDs, accepted and duplicate artifact keys, generated and
open obligations, branch states, progress, call counts, and stop reasons.

| Case / simple solution | Baseline | After |
| --- | --- | --- |
| Direct contract: send once, deliver once on a reliable nonduplicating channel | Pass | Pass |
| Intermediate invariant: persistent-in-run seen-ID set, monotone invariant, guarded delivery | Partial component prematurely sent to prove; no final candidate | Pass |
| Failed route correction: preserve duplicate-delivery counterexample, construct seen-ID correction | Premature prove; failure artifact omitted from focused context | Pass |
| Combine components: fair-loss retries + duplicate-aware acknowledgments | No legal synthesis of same-iteration mechanisms | Pass |
| No reframing needed: grow-only set union + dissemination | Premature prove instead of finishing gossip construction | Pass |
| Transitive context: stored epoch state → max merge → broadcast | Proof executor cannot see stored epoch state | Pass |
| Bounded obligations: queue storage + occupancy invariant + guards | Focus pivots to obligation, loses storage; grows to three open obligations | Pass; one obligation remains open |

The transitive-context case starts from three reconstructed quarantined components;
the other six start with only the contract. Forbidden assumptions include FIFO
channels, perfect failure detectors, and atomic network delivery absent from the
contracts. The failed-route case requires the failed artifact to remain persisted.
The synthesis case requires a real synthesis iteration consuming both components.

## Changes

* A new internal continuation move preserves the previous develop target/focus
  only when that completed iteration materially accepted a live, explicitly
  unfinished protocol component. Obligation-only growth, duplicates, finished
  candidates, errors, intervening operations, or challenged/terminal components
  do not qualify. The move has a distinct `:continue` ID so continuing a route
  and escaping it are never conflated. Strategist input exposes that intent;
  model output schemas and provider interfaces are unchanged.
* Distinct mechanisms may be synthesized even with the same branch/iteration
  provenance. Existing duplicate suppression, terminal filtering, pair limits,
  explicit consumed references, and synthesis validation remain in place.
* Focused context follows recorded forward dependency ancestry with cycle
  protection, stopping at contract inputs. It does not recursively expand reverse
  neighbors into unrelated branches. Epistemic partitions are preserved.
* Develop instructions prefer concrete state/messages/guards/invariants over
  novelty and discourage manufacturing obligations or re-emitting the same
  premise. Genuine missing premises must still be exposed. Continuation and
  branch-escape instructions remain separate. The existing schema already
  permits one component, so no schema relaxation was necessary. Reframe, attack,
  prove, and synthesis output contracts were not weakened.

## Before/after results and limits

The main benchmark improves from **1/7 to 7/7** structural passes. Each version
uses **14 execution calls + 1 strategy call**. Every case stops at its unchanged
execution-call cap (`max_calls_exhausted`), not an invented success/verification
stop. No benchmark iteration reports obligation resolution. Generated artifacts
remain quarantined; the queue's capacity obligation is deliberately still open.

Two negative controls remain failures in both versions:

* A structurally complete candidate silently adds a FIFO-channel assumption.
  The checker rejects it, although production write validation correctly does
  not pretend to decide arbitrary scientific semantics.
* An executor emits unnecessary distinct obligations instead of construction.
  The checker detects the excess and missing candidate; obligation-only output
  never earns construction continuation. Existing lexical deduplication rejects
  repeated premises, but genuinely different executor-invented obligations can
  still grow within the existing call/artifact limits. We do not silently discard
  potentially necessary premises to improve the score.

These are **orchestration regressions**, not measured real-model intelligence.
The scripts already know the simple solution structures. Prompt changes need a
separate real-model evaluation to establish any executor-quality improvement.
The harness does not verify proofs, establish liveness for arbitrary executions,
or guarantee that dependency links supplied by an executor are complete.
