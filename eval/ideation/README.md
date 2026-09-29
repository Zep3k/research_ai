# Bounded ideation and Case 02B

Run the offline evaluation/regressions:

```sh
.venv/bin/python -m pytest tests/test_research_ideation.py -q
```

`case02b.json` reconstructs a weak-agreement contract and the relevant known
primitive. The graph contains a quarantined counterexample to certificate
uniqueness: conflicting n-f certificates may exist. The supplied contract allows
v/bottom, and also requires persistence and termination. The old exclusion
obligation remains open throughout the test.

The scripted generator supplies three alternatives. The first propagates
conflict evidence using the existing authenticated super-send primitive, explores
a guarded bottom output, and explicitly identifies timely dissemination and
preserving persistence/termination as risks. The other alternatives examine local
filtering and reuse of relay evidence. They are hypotheses, not known-correct
protocols. No certificate count or thesis mechanism appears in production trigger
logic or prompts.

The controller makes all three alternatives selectable as ordinary primary
`develop` moves. In the positive test the strategist chooses the propagation
candidate; only that idea enters execution context and the accepted artifact's
provenance. The other ideas remain only in API-call telemetry. A second test has
the strategist decline every idea. Another test supplies structurally valid but
unhelpful machinery-adding ideas: validation succeeds but the fixture's useful
candidate check fails. This separates orchestration from idea quality.

The positive run uses **1 ideation + 1 strategy + 2 execution calls**, within the
existing four-call ceiling for `max_calls=2`. It stops at the original execution
cap, leaves the obligation open, and preserves quarantine. Its final execution
uses the deterministic baseline because planning slots are spent. Telemetry
contains the trigger, generated summaries, selection, call ID, model, normalized
usage, and cost; the tests use zero network calls and synthetic usage/cost.

Production triggers are generic persisted graph/iteration predicates:

- Active counterexample, challenged candidate, or refuted/failed approach;
- A recent completed attack reporting a concrete critical issue;
- A recent reframe reporting an alternative route;
- Recent synthesis consuming complementary inputs without resolution or a new
  obligation candidate awaiting ordinary testing;
- Three completed develop/synthesis iterations without resolution.

Ideation requires an existing primary develop move and two remaining planning
slots. It runs at most once per invocation. Persisted trigger fingerprints and a
three-completed-iteration cooldown prevent immediate repeats, including after
resuming a failed request. Explicit provider overrides and strategy-off runs do
not ideate. Terminal workstreams and existing stop conditions are not reopened.

Tests also cover strict batch sizes and fields, duplicate ideas, invalid graph
references, unoffered selections, conservative budget refusal, provider failures,
no extra calls, stable prompt prefixes, selected-only scientific persistence,
cooldowns, debug telemetry, and idempotent migration from v11 call receipts.

**Result:** Case 02B exposes a useful simplifying candidate under deterministic
scripted responses. This demonstrates availability, selection, context delivery,
and persistence behavior. It does not establish that a live model reliably invents
that candidate, that all generated mechanisms are semantically distinct, or that
the proposed protocol meets weak agreement. No proof verification is claimed.
