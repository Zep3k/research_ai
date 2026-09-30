"""Persisted focus recovery, guarded top-level ideation and strict attack scope."""
import json
from dataclasses import replace

import pytest

from theory.db import connect
from theory.graph import add_relation, set_attribute
from theory.research import (
    _constructive_continuation, _history, _open_obligation_ids, _select_focus_obligation,
    _top_level_ideation_context, _problem_contract, _research_prompt,
    generate_legal_research_moves, OperationChoice, research,
)
from theory.research_context import focus_research_context, for_workstream
from theory.research_ideation import build_ideation_prompt, ideation_telemetry, _obligation_triggers
from theory.research_routes import ROUTE_IDS, STARTED_AT, SUPERSEDED_BY
from test_research import DynamicProvider, artifact, step_report, decision_from_prompt
from test_research_consolidation import construction, append_step
from test_execution_context_scope import routed, generated, append_history, CONTRACT, MODEL
from test_obligation_ideation import local, batch, attempt, trigger
from test_research_ideation import install
from test_route_exhaustion import add_work, mark


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Real providers are forbidden"))


def continuation(ws, primary):
    context = for_workstream(ws)
    return _constructive_continuation(context, ws, _history(ws), _open_obligation_ids(context, ws, primary))


def test_continuation_recovers_single_owned_bottleneck_without_changing_target(construction, monkeypatch):
    ws, primary, root = construction
    obligation = add_work(ws, "proof_obligation", route=root, related=(primary, root))
    choice = continuation(ws, primary)
    assert choice.continue_construction and choice.target_entity_id == primary
    assert choice.focus_obligation_id == obligation
    provider = DynamicProvider(lambda decision, _: step_report(decision, [artifact(
        "protocol_component", "Extend the current construction with a still-unproved local release rule.",
        "local_release_extension", [primary, root, obligation], branch_status="unresolved",
    )]))
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    research(ws, strategy="off", max_calls=1)
    assert _history(ws)[-1]["focus_obligation_id"] == obligation
    context = for_workstream(ws)
    goal = next(e for e in context.entities if e["id"] == primary)
    moves = generate_legal_research_moves(context, ws, goal, _history(ws))
    assert all(m.focus_obligation_id is None for m in moves
               if m.operation == "develop" and m.target_entity_id == primary and not m.continue_construction)


def test_continuation_and_unresolved_local_synthesis_trigger_obligation_ideation(construction, monkeypatch):
    ws, primary, root = construction
    obligation = add_work(ws, "proof_obligation", route=root, related=(primary, root))
    state = (ws, primary, root, obligation)
    provider = install(monkeypatch, batch(state))
    original = provider.complete
    def unresolved_synthesis(**kwargs):
        result = original(**kwargs)
        if kwargs["response_model"].__name__ == "ResearchStepReport":
            report = json.loads(result.text)
            if report["operation"] == "synthesize":
                refs = report["artifacts"][0]["related_entity_ids"]
                report["artifacts"] = [artifact("obstruction", "The local pieces do not establish a common release predicate.",
                    "unresolved_release_synthesis", refs, branch_status="blocked")]
                result = result.model_copy(update={"text": json.dumps(report)})
        return result
    monkeypatch.setattr(provider, "complete", unresolved_synthesis)
    research(ws, strategy="off", max_calls=1)
    add_work(ws, "finding", route=root, related=(obligation,))
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (ws,))
    provider.select = lambda s: {
        "selected_move_id": next(m["move_id"] for m in s["legal_moves"]
                                 if m["operation"] == "synthesize" and m["focus_obligation_id"] == obligation),
        "rationale": "Combine the local unresolved inputs.",
    }
    research(ws, max_calls=1)
    rows = _history(ws)
    assert rows[-2]["selected_move_id"].endswith(":continue")
    assert rows[-2]["target_entity_id"] == primary
    assert rows[-2]["focus_obligation_id"] == rows[-1]["focus_obligation_id"] == obligation
    assert rows[-1]["operation"] == "synthesize"
    assert trigger(state).reason == "repeated_obligation_failure"


@pytest.mark.parametrize("kind", ["none", "global", "foreign"])
def test_continuation_without_owned_bottleneck_keeps_null_focus(construction, kind):
    ws, primary, root = construction
    if kind == "global":
        add_work(ws, "proof_obligation", related=(primary,))
    elif kind == "foreign":
        other = add_work(ws, "proof_attempt")
        mark(other, **{ROUTE_IDS: json.dumps([other]), STARTED_AT: "999"})
        add_work(ws, "proof_obligation", route=other, related=(primary, other))
    choice = continuation(ws, primary)
    assert choice.continue_construction and choice.target_entity_id == primary
    assert choice.focus_obligation_id is None


def test_multiple_owned_obligations_use_existing_fair_focus_policy(construction):
    ws, primary, root = construction
    first = add_work(ws, "proof_obligation", route=root)
    second = add_work(ws, "proof_obligation", route=root)
    context = for_workstream(ws)
    history = ({"id": 0, "status": "completed", "operation": "develop", "target_entity_id": first,
                "focus_obligation_id": first, "artifact_ids_json": "[]"}, *_history(ws))
    choice = _constructive_continuation(context, ws, history, (first, second))
    assert choice.focus_obligation_id == _select_focus_obligation(context, history, (first, second)) == second
    reordered = replace(context, entities=context.entities[::-1])
    again = _constructive_continuation(reordered, ws, history, (second, first))
    assert again.focus_obligation_id == choice.focus_obligation_id
    assert again.target_entity_id == choice.target_entity_id


def test_unrelated_root_develops_never_count_as_local_attempts(construction):
    ws, primary, root = construction
    obligation = add_work(ws, "proof_obligation", route=root)
    append_step(ws, primary)
    assert all(r["focus_obligation_id"] is None for r in _history(ws))
    assert _obligation_triggers(for_workstream(ws), _history(ws), (obligation,)) == ()


@pytest.mark.parametrize("ablation", ["auto", "off", "openai", "anthropic"])
def test_direct_local_work_suppresses_top_level_ideation(local, monkeypatch, ablation):
    ws, _, root, obligation = local
    add_work(ws, "obstruction", route=root, state="blocked", related=(obligation,))
    provider = install(monkeypatch, batch(local), select_idea=False)
    result = research(ws, strategy="off" if ablation == "off" else "auto",
                      provider_name=ablation if ablation in {"openai", "anthropic"} else "auto", max_calls=2)
    assert result.ideation_calls_made == 0 and not ideation_telemetry(ws)
    if ablation == "auto":
        assert any(m["operation"] == "reframe" for m in provider.offered[0])
        assert any(m["operation"] == "develop" and m["focus_obligation_id"] is None
                   for m in provider.offered[0])


def test_local_ideation_is_not_suppressed_by_actionable_frontier(local, monkeypatch):
    attempt(local)
    attempt(local, operation="synthesize")
    install(monkeypatch, batch(local))
    result = research(local[0], max_calls=2)
    assert result.ideation_calls_made == 1
    trace, = ideation_telemetry(local[0])
    assert trace["planning"]["trigger"] == "repeated_obligation_failure"
    assert trace["planning"]["focus_obligation_id"] == local[3]


@pytest.mark.parametrize("operation", ["attack", "prove", "synthesize", "develop"])
def test_each_actionable_focused_operation_gates_top_level_call(local, monkeypatch, operation):
    ws, _, root, obligation = local
    mark(root, research_branch_status="promising", precise_candidate="false" if operation == "develop" else "true")
    add_work(ws, "obstruction", route=root, state="blocked", related=(obligation,))
    if operation == "attack":
        add_work(ws, "proof_attempt", route=root, related=(obligation,))
    elif operation == "synthesize":
        for _ in range(2):
            add_work(ws, "finding", route=root, related=(obligation,))
    provider = install(monkeypatch, batch(local), select_idea=False)
    # Admit the selector; skip execution so this gate test needs no attack output.
    monkeypatch.setattr("theory.research.budget_guard", lambda *args, **kwargs: 0.25)
    result = research(ws, max_calls=2, max_cost_usd=0.25)
    assert result.ideation_calls_made == result.calls_made == 0
    assert result.strategy_calls_made == 1
    assert [m["operation"] for m in provider.offered[0]
            if m["focus_obligation_id"] == obligation and m["operation"] != "reframe"] == [operation]
    assert not ideation_telemetry(ws)


def test_top_level_ideation_without_local_frontier_uses_compact_telemetry(local, monkeypatch):
    ws, primary, root, obligation = local
    mark(obligation, research_obligation_state="resolved_candidate")
    negative = add_work(ws, "obstruction", route=root, state="blocked", related=(root,))
    install(monkeypatch, batch(local))
    result = research(ws, max_calls=2)
    assert result.ideation_calls_made == 1
    trace, = ideation_telemetry(ws)
    assert trace["planning"]["trigger"] == "concrete_refutation"
    scope = json.loads(trace["context_scope_json"])
    assert {primary, root, negative}.issubset(scope["included_entity_ids"])
    assert scope["focused_entity_count"] == len(scope["included_entity_ids"])
    assert scope["full_workstream_entity_count"] == 4


def test_inactive_route_trigger_does_not_consume_top_level_ideation_call(local, monkeypatch):
    ws, primary, root, obligation = local
    mark(obligation, research_obligation_state="resolved_candidate")
    old = add_work(ws, "proof_attempt")
    mark(old, **{ROUTE_IDS: json.dumps([old]), STARTED_AT: "999", SUPERSEDED_BY: json.dumps([root])})
    add_work(ws, "obstruction", route=old, state="blocked", related=(primary, root))
    install(monkeypatch, batch(local))
    assert research(ws, max_calls=2).ideation_calls_made == 0
    assert not ideation_telemetry(ws)


def test_top_level_scope_is_bounded_and_keeps_trigger_dependencies_and_evidence(routed):
    ws, primary, model, root, old, current, *_ = routed
    with connect() as con:
        premise = generated(con, ws, "trigger_premise", kind="Finding", related=(model,))
        negative = generated(con, ws, "direct_failure", route=root, kind="Obstruction", status="blocked")
        older_failure = generated(con, ws, "older_child_failure", route=root, kind="FailedApproach", status="failed",
                                  related=(current[0],))
        trigger_id = generated(con, ws, "trigger_counterexample", route=root, kind="Counterexample", status="blocked")
        set_attribute(con, negative, "related_entity_ids", json.dumps([root]))
        add_relation(con, trigger_id, "DEPENDS_ON", premise)
        retired = generated(con, ws, "retired_relation_only", kind="FailedApproach", status="failed")
        add_relation(con, retired, "BLOCKS", current[-1], status="retired")
    before = _top_level_ideation_context(for_workstream(ws), ws, primary, (trigger_id,))
    expected = {primary, model, root, current[0], *current[-4:], premise, negative, older_failure, trigger_id}
    assert set(before.context_scope["included_entity_ids"]) == expected
    assert retired not in expected and old not in expected
    unrelated = append_history(routed)
    full = for_workstream(ws)
    after = _top_level_ideation_context(full, ws, primary, (trigger_id,))
    assert set(after.context_scope["included_entity_ids"]) == expected
    assert unrelated.isdisjoint(expected)
    from theory.research_ideation import IdeationTrigger
    event = IdeationTrigger("concrete_refutation", (trigger_id,))
    contract = _problem_contract(full, ws)
    prompt_before = build_ideation_prompt(before, contract, event).render()
    prompt_after = build_ideation_prompt(after, contract, event).render()
    assert "UNRELATED_HISTORY_" not in prompt_after
    assert {e["id"]: e["body"] for e in after.entities}[primary] == CONTRACT
    assert {e["id"]: e["body"] for e in after.entities}[model] == MODEL
    assert abs(len(prompt_before.encode()) - len(prompt_after.encode())) <= 8


@pytest.fixture
def attack_state(routed):
    ws, primary, model, root, old, current, *_ = routed
    with connect() as con:
        obligation = generated(con, ws, "local_requirement", route=root, kind="OpenQuestion", related=(primary,))
        proof = generated(con, ws, "tested_candidate", route=root, kind="ProofAttempt", related=(obligation,))
        premise = generated(con, ws, "cited_premise", route=root, kind="Lemma", related=(model,))
        deep = generated(con, ws, "deep_premise", route=root, kind="Finding", related=(primary,))
        add_relation(con, proof, "DEPENDS_ON", premise)
        add_relation(con, premise, "USES", deep)
        add_relation(con, deep, "USES", premise)
        negative = generated(con, ws, "premise_failure", route=root, kind="Obstruction", status="blocked",
                             related=(premise, current[1]))
        add_relation(con, negative, "BLOCKS", premise)
        old_negative = generated(con, ws, "old_failure", route=old, kind="Obstruction", status="blocked", related=(proof,))
    return routed, obligation, proof, premise, deep, negative, old_negative


def attack_scope(state, **extra):
    routed, obligation, proof, *_ = state
    ws, primary, *_ = routed
    return focus_research_context(for_workstream(ws), workstream_id=ws, primary_entity_id=primary,
        target_entity_id=proof, focus_obligation_id=obligation, operation="attack", **extra)


def test_attack_keeps_explicit_premises_and_negative_evidence_without_siblings(attack_state):
    routed, obligation, proof, premise, deep, negative, old_negative = attack_state
    ws, primary, model, root, _, current, *_ = routed
    focused = attack_scope(attack_state)
    expected = {primary, model, root, obligation, proof, premise, deep, negative}
    assert set(focused.context_scope["included_entity_ids"]) == expected
    assert set(current).isdisjoint(expected) and old_negative not in expected
    assert focused.context_scope["expansion_anchor_ids"] == [obligation, proof]
    assert focused.context_scope["focused_entity_count"] == len(expected)
    assert focused.context_scope["full_workstream_entity_count"] == len(for_workstream(ws).entities)
    assert focused.attributes[premise] == for_workstream(ws).attributes[premise]
    assert {e["id"]: e["body"] for e in focused.entities}[primary] == CONTRACT


def test_unrelated_same_route_history_does_not_grow_attack_prompt(attack_state):
    routed, obligation, proof, *_ = attack_state
    ws, primary, _, root, _, *_ = routed
    before = attack_scope(attack_state)
    goal = next(e for e in before.entities if e["id"] == primary)
    choice = OperationChoice("attack", proof, "Test the cited argument.", focus_obligation_id=obligation)
    prompt_before = _research_prompt(before, goal, choice)
    with connect() as con:
        for n in range(100):
            sibling = generated(con, ws, f"same_route_history_{n}", route=root, kind="Lemma", related=(obligation,))
            con.execute("UPDATE entities SET body=? WHERE id=?", ("UNRELATED_SAME_ROUTE_HISTORY " * 300, sibling))
    after = attack_scope(attack_state)
    prompt_after = _research_prompt(after, goal, choice)
    assert before.entities == after.entities
    assert "UNRELATED_SAME_ROUTE_HISTORY" not in prompt_after
    assert abs(len(prompt_before.encode()) - len(prompt_after.encode())) <= 8


def test_explicit_consumed_historical_evidence_never_disappears(attack_state):
    old_negative = attack_state[-1]
    focused = attack_scope(attack_state, consumed_entity_ids=(old_negative,))
    assert old_negative in focused.context_scope["included_entity_ids"]
    assert old_negative in focused.context_scope["expansion_anchor_ids"]


def test_actual_cited_historical_premise_is_kept_for_refutation_audit(attack_state):
    routed, _, proof, *_ = attack_state
    ws, _, _, _, old, *_ = routed
    with connect() as con:
        premise = generated(con, ws, "historical_cited_premise", route=old, kind="Lemma")
        add_relation(con, proof, "DEPENDS_ON", premise)
        failure = generated(con, ws, "historical_premise_refutation", route=old, kind="Obstruction", status="blocked")
        add_relation(con, failure, "REFUTES", premise)
    assert {premise, failure}.issubset(attack_scope(attack_state).context_scope["included_entity_ids"])


def test_attack_retains_explicit_bypass_replacement_premises(attack_state):
    routed, _, proof, *_ = attack_state
    ws, _, _, root, *_ = routed
    replacement = add_work(ws, "proof_obligation", route=root)
    mark(proof, research_bypass_replacement_obligation_ids=json.dumps([replacement]))
    assert replacement in attack_scope(attack_state).context_scope["included_entity_ids"]


def test_input_attack_target_keeps_directed_premises_without_contract_neighbors(routed):
    ws, primary, model, _, _, *_ = routed
    unrelated = append_history(routed)
    with connect() as con:
        premise = generated(con, ws, "input_target_premise", kind="Lemma", related=(model,))
        add_relation(con, primary, "DEPENDS_ON", premise)
    focused = focus_research_context(for_workstream(ws), workstream_id=ws,
        primary_entity_id=primary, target_entity_id=primary, operation="attack")
    assert set(focused.context_scope["included_entity_ids"]) == {primary, model, premise}
    assert primary not in focused.context_scope["expansion_anchor_ids"]
    assert focused.context_scope["expansion_anchor_ids"] == [premise]
    assert unrelated.isdisjoint(focused.context_scope["included_entity_ids"])
