"""A failed mechanism is evidence, not exhaustion of an unbuilt frontier."""
import json

import pytest

from theory.db import connect
from theory.graph import add_entity, add_relation, link_workstream_entity, set_attribute
from theory.research import (
    _all_branches_terminal, _history, _open_obligation_ids, choose_next_operation,
    generate_legal_research_moves, research,
)
from theory.research_context import for_workstream
from theory.research_ideation import IdeaBatch, choose_ideation_trigger, ideation_telemetry
from theory.research_routes import ROUTE_IDS, STARTED_AT, SUPERSEDED_BY, live_construction_route_ids
from test_research import DynamicProvider, artifact, init_workspace, make_research_workstream, step_report
from test_research_consolidation import append_step
from test_research_ideation import OfflineProvider


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Unexpected provider call"))


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    return make_research_workstream()


def add_work(ws, kind, *, route=None, state="unresolved", related=()):
    types = {"obstruction": "Obstruction", "counterexample": "Counterexample",
             "failed_approach": "FailedApproach", "proof_obligation": "OpenQuestion",
             "proof_attempt": "ProofAttempt", "finding": "Finding"}
    with connect() as con:
        entity = add_entity(con, types[kind], f"Persisted {kind}", trust_state="quarantined")
        link_workstream_entity(con, ws, entity, "created")
        for key, value in {"research_artifact_type": kind, "research_branch_status": state,
                           ROUTE_IDS: json.dumps([route] if route else []),
                           "related_entity_ids": json.dumps(list(related))}.items():
            set_attribute(con, entity, key, value)
        if kind == "proof_obligation":
            set_attribute(con, entity, "research_obligation_state", "open")
    return entity


def mark(entity, **attributes):
    with connect() as con:
        for key, value in attributes.items():
            set_attribute(con, entity, key, value)


def assert_exhaustion(ws, expected):
    assert _all_branches_terminal(for_workstream(ws), ws) is expected
    # Reconstruct a fresh SQLite snapshot, as on resume.
    assert _all_branches_terminal(for_workstream(ws), ws) is expected


@pytest.mark.parametrize("kind,state", [("obstruction", "blocked"), ("counterexample", "refuted"),
                                         ("failed_approach", "failed")])
def test_standalone_terminal_negative_cannot_exhaust_the_frontier(fresh, kind, state):
    ws, primary = fresh
    negative = add_work(ws, kind, state=state, related=(primary,))
    assert_exhaustion(ws, False)
    context = for_workstream(ws)
    goal = next(e for e in context.entities if e["id"] == primary)
    choice = choose_next_operation(context, ws, goal, ())
    assert (choice.operation, choice.target_entity_id) == ("develop", primary)
    assert any(m.operation == "develop" and m.target_entity_id == primary
               for m in generate_legal_research_moves(context, ws, goal, ()))
    assert context.attributes[negative]["research_branch_status"] == state
    assert context.attributes[negative][ROUTE_IDS] == "[]"
    assert STARTED_AT not in context.attributes[negative]


@pytest.mark.parametrize("provider_name,strategy", [("auto", "off"), ("openai", "auto"), ("anthropic", "auto")])
def test_initial_obstruction_leaves_workstream_active_for_next_develop(fresh, monkeypatch, provider_name, strategy):
    ws, primary = fresh

    def execute(decision, number):
        assert (decision["operation"], decision["target_entity_id"]) == ("develop", primary)
        if number == 1:
            return step_report(decision, [artifact(
                "obstruction", "This cutoff mechanism omits a late conflicting seal.", "late_seal_obstruction",
                [primary], branch_status="blocked",
            )], unresolved=["The full protocol is not established."])
        with connect() as con:
            assert con.execute("SELECT status FROM workstreams WHERE id=?", (ws,)).fetchone()[0] == "active"
        assert not _all_branches_terminal(for_workstream(ws), ws)
        return step_report(decision, [artifact(
            "protocol_component", "Relay authenticated conflicts before applying a guarded release deadline.",
            "conflict_relay_component", [primary], branch_status="unresolved",
        )])

    provider = DynamicProvider(execute)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    outcome = research(ws, provider_name, max_calls=2, strategy=strategy)
    assert outcome.stop_reason == "max_calls_exhausted" and outcome.calls_made == 2
    assert outcome.strategy_calls_made == outcome.ideation_calls_made == 0
    assert len(provider.calls) == 2
    with connect() as con:
        iterations = [dict(r) for r in con.execute("SELECT * FROM research_iterations ORDER BY id")]
    assert iterations[0]["status"] == "completed" and iterations[0]["stop_reason"] is None
    context = for_workstream(ws)
    negative = json.loads(iterations[0]["artifact_ids_json"])[0]
    assert context.attributes[negative]["research_branch_status"] == "blocked"
    assert context.attributes[negative][ROUTE_IDS] == "[]"
    assert STARTED_AT not in context.attributes[negative]


def test_fresh_obstruction_triggers_ideation_on_following_iteration(fresh, monkeypatch):
    ws, primary = fresh
    batch = {"ideas": [
        {"idea_id": name, "mechanism": mechanism,
         "exploits": [{"entity_id": primary, "exploitation": "Preserve the supplied delivery contract."}],
         "route_change": "Replace the failed cutoff/seal mechanism.", "main_risk": "Its timing argument remains open."}
        for name, mechanism in (
            ("relay_conflicts", "Propagate authenticated conflict witnesses before releasing a guarded decision."),
            ("echo_deadlines", "Coordinate round-indexed echoes with a shared final release deadline."),
            ("bounded_buffers", "Reserve bounded sender queues and reconcile message certificates at delivery."),
        )
    ]}

    class FirstObstruction(OfflineProvider):
        def complete(self, **kwargs):
            result = super().complete(**kwargs)
            if self.executions == 1 and kwargs["response_model"].__name__ == "ResearchStepReport":
                report = json.loads(result.text)
                decision = {"operation": report["operation"], "target_entity_id": report["target_entity_id"],
                            "required_consumed_entity_ids": []}
                report = step_report(decision, [artifact(
                    "obstruction", "The cutoff accepts incompatible seals under delayed conflict evidence.",
                    "incompatible_seals", [primary], branch_status="blocked",
                )], unresolved=["No complete protocol has been established."])
                result = result.model_copy(update={"text": json.dumps(report)})
            if kwargs["response_model"] is IdeaBatch:
                with connect() as con:
                    assert con.execute("SELECT status FROM workstreams WHERE id=?", (ws,)).fetchone()[0] == "active"
                assert not _all_branches_terminal(for_workstream(ws), ws)
            return result

    provider = FirstObstruction(batch)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    result = research(ws, max_calls=2)
    assert result.calls_made == 2 and result.stop_reason == "max_calls_exhausted"
    assert result.ideation_calls_made == result.strategy_calls_made == 1
    assert len(provider.requests) == result.total_api_calls_made == 4
    trace, = ideation_telemetry(ws)
    negative = json.loads(_history(ws)[0]["artifact_ids_json"])[0]
    assert trace["planning"]["trigger"] == "concrete_refutation"
    assert trace["planning"]["entity_ids"] == [negative]
    assert _history(ws)[1]["selected_move_id"].endswith(":relay_conflicts")


def test_obstruction_resume_preserves_eligibility_and_ignores_uncertainty_prose(fresh, monkeypatch):
    ws, primary = fresh
    provider = DynamicProvider(lambda decision, _: step_report(decision, [artifact(
        "obstruction", "A local seal can miss a delayed contradictory message.", "missing_conflict_message",
        [primary], branch_status="blocked",
    )], unresolved=["The primary frontier is exhausted."]))
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    first = research(ws, strategy="off", max_calls=1)
    assert first.stop_reason == "max_calls_exhausted"  # Ordinary call-limit lifecycle is unchanged.
    context = for_workstream(ws)
    negative, = first.artifact_ids
    assert_exhaustion(ws, False)
    trigger = choose_ideation_trigger(context, _history(ws), ())
    assert trigger.reason == "concrete_refutation" and trigger.entity_ids == (negative,)
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (ws,))
    second = research(ws, "openai", max_calls=1)
    assert second.calls_made == 1 and second.stop_reason == "max_calls_exhausted"
    assert_exhaustion(ws, False)


def test_unowned_negative_does_not_retire_another_live_route(fresh):
    ws, primary = fresh
    root, = append_step(ws, primary)
    add_work(ws, "obstruction", state="blocked", related=(primary,))
    assert live_construction_route_ids(for_workstream(ws)) == frozenset({root})
    assert_exhaustion(ws, False)


@pytest.mark.parametrize("cause", ["blocked", "failed", "refuted", "relation", "owned_negative", "superseded"])
def test_established_routes_can_still_exhaust_without_calling_a_provider(fresh, cause):
    ws, primary = fresh
    root, = append_step(ws, primary)
    if cause in {"blocked", "failed", "refuted"}:
        mark(root, research_branch_status=cause)
    elif cause in {"relation", "owned_negative"}:
        negative = add_work(ws, "obstruction", route=root, state="blocked",
                            related=(root,) if cause == "owned_negative" else ())
        with connect() as con:
            # Ownership and relevance alone never establish refutation scope.
            add_relation(con, negative, "BLOCKS" if cause == "relation" else "REFUTES", root)
    else:
        replacement, = append_step(ws, primary)
        mark(root, **{SUPERSEDED_BY: json.dumps([replacement])})
        mark(replacement, research_branch_status="blocked")
        # Unfinished artifacts on a superseded route do not resurrect it.
        append_step(ws, primary, route=root, continuation=True)
    assert_exhaustion(ws, True)
    first = research(ws, max_calls=2)
    assert first.stop_reason == "all_branches_blocked_or_refuted"
    assert first.calls_made == first.total_api_calls_made == 0 and first.final_status == "blocked"
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 0
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (ws,))
    second = research(ws, max_calls=2)
    assert second == first  # Resume reconstructs the same stop entirely from the graph.


@pytest.mark.parametrize("work", ["global_obligation", "global_candidate", "pending_bypass", "survived_bypass"])
def test_active_work_prevents_exhaustion_after_route_failure(fresh, work):
    ws, primary = fresh
    root, = append_step(ws, primary)
    mark(root, research_branch_status="failed")
    if work == "global_obligation":
        add_work(ws, "proof_obligation", related=(primary,))
    elif work == "global_candidate":
        add_work(ws, "proof_attempt", related=(primary,))
    else:
        parent = add_work(ws, "proof_obligation", route=root, related=(primary,))
        candidate = add_work(ws, "finding", route=root, related=(parent, primary))
        mark(parent, research_obligation_state="reframe_pending_attack" if work == "pending_bypass" else "bypassed",
             research_reframe_candidate_id=str(candidate))
        mark(candidate, research_reframe_target_obligation_id=str(parent),
             research_bypass_replacement_obligation_ids="[]")
        if work == "survived_bypass":
            mark(candidate, research_bypass_activated_iteration_id="2", research_attack_state="survived_attack")
        assert _open_obligation_ids(for_workstream(ws), ws, primary) == ()
    assert_exhaustion(ws, False)


def test_only_active_obligations_count_toward_exhaustion(fresh):
    ws, primary = fresh
    root, = append_step(ws, primary)
    mark(root, research_branch_status="blocked")
    obligation = add_work(ws, "proof_obligation", route=root, related=(primary,))
    assert _open_obligation_ids(for_workstream(ws), ws, primary) == ()
    assert_exhaustion(ws, True)
    # Global ownership keeps the same open obligation active.
    mark(obligation, **{ROUTE_IDS: "[]"})
    assert_exhaustion(ws, False)


def test_obligation_only_route_is_not_substantive_until_it_has_construction(fresh):
    ws, primary = fresh
    root, = append_step(ws, primary, artifact_type="proof_obligation")
    mark(root, research_branch_status="blocked", research_obligation_state="blocked")
    assert_exhaustion(ws, False)
    component, = append_step(ws, primary, route=root)
    mark(component, research_branch_status="blocked")
    assert_exhaustion(ws, True)


def test_live_neighbor_from_another_workstream_does_not_mask_local_exhaustion(fresh):
    ws, primary = fresh
    root, = append_step(ws, primary)
    mark(root, research_branch_status="failed")
    with connect() as con:
        other = add_entity(con, "Technique", "An unrelated live construction")
        set_attribute(con, other, STARTED_AT, "999")
        set_attribute(con, other, ROUTE_IDS, json.dumps([other]))
        add_relation(con, other, "USES", primary)
    assert other in {e["id"] for e in for_workstream(ws).entities}
    assert_exhaustion(ws, True)
