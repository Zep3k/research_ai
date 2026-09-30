"""Explicit failure scope preserves the uniform-publication parent frontier."""
import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from theory.db import connect
from theory.errors import ModelOutputError
from theory.graph import add_relation, set_attribute
from theory.research import (
    OperationChoice, ResearchArtifact, ResearchStepReport, _all_branches_terminal,
    _candidate_has_unresolved_critical_issue, _history, _open_obligation_ids,
    _validate_step_report, choose_next_operation, generate_legal_research_moves, research,
)
from theory.research_context import for_workstream
from theory.research_routes import SUPERSEDED_BY, live_construction_route_ids, route_entity_is_live
from test_research import DynamicProvider, artifact, init_workspace, make_research_workstream, step_report
from test_research_consolidation import append_step
from test_route_exhaustion import add_work, mark


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Real providers are forbidden"))


@pytest.fixture
def publication(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, primary = make_research_workstream()
    root, = append_step(ws, primary)
    obligation = add_work(ws, "proof_obligation", route=root, related=(primary, root))
    with connect() as con:
        con.execute("UPDATE entities SET body='Construct uniform publication under the same contract.' WHERE id=?", (obligation,))
        con.execute("UPDATE research_iterations SET iteration_number=9 WHERE workstream_id=?", (ws,))
    return ws, primary, root, obligation


def frontier(ws, primary):
    context = for_workstream(ws)
    goal = next(e for e in context.entities if e["id"] == primary)
    history = _history(ws)
    return context, generate_legal_research_moves(context, ws, goal, history), choose_next_operation(context, ws, goal, history)


def resume(ws, **kwargs):
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (ws,))
    return research(ws, max_calls=1, strategy="off", **kwargs)


def test_iterations_10_11_failed_echo_rule_preserves_uniform_publication(publication, monkeypatch):
    ws, primary, root, obligation = publication
    child = None

    def execute(decision, number):
        assert decision["operation"] == "develop"
        refs = [primary, root, obligation]
        if number == 1:
            return step_report(decision, [artifact(
                "protocol_component", "Publish the collection of every valid echo received by round B.",
                "echo_publication_rule", refs, branch_status="unresolved",
            )])
        failure = artifact(
            "obstruction", "Delayed echoes can give two honest parties different round-B publication collections.",
            "echo_cutoff_nonuniform", [*refs, child], branch_status="blocked",
        )
        failure["refutes_entity_ids"] = [child]
        return step_report(decision, [failure], unresolved=["Uniform publication remains unconstructed."])

    provider = DynamicProvider(execute)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    child, = resume(ws).artifact_ids
    failure, = resume(ws).artifact_ids
    context, moves, baseline = frontier(ws, primary)
    assert not route_entity_is_live(context, child)
    assert not route_entity_is_live(context, failure)
    assert route_entity_is_live(context, root) and route_entity_is_live(context, obligation)
    assert live_construction_route_ids(context) == frozenset({root})
    assert _open_obligation_ids(context, ws, primary) == (obligation,)
    assert not _all_branches_terminal(context, ws)
    assert baseline.operation in {"develop", "synthesize", "reframe"}
    assert baseline.focus_obligation_id == obligation
    assert any(m.focus_obligation_id == obligation and m.operation == "reframe" for m in moves)
    assert all(not (m.operation in {"prove", "attack"} and m.target_entity_id == child) for m in moves)
    assert not _candidate_has_unresolved_critical_issue(context, root)
    assert _candidate_has_unresolved_critical_issue(context, child)
    assert context.attributes[failure]["research_branch_status"] == "blocked"
    assert json.loads(context.attributes[failure]["related_entity_ids"]) == [primary, root, obligation, child]
    scoped_edges = [r for r in context.relations if r["relation_type"] == "REFUTES"]
    assert [(r["source_entity_id"], r["target_entity_id"]) for r in scoped_edges] == [(failure, child)]
    assert scoped_edges[0]["trust_state"] == "quarantined"
    iterations = _history(ws)[-2:]
    assert [i["iteration_number"] for i in iterations] == [10, 11]
    assert all(i["selected_move_id"].endswith(":continue") for i in iterations)
    assert iterations[-1]["open_obligations_before"] == iterations[-1]["open_obligations_after"] == 1
    assert iterations[-1]["resolution_progress"] == 0
    assert frontier(ws, primary) == (context, moves, baseline)  # Reconstruction after resume.


@pytest.mark.parametrize("kind,state", [("obstruction", "blocked"), ("failed_approach", "refuted"),
                                         ("counterexample", "unresolved")])
def test_relevance_and_owned_negative_evidence_never_establish_closure(publication, kind, state):
    ws, primary, root, obligation = publication
    child = add_work(ws, "proof_attempt", route=root, related=(root, obligation))
    negative = add_work(ws, kind, route=root, state=state, related=(primary, root, obligation, child))
    with connect() as con:
        # Even explicit prose about impossibility must not be reinterpreted as a directed edge.
        con.execute("UPDATE entities SET body='This parent construction is impossible.' WHERE id=?", (negative,))
    context = for_workstream(ws)
    assert all(route_entity_is_live(context, i) for i in (primary, root, obligation, child))
    assert not _candidate_has_unresolved_critical_issue(context, root)
    assert not _candidate_has_unresolved_critical_issue(context, child)
    assert _open_obligation_ids(context, ws, primary) == (obligation,)
    assert not _all_branches_terminal(context, ws)


@pytest.mark.parametrize("relation", ["REFUTES", "BLOCKS", "CONTRADICTS", "FAILS_AT"])
def test_closure_relations_apply_only_to_their_explicit_target(publication, relation):
    ws, primary, root, obligation = publication
    child = add_work(ws, "proof_attempt", route=root, related=(root, obligation))
    negative = add_work(ws, "obstruction", route=root, state="blocked", related=(root, obligation, child))
    source, target = (child, negative) if relation == "FAILS_AT" else (negative, child)
    with connect() as con:
        add_relation(con, source, relation, target)
    context = for_workstream(ws)
    assert not route_entity_is_live(context, child)
    assert route_entity_is_live(context, root) and route_entity_is_live(context, obligation)
    assert _open_obligation_ids(context, ws, primary) == (obligation,)
    retired = replace(context, relations=tuple({**r, "status": "retired"} for r in context.relations))
    assert route_entity_is_live(retired, child)


@pytest.mark.parametrize("relation", ["REFUTES", "BLOCKS", "CONTRADICTS", "FAILS_AT", "superseded"])
def test_explicit_parent_route_closure_still_exhausts_the_frontier(publication, relation):
    ws, primary, root, obligation = publication
    negative = add_work(ws, "obstruction", route=root, state="blocked", related=(root, obligation))
    with connect() as con:
        if relation == "superseded":
            set_attribute(con, root, SUPERSEDED_BY, json.dumps([negative]))
        elif relation == "FAILS_AT":
            add_relation(con, root, relation, negative)
        else:
            add_relation(con, negative, relation, root)
    context = for_workstream(ws)
    assert not route_entity_is_live(context, root)
    assert _open_obligation_ids(context, ws, primary) == ()
    assert _all_branches_terminal(context, ws)
    first = resume(ws)
    assert first.calls_made == first.total_api_calls_made == 0
    assert first.stop_reason == "all_branches_blocked_or_refuted"
    assert resume(ws) == first


def test_explicit_impossibility_of_obligation_has_its_own_scope(publication):
    ws, primary, root, obligation = publication
    negative = add_work(ws, "obstruction", route=root, state="blocked", related=(root, obligation))
    with connect() as con:
        add_relation(con, negative, "REFUTES", obligation)
    context = for_workstream(ws)
    assert not route_entity_is_live(context, obligation)
    assert route_entity_is_live(context, root)
    assert _open_obligation_ids(context, ws, primary) == ()
    # The persisted premise remains auditable; no ancestor is implicitly closed.
    assert context.attributes[obligation]["research_obligation_state"] == "open"
    assert not _all_branches_terminal(context, ws)


@pytest.mark.parametrize("scope,error", [
    ("unknown", "unknown/out-of-context"),
    ("missing_reference", "must include every explicitly refuted entity"),
    ("legacy", None),
])
def test_structured_refutation_scope_must_be_in_context(publication, scope, error):
    ws, primary, root, obligation = publication
    context = for_workstream(ws)
    choice = OperationChoice("develop", obligation, "Check publication.", open_obligation_ids=(obligation,), focus_obligation_id=obligation)
    decision = {"operation": "develop", "target_entity_id": obligation, "required_consumed_entity_ids": []}
    negative = artifact("obstruction", "The tested echo cutoff gives incompatible collections.", "cutoff_scope",
                        [obligation], branch_status="blocked")
    negative["refutes_entity_ids"] = {
        "unknown": [999999], "missing_reference": [root], "legacy": [],
    }[scope]
    report = ResearchStepReport.model_validate(step_report(decision, [negative]))
    if error:
        with pytest.raises(ModelOutputError, match=error):
            _validate_step_report(report, context, choice)
    else:
        _validate_step_report(report, context, choice)


def test_refutation_scope_is_optional_and_only_negative_artifacts_can_use_it():
    legacy = artifact("obstruction", "A scoped defect.", "scoped_defect", [1], branch_status="blocked")
    assert ResearchArtifact.model_validate(legacy).refutes_entity_ids == []
    for ids in ([0], [1, 1]):
        with pytest.raises(ValidationError):
            ResearchArtifact.model_validate({**legacy, "refutes_entity_ids": ids})
    positive = artifact("protocol_component", "A delivery mechanism.", "delivery_mechanism", [1], branch_status="unresolved")
    with pytest.raises(ValidationError, match="Only negative artifacts"):
        ResearchArtifact.model_validate({**positive, "refutes_entity_ids": [1]})


def test_bounded_critical_attack_persists_scope_on_its_target_only(publication, monkeypatch):
    ws, primary, root, obligation = publication
    candidate = add_work(ws, "proof_attempt", route=root, related=(root, obligation))
    mark(candidate, research_focus_obligation_id=str(obligation))
    mark(root, research_branch_status="promising")
    provider = DynamicProvider(lambda decision, _: step_report(decision, [artifact(
        "obstruction", "An admissible echo delay breaks this particular timing argument.", "timing_attack_defect",
        [candidate, root, obligation], branch_status="blocked",
    )], attack_outcome="critical_issue"))
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    negative, = resume(ws, provider_name="openai").artifact_ids
    context = for_workstream(ws)
    assert any(r["source_entity_id"] == negative and r["relation_type"] == "REFUTES"
               and r["target_entity_id"] == candidate for r in context.relations)
    assert not route_entity_is_live(context, candidate)
    assert route_entity_is_live(context, root)
    assert _open_obligation_ids(context, ws, primary) == (obligation,)
