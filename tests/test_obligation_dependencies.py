"""Obligation hierarchy requires explicit persisted prerequisites, never association."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from theory.db import connect
from theory.graph import add_relation, set_attribute
from theory.research import (
    _history, _open_obligation_ids, build_research_state, eligible_open_obligation_ids,
    generate_legal_research_moves, research,
)
from theory.research_context import ResearchContext, for_workstream
from theory.research_routes import ROUTE_IDS
from test_research import artifact, step_report
from test_research_strategy import install_providers
from test_route_commitment import committed, construction
from test_route_exhaustion import add_work, mark


def projected_v7():
    data = json.loads((Path(__file__).parent / "fixtures" / "obligation_dependency_frontier.json").read_text())
    context = ResearchContext(
        workstream=data["workstream"], entities=tuple(data["entities"]),
        attributes={int(i): attrs for i, attrs in data["attributes"].items()},
        relations=tuple(data["relations"]), workstream_links=tuple(data["workstream_links"]),
    )
    return context, next(e for e in context.entities if e["id"] == 1)


def test_v7_central_agreement_and_local_forwarding_are_independently_actionable():
    context, primary = projected_v7()
    assert _open_obligation_ids(context, 1, 1) == (8, 9)
    assert 8 in json.loads(context.attributes[9]["related_entity_ids"])
    assert context.attributes[9]["research_focus_obligation_id"] == "6"
    assert eligible_open_obligation_ids(context, (8, 9)) == (8, 9)
    moves = generate_legal_research_moves(context, 1, primary, ())
    assert {m.focus_obligation_id for m in moves} == {8, 9}
    state = build_research_state(context, workstream_id=1, primary=primary, history=(), legal_moves=moves)
    assert state.controller_summary.eligible_obligation_ids == (8, 9)
    assert {o.id for o in state.open_obligations if o.eligible} == {8, 9}


@pytest.mark.parametrize("association", ["related", "focused", "same_route"])
def test_association_is_not_a_prerequisite(committed, association):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root)
    if association == "related":
        mark(child, related_entity_ids=json.dumps([parent]))
    elif association == "focused":
        mark(child, research_focus_obligation_id=str(parent), research_operation="develop")
    context = for_workstream(ws)
    assert eligible_open_obligation_ids(context, (parent, child)) == (parent, child)


@pytest.mark.parametrize("operation", ["develop", "prove", "synthesize", "attack"])
def test_persisted_focused_creation_does_not_invent_a_dependency(committed, monkeypatch, operation):
    ws, primary, root, parent = committed
    if operation == "prove":
        lemma = construction(ws, "A local conditional lemma", "lemma")
        mark(lemma, **{ROUTE_IDS: json.dumps([root]), "related_entity_ids": json.dumps([parent])})
    elif operation == "synthesize":
        for _ in range(2):
            add_work(ws, "finding", route=root, related=(parent,))
    elif operation == "attack":
        proof = add_work(ws, "proof_attempt", route=root, related=(parent,))
        with connect() as con:
            add_relation(con, proof, "ATTEMPTS", parent)

    def execute(decision, _):
        assert decision["operation"] == operation
        refs = sorted({decision["target_entity_id"], parent, *decision["required_consumed_entity_ids"]})
        outputs = [artifact("proof_obligation", "Bound the forwarding cost of the local publication rule.",
                            "independent_forwarding_bound", refs)]
        if operation == "synthesize":
            outputs.append(artifact("obstruction", "The attempted combination leaves the publication witness unspecified.",
                                    "local_combination_gap", refs, branch_status="blocked"))
        return step_report(decision, outputs,
                           attack_outcome="inconclusive" if operation == "attack" else "not_applicable",
                           unresolved=["The local cost bound remains unresolved."] if operation == "attack" else [])

    requests, _ = install_providers(monkeypatch, execute=execute)
    result = research(ws, strategy="off", max_calls=1)
    child = result.artifact_ids[0]
    context = for_workstream(ws)
    assert context.attributes[child]["research_focus_obligation_id"] == str(parent)
    assert parent in json.loads(context.attributes[child]["related_entity_ids"])
    assert context.attributes[child][ROUTE_IDS] == json.dumps([root])
    assert eligible_open_obligation_ids(context, (parent, child)) == (parent, child)
    assert len(requests) == result.total_api_calls_made == 1


def test_active_directed_prerequisite_defers_parent_and_survives_reload(committed):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root)
    with connect() as con:
        relation = add_relation(con, parent, "DEPENDS_ON", child)
    for context in (for_workstream(ws), for_workstream(ws)):
        assert eligible_open_obligation_ids(context, (parent, child)) == (child,)
        goal = next(e for e in context.entities if e["id"] == primary)
        assert {m.focus_obligation_id for m in generate_legal_research_moves(context, ws, goal, ())} == {child}
        assert any(r["id"] == relation for r in context.relations)


@pytest.mark.parametrize("relation", ["USES", "reversed_dependency"])
def test_association_and_dependency_direction_are_distinct(committed, relation):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root)
    with connect() as con:
        add_relation(con, child if relation == "reversed_dependency" else parent,
                     "DEPENDS_ON" if relation == "reversed_dependency" else relation,
                     parent if relation == "reversed_dependency" else child)
    assert eligible_open_obligation_ids(for_workstream(ws), (parent, child)) == (
        (parent,) if relation == "reversed_dependency" else (parent, child)
    )


@pytest.mark.parametrize("terminal", ["resolved_candidate", "blocked", "bypassed", "unnecessary",
                                     "failed", "refuted", "inactive", "contradicted", "route_inactive"])
def test_adjudicated_or_inactive_prerequisite_cannot_hide_parent(committed, terminal):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root)
    with connect() as con:
        add_relation(con, parent, "DEPENDS_ON", child)
    if terminal in {"failed", "refuted"}:
        mark(child, research_branch_status=terminal)
    elif terminal in {"inactive", "contradicted"}:
        with connect() as con:
            con.execute(f"UPDATE entities SET {'status' if terminal == 'inactive' else 'trust_state'}=? WHERE id=?",
                        ("abandoned" if terminal == "inactive" else terminal, child))
    elif terminal == "route_inactive":
        old = construction(ws, "An explicitly failed owner")
        mark(old, research_branch_status="failed")
        mark(child, **{ROUTE_IDS: json.dumps([old])})
    else:
        mark(child, research_obligation_state=terminal)
    context = for_workstream(ws)
    # Even a caller holding stale open IDs cannot make this child block its parent.
    assert eligible_open_obligation_ids(context, (parent, child)) == (parent,)
    assert _open_obligation_ids(context, ws, primary) == (parent,)


def test_pending_proof_attack_overrides_an_explicit_prerequisite(committed):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root)
    proof = add_work(ws, "proof_attempt", route=root, related=(parent,))
    with connect() as con:
        add_relation(con, parent, "DEPENDS_ON", child)
        add_relation(con, proof, "ATTEMPTS", parent)
    context = for_workstream(ws)
    assert eligible_open_obligation_ids(context, (parent, child)) == (parent, child)
    goal = next(e for e in context.entities if e["id"] == primary)
    moves = generate_legal_research_moves(context, ws, goal, ())
    assert {m.operation for m in moves if m.focus_obligation_id == parent} == {"attack"}
    assert any(m.target_entity_id == proof and m.focus_obligation_id == parent for m in moves)


def bypass_candidate(ws, root, parent, child, *, pending=False):
    candidate = add_work(ws, "finding", route=root)
    attributes = {"research_reframe_target_obligation_id": str(parent),
                  "research_bypass_replacement_obligation_ids": json.dumps([child])}
    if pending:
        mark(parent, research_obligation_state="reframe_pending_attack",
             research_reframe_candidate_id=str(candidate))
    else:
        attributes.update(research_bypass_activated_iteration_id="1", research_attack_state="survived_attack")
    mark(candidate, **attributes)
    return candidate


def test_pending_necessity_attack_overrides_explicit_replacement(committed):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root)
    candidate = bypass_candidate(ws, root, parent, child, pending=True)
    with connect() as con:
        add_relation(con, parent, "DEPENDS_ON", child)
    context = for_workstream(ws)
    assert eligible_open_obligation_ids(context, (parent, child)) == (parent, child)
    goal = next(e for e in context.entities if e["id"] == primary)
    moves = generate_legal_research_moves(context, ws, goal, ())
    assert any(m.operation == "attack" and m.target_entity_id == candidate
               and m.focus_obligation_id == parent for m in moves)


def test_validated_replacement_is_required_followup_without_related_ids(committed):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root)
    bypass_candidate(ws, root, parent, child)
    context = for_workstream(ws)
    assert eligible_open_obligation_ids(context, (parent, child)) == (child,)
    mark(parent, research_obligation_state="bypassed")
    resumed = for_workstream(ws)
    assert _open_obligation_ids(resumed, ws, primary) == (child,)
    assert eligible_open_obligation_ids(resumed, (child,)) == (child,)


def test_shared_replacement_cannot_hide_parent_of_a_failed_bypass(committed):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root, related=(parent,))
    candidate = bypass_candidate(ws, root, parent, child)
    other_parent = add_work(ws, "proof_obligation", route=root)
    bypass_candidate(ws, root, other_parent, child)
    mark(candidate, research_attack_state="challenged")
    mark(parent, research_necessity_audit_state="reactivated")
    context = for_workstream(ws)
    assert eligible_open_obligation_ids(context, (parent, child, other_parent)) == (parent, child)


@pytest.mark.parametrize("independent_leaf", [False, True])
def test_cycles_cannot_deadlock_even_alongside_an_independent_leaf(committed, independent_leaf):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root)
    ids = [parent, child]
    if independent_leaf:
        ids.append(add_work(ws, "proof_obligation", route=root))
    with connect() as con:
        add_relation(con, parent, "DEPENDS_ON", child)
        add_relation(con, child, "DEPENDS_ON", parent)
    assert eligible_open_obligation_ids(for_workstream(ws), tuple(ids)) == tuple(ids)


@pytest.mark.parametrize("malformed", ["self", "dangling", "non_obligation", "retired", "missing_target", "bad_target"])
def test_malformed_dependency_cannot_suppress_parent(committed, malformed):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root)
    context = for_workstream(ws)
    target = {"self": parent, "dangling": 999999, "non_obligation": root,
              "retired": child, "missing_target": None, "bad_target": "not-an-id"}[malformed]
    relation = {"relation_type": "DEPENDS_ON", "source_entity_id": parent,
                "target_entity_id": target, "status": "retired" if malformed == "retired" else "active"}
    if malformed == "missing_target":
        relation.pop("target_entity_id")
    context = replace(context, relations=(relation,))
    assert eligible_open_obligation_ids(context, (parent, child)) == (parent, child)


def test_dependency_selection_is_pure_and_order_independent(committed, monkeypatch):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root)
    sibling = add_work(ws, "proof_obligation", route=root)
    with connect() as con:
        add_relation(con, parent, "DEPENDS_ON", child)
    context = for_workstream(ws)
    snapshot = context.as_dict()
    for name in ("connect", "call_model", "get_provider", "choose_model_route"):
        monkeypatch.setattr(f"theory.research.{name}", lambda *args, **kwargs: pytest.fail("Must use only graph state"))
    assert eligible_open_obligation_ids(context, (parent, child, sibling)) == (child, sibling)
    reordered = replace(context, entities=context.entities[::-1], relations=context.relations[::-1])
    assert eligible_open_obligation_ids(reordered, (sibling, child, parent)) == (child, sibling)
    assert context.as_dict() == snapshot


def test_strategist_receives_all_independent_committed_obligations(committed, monkeypatch):
    ws, primary, root, parent = committed
    child = add_work(ws, "proof_obligation", route=root, related=(parent,))
    mark(child, research_focus_obligation_id=str(parent))
    offered = []

    def select(state):
        offered.append(state)
        assert state["controller_summary"]["eligible_obligation_ids"] == [parent, child]
        assert {m["focus_obligation_id"] for m in state["legal_moves"]} == {parent, child}
        assert {o["id"] for o in state["open_obligations"] if o["eligible"]} == {parent, child}
        return {"selected_move_id": next(m["move_id"] for m in state["legal_moves"]
                                        if m["operation"] == "develop" and m["focus_obligation_id"] == child),
                "rationale": "Compare the independently actionable obligations."}

    requests, _ = install_providers(monkeypatch, select=select)
    result = research(ws, max_calls=1)
    assert len(offered) == result.strategy_calls_made == 1
    assert len(requests) == result.total_api_calls_made == 2
