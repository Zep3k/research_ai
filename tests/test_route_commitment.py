"""Structural route adjudication and the persisted parent/child proof regression."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from theory.research import (
    _construction_entity_is_active, _open_obligation_ids, _relevant_unattacked_proof_attempts,
    _validate_construction_choice, _history, OperationChoice, choose_next_operation,
    generate_legal_research_moves, research,
)
from theory.research_context import ResearchContext, for_workstream
from theory.research_ideation import IdeationTrigger
from theory.research_routes import (
    CONSTRUCTION_TYPES, ROUTE_IDS, STARTED_AT, SUPERSEDED_BY, committed_construction_route_ids,
    live_construction_route_ids, route_entity_is_live,
)
from theory.db import connect
from theory.errors import TheoryError
from theory.graph import add_entity, add_relation, link_workstream_entity, set_attribute
from test_research import artifact, init_workspace, make_research_workstream, step_report
from test_research_strategy import install_providers
from test_route_exhaustion import add_work, mark


def persisted_proof_graph():
    data = json.loads((Path(__file__).parent / "fixtures" / "proof_parent_child_frontier.json").read_text())
    context = ResearchContext(
        workstream=data["workstream"], entities=tuple(data["entities"]),
        attributes={int(i): attrs for i, attrs in data["attributes"].items()},
        relations=tuple(data["relations"]), workstream_links=tuple(data["workstream_links"]),
    )
    return context, next(e for e in context.entities if e["id"] == 1), tuple(data["history"])


@pytest.mark.parametrize("use_history", [False, True])
def test_persisted_proof_attempt_on_open_parent_must_surface_attack(use_history):
    context, primary, history = persisted_proof_graph()
    assert live_construction_route_ids(context) == frozenset({4})
    assert route_entity_is_live(context, 10) and _construction_entity_is_active(context, 10)
    assert _open_obligation_ids(context, 1, 1) == (8, 9, 11)
    assert [e["id"] for e in _relevant_unattacked_proof_attempts(context, (), (8,))] == [10]
    moves = generate_legal_research_moves(context, 1, primary, history if use_history else ())
    assert "attack:10:8:none" in {move.move_id for move in moves}
    assert {move.operation for move in moves if move.focus_obligation_id == 8} == {"attack"}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Real provider forbidden"))


def construction(ws, name="Unfinished construction", kind="protocol_component"):
    types = {"protocol_component": "Technique", "lemma": "Lemma", "proof_attempt": "ProofAttempt",
             "synthesis": "ProofAttempt", "finding": "Finding", "proof_obligation": "OpenQuestion"}
    with connect() as con:
        root = add_entity(con, types[kind], name, body=name, trust_state="quarantined")
        link_workstream_entity(con, ws, root, "created")
        for key, value in {ROUTE_IDS: json.dumps([root]), STARTED_AT: str(root),
                           "research_artifact_type": kind, "research_branch_status": "unresolved",
                           "precise_candidate": "false"}.items():
            set_attribute(con, root, key, value)
    return root


@pytest.fixture
def committed(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, primary = make_research_workstream()
    root = construction(ws)
    obligation = add_work(ws, "proof_obligation", route=root, related=(primary,))
    return ws, primary, root, obligation


def frontier(state, history=None, strategy_enabled=True):
    ws, primary, *_ = state
    context = for_workstream(ws)
    goal = next(e for e in context.entities if e["id"] == primary)
    history = _history(ws) if history is None else history
    moves = generate_legal_research_moves(context, ws, goal, history, strategy_enabled=strategy_enabled)
    return context, moves, choose_next_operation(context, ws, goal, history)


@pytest.mark.parametrize("kind", sorted(CONSTRUCTION_TYPES))
def test_exact_commitment_predicate_uses_persisted_construction_and_ownership(committed, kind, monkeypatch):
    ws, primary, root, obligation = committed
    mark(root, research_artifact_type=kind)
    context = for_workstream(ws)
    snapshot = context.as_dict()
    monkeypatch.setattr("theory.research.connect", lambda: pytest.fail("Commitment cannot query history"))
    assert committed_construction_route_ids(context, ws, (obligation,)) == frozenset({root})
    assert committed_construction_route_ids(replace(context, entities=context.entities[::-1]), ws,
                                            (obligation,)) == frozenset({root})
    assert context.as_dict() == snapshot


@pytest.mark.parametrize("kind", ["finding", "proof_obligation"])
def test_ownership_without_substantive_construction_does_not_commit(committed, kind):
    ws, primary, root, obligation = committed
    mark(root, research_artifact_type=kind)
    context, moves, _ = frontier(committed)
    assert not committed_construction_route_ids(context, ws, _open_obligation_ids(context, ws, primary))
    assert any(m.operation == "develop" and m.target_entity_id == primary and m.focus_obligation_id is None
               for m in moves)


def test_commitment_restricts_frontier_and_unrelated_continuation_without_history_inference(committed):
    ws, primary, root, obligation = committed
    other = construction(ws, "Separate unfinished route")
    global_obligation = add_work(ws, "proof_obligation", related=(primary,))
    history = ({"id": 100, "iteration_number": 100, "status": "completed", "operation": "develop",
                "target_entity_id": primary, "focus_obligation_id": None, "material_progress": 1,
                "artifact_ids_json": json.dumps([other]), "selected_move_id": f"develop:{primary}:none:none"},)
    mark(other, related_entity_ids=json.dumps([primary]))
    for previous in ((), history):
        context, moves, baseline = frontier(committed, previous)
        assert {m.focus_obligation_id for m in moves} == {obligation}
        assert baseline.target_entity_id == obligation and baseline.operation == "develop"
        assert {m.operation for m in moves} == {"develop", "reframe"}
        assert all(not m.continue_construction and m.target_entity_id != primary for m in moves)
        assert global_obligation in _open_obligation_ids(context, ws, primary)


@pytest.mark.parametrize("operation", ["develop", "synthesize", "prove", "attack"])
def test_focused_adjudication_operations_remain_legal(committed, operation):
    ws, primary, root, obligation = committed
    if operation == "synthesize":
        for name in ("First scoped bound", "Second scoped bound"):
            with connect() as con:
                evidence = add_entity(con, "Finding", name, body=name, trust_state="quarantined")
                link_workstream_entity(con, ws, evidence, "created")
                set_attribute(con, evidence, ROUTE_IDS, json.dumps([root]))
                set_attribute(con, evidence, "related_entity_ids", json.dumps([obligation]))
    elif operation == "prove":
        lemma = construction(ws, "Precise local certificate lemma", "lemma")
        mark(lemma, **{ROUTE_IDS: json.dumps([root]), "related_entity_ids": json.dumps([obligation])})
    elif operation == "attack":
        proof = add_work(ws, "proof_attempt", route=root, related=(obligation,))
        with connect() as con:
            add_relation(con, proof, "ATTEMPTS", obligation)
    _, moves, _ = frontier(committed)
    assert any(m.operation == operation and m.focus_obligation_id == obligation for m in moves)
    assert {m.operation for m in moves} == ({"attack"} if operation == "attack" else {operation, "reframe"})


def test_all_live_unattacked_attempts_surface_and_attack_precedes_continuation(committed):
    ws, primary, root, obligation = committed
    proofs = tuple(add_work(ws, "proof_attempt", route=root, related=(obligation,)) for _ in range(2))
    with connect() as con:
        for proof in proofs:
            add_relation(con, proof, "ATTEMPTS", obligation)
    child = add_work(ws, "proof_obligation", route=root, related=(obligation,))
    mark(child, research_focus_obligation_id=str(obligation))
    component = construction(ws, "Local missing mechanism")
    mark(component, **{ROUTE_IDS: json.dumps([root]), "related_entity_ids": json.dumps([obligation])})
    history = ({"id": 100, "status": "completed", "operation": "develop", "material_progress": 1,
                "target_entity_id": obligation, "focus_obligation_id": obligation,
                "artifact_ids_json": json.dumps([component]), "selected_move_id": f"develop:{obligation}:{obligation}:none"},)
    _, moves, _ = frontier(committed, history)
    assert {m.target_entity_id for m in moves if m.focus_obligation_id == obligation} == set(proofs)
    assert {m.operation for m in moves if m.focus_obligation_id == obligation} == {"attack"}


@pytest.mark.parametrize("state", ["resolved_candidate", "blocked", "bypassed", "unnecessary"])
def test_adjudicated_commitment_releases_normal_exploration(committed, state):
    ws, primary, root, obligation = committed
    mark(obligation, research_obligation_state=state)
    context, moves, _ = frontier(committed)
    assert not committed_construction_route_ids(context, ws, _open_obligation_ids(context, ws, primary))
    assert any(m.operation == "develop" and m.target_entity_id == primary for m in moves)


def test_route_neutral_obligations_do_not_create_commitment(committed):
    ws, primary, root, obligation = committed
    mark(obligation, **{ROUTE_IDS: "[]"})
    context, moves, _ = frontier(committed)
    assert not committed_construction_route_ids(context, ws, (obligation,))
    assert any(m.operation == "develop" and m.target_entity_id == primary for m in moves)


@pytest.mark.parametrize("failure", ["blocked", "failed", "refuted", "superseded", "inactive"])
def test_inactive_route_cannot_own_active_work_or_produce_ordinary_descendants(committed, failure):
    ws, primary, root, obligation = committed
    other = construction(ws, "Live independent construction")
    other_obligation = add_work(ws, "proof_obligation", route=other, related=(primary,))
    if failure == "superseded":
        mark(root, **{SUPERSEDED_BY: json.dumps([other])})
    elif failure == "inactive":
        with connect() as con:
            con.execute("UPDATE entities SET status='abandoned' WHERE id=?", (root,))
    else:
        mark(root, research_branch_status=failure)
    old_lemma = construction(ws, "Failed route precise lemma", "lemma")
    mark(old_lemma, **{ROUTE_IDS: json.dumps([root]), "related_entity_ids": json.dumps([other_obligation])})
    context, moves, _ = frontier(committed)
    assert _open_obligation_ids(context, ws, primary) == (other_obligation,)
    assert {m.focus_obligation_id for m in moves} == {other_obligation}
    assert all(old_lemma not in {m.target_entity_id, *m.consumed_entity_ids} for m in moves)
    for operation in ("develop", "prove", "synthesize"):
        with pytest.raises(TheoryError, match="inactive entity"):
            _validate_construction_choice(context, OperationChoice(operation, old_lemma, "Ordinary descent"))


def test_multiple_live_committed_routes_and_shared_ownership_expose_each_frontier(committed):
    ws, primary, root, obligation = committed
    other = construction(ws, "Second committed construction")
    other_obligation = add_work(ws, "proof_obligation", route=other, related=(primary,))
    shared = add_work(ws, "proof_obligation", related=(primary,))
    mark(shared, **{ROUTE_IDS: json.dumps([root, other])})
    context, moves, _ = frontier(committed)
    assert committed_construction_route_ids(context, ws, (obligation, other_obligation, shared)) == frozenset({root, other})
    assert {m.focus_obligation_id for m in moves} == {obligation, other_obligation, shared}
    mark(root, research_branch_status="failed")
    context, moves, _ = frontier(committed)
    assert _open_obligation_ids(context, ws, primary) == (other_obligation, shared)
    assert committed_construction_route_ids(context, ws, (other_obligation, shared)) == frozenset({other})
    assert {m.focus_obligation_id for m in moves} == {other_obligation, shared}


def test_commitment_may_use_an_owned_construction_descendant(committed):
    ws, primary, root, obligation = committed
    mark(root, research_artifact_type="finding")
    descendant = construction(ws, "A substantive owned component")
    mark(descendant, **{ROUTE_IDS: json.dumps([root])})
    context, moves, _ = frontier(committed)
    assert committed_construction_route_ids(context, ws, (obligation,)) == frozenset({root})
    assert all(m.focus_obligation_id == obligation for m in moves)


def test_failed_positive_premise_rejected_but_explicit_negative_evidence_retained(committed):
    ws, primary, root, obligation = committed
    failed = construction(ws, "Failed implementation")
    mark(failed, **{ROUTE_IDS: json.dumps([root]), "research_branch_status": "blocked"})
    failure = add_work(ws, "obstruction", route=failed, related=(failed,))
    context = for_workstream(ws)
    with pytest.raises(TheoryError, match="inactive entity"):
        _validate_construction_choice(context, OperationChoice(
            "synthesize", obligation, "Attempt ordinary reuse", consumed_entity_ids=(root, failed)))
    _validate_construction_choice(context, OperationChoice(
        "synthesize", obligation, "Use scoped failure evidence", consumed_entity_ids=(root, failure)))
    _validate_construction_choice(context, OperationChoice("reframe", failed, "Explicit repair audit"))


@pytest.mark.parametrize("local_trigger", [False, True])
def test_pending_attack_cannot_receive_ideated_develop_or_top_level_ideas(committed, monkeypatch, local_trigger):
    ws, primary, root, obligation = committed
    proof = add_work(ws, "proof_attempt", route=root, related=(obligation,))
    with connect() as con:
        add_relation(con, proof, "ATTEMPTS", obligation)
    trigger = IdeationTrigger("repeated_obligation_failure" if local_trigger else "concrete_refutation",
                              (obligation,), focus_obligation_id=obligation if local_trigger else None)
    monkeypatch.setattr("theory.research.choose_ideation_trigger", lambda context, *args, **kwargs:
                        trigger if "research_attack_state" not in context.attributes[proof] else None)
    # Two-call allowance reaches the ordinary ideation gate. The pending attack
    # must precede any transient development idea for the same obligation.
    requests, _ = install_providers(monkeypatch, execute=lambda decision, _: step_report(
        decision, [], attack_outcome="inconclusive", unresolved=["Still requires adjudication."]
    ) if decision["operation"] == "attack" else step_report(decision, [artifact(
        "finding", "The local witness still needs a bounded construction.", "local_witness_gap",
        [decision["target_entity_id"]],
    )]))
    result = research(ws, max_calls=2)
    assert result.ideation_calls_made == 0
    with connect() as con:
        first = con.execute("SELECT * FROM research_iterations ORDER BY iteration_number LIMIT 1").fetchone()
        assert json.loads(first["legal_move_ids_json"]) == [f"attack:{proof}:{obligation}:none"]
        assert first["selection_mode"] == "single_legal_move"
        assert not con.execute("SELECT 1 FROM api_calls WHERE purpose='research:ideate'").fetchone()
    assert not any(request[1]["response_model"].__name__ == "IdeaBatch" for request in requests)


@pytest.mark.parametrize("outcome,addressed", [("no_critical_issue", True), ("no_critical_issue", False),
                                              ("critical_issue", True), ("inconclusive", True)])
def test_persisted_prove_then_attack_uses_existing_closure_semantics(committed, monkeypatch, outcome, addressed):
    ws, primary, root, obligation = committed
    lemma = construction(ws, "A precise scoped lemma", "lemma")
    mark(lemma, **{ROUTE_IDS: json.dumps([root]), "related_entity_ids": json.dumps([obligation])})

    def prove(decision, _):
        assert decision["operation"] == "prove" and decision["target_entity_id"] == lemma
        return step_report(decision, [
            artifact("proof_attempt", "A provisional derivation conditional on the local witness.",
                     "scoped_parent_proof", [lemma, obligation]),
            artifact("proof_obligation", "Construct the witness used by the parent derivation.",
                     "scoped_child_witness", [lemma, obligation]),
        ], addressed=[obligation] if addressed else [])

    install_providers(monkeypatch, execute=prove)
    first = research(ws, strategy="off", max_calls=1)
    proof, child = first.artifact_ids
    context, moves, _ = frontier(committed)
    assert any(m.operation == "attack" and m.target_entity_id == proof and m.focus_obligation_id == obligation for m in moves)
    assert child in _open_obligation_ids(context, ws, primary)
    assert context.attributes[proof][ROUTE_IDS] == json.dumps([root])
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (ws,))

    def attack(decision, _):
        assert decision["operation"] == "attack" and decision["target_entity_id"] == proof
        critical = [artifact("obstruction", "The proposed witness omits an admissible boundary.",
                             "boundary_refutation", [proof], branch_status="blocked")] if outcome == "critical_issue" else []
        return step_report(decision, critical, attack_outcome=outcome,
                           unresolved=["The boundary remains undecided."] if outcome == "inconclusive" else [])

    install_providers(monkeypatch, select=lambda state: {
        "selected_move_id": next(m["move_id"] for m in state["legal_moves"] if m["operation"] == "attack"),
        "rationale": "Adjudicate the persisted candidate.",
    }, execute=attack)
    assert research(ws, max_calls=1).calls_made == 1
    after = for_workstream(ws)
    resolved = outcome == "no_critical_issue" and addressed
    assert (obligation not in _open_obligation_ids(after, ws, primary)) is resolved
    assert child in _open_obligation_ids(after, ws, primary)
    assert committed_construction_route_ids(after, ws, _open_obligation_ids(after, ws, primary)) == frozenset({root})
