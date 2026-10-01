"""Bound construction continuations using persisted route-local receipts."""
import json
from dataclasses import replace

import pytest

from theory.config import Config
from theory.db import connect, utcnow
from theory.graph import add_entity, link_workstream_entity, set_attribute
from theory.research import (
    MAX_CONSTRUCTIVE_CONTINUATIONS, LegalResearchMove, _constructive_continuation,
    _constructive_continuation_streak, _history, _research_prompt_sections,
    _route_consolidation_choice, choose_next_operation, generate_legal_research_moves,
    research,
)
from theory.research_context import for_workstream
from theory.research_routes import ROUTE_IDS, STARTED_AT
from test_research import artifact, init_workspace, make_research_workstream, step_report
from test_research_strategy import install_providers


@pytest.fixture(autouse=True)
def no_real_providers(monkeypatch):
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Real provider forbidden"))


def append_step(ws, primary_id, *, route=None, continuation=False, operation="develop",
                artifact_type="protocol_component", branch_status="unresolved", status="completed",
                extra_types=()):
    """Reconstruct one completed controller receipt with durable route ancestry."""
    with connect() as con:
        number = con.execute("SELECT COALESCE(MAX(iteration_number),0)+1 FROM research_iterations WHERE workstream_id=?", (ws,)).fetchone()[0]
        ids = []
        if status == "completed":
            for kind in (artifact_type, *extra_types):
                entity_id = add_entity(con, {"protocol_component": "Technique", "lemma": "Lemma",
                                             "finding": "Finding", "proof_attempt": "ProofAttempt",
                                             "proof_obligation": "OpenQuestion"}[kind], f"{kind} at stage {number}",
                                       trust_state="quarantined")
                link_workstream_entity(con, ws, entity_id, "created")
                ids.append(entity_id)
                for key, value in {
                    "research_artifact_type": kind, "related_entity_ids": json.dumps([primary_id]),
                    "research_material_key": f"{kind}_{number}",
                    "research_branch_status": branch_status,
                    ROUTE_IDS: json.dumps([route or ids[0]]),
                }.items():
                    set_attribute(con, entity_id, key, value)
                if kind in {"protocol_component", "lemma", "proof_attempt"}:
                    set_attribute(con, entity_id, "precise_candidate", "true")
                if kind == "proof_obligation":
                    set_attribute(con, entity_id, "is_proof_obligation", "true")
                    set_attribute(con, entity_id, "research_obligation_state", "open")
        move = f"{operation}:{primary_id}:none:none" + (":continue" if continuation else ":idea:1:echo_route" if operation == "develop" else "")
        receipt = con.execute(
            "INSERT INTO research_iterations(project_id,workstream_id,iteration_number,operation,target_entity_id,"
            "rationale,status,created_at,completed_at,material_progress,artifact_ids_json,selected_move_id,develop_provenance) "
            "VALUES(1,?,?,?,?,?,?,?,?,?,?,?,'idea')",
            (ws, number, operation, primary_id, "Reconstructed bounded step.", status, utcnow(), utcnow(),
             int(bool(ids)), json.dumps(ids), move),
        ).lastrowid
        if ids and route is None:
            set_attribute(con, ids[0], STARTED_AT, str(receipt))
    return tuple(ids)


@pytest.fixture
def construction(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, primary_id = make_research_workstream()
    root, = append_step(ws, primary_id)
    return ws, primary_id, root


def frontier(ws, primary_id):
    context = for_workstream(ws)
    primary = next(e for e in context.entities if e["id"] == primary_id)
    history = _history(ws)
    return context, primary, history, generate_legal_research_moves(context, ws, primary, history), choose_next_operation(context, ws, primary, history)


def reach_checkpoint(ws, primary_id, root):
    created = [root]
    for streak in range(MAX_CONSTRUCTIVE_CONTINUATIONS):
        context, _, history, _, baseline = frontier(ws, primary_id)
        assert _constructive_continuation_streak(context, history) == (frozenset({root}), streak)
        assert baseline.continue_construction
        created.extend(append_step(ws, primary_id, route=root, continuation=True))
    return tuple(created)


def test_case02b_continuation_limit_and_route_consolidation(construction):
    ws, primary_id, root = construction
    components = reach_checkpoint(ws, primary_id, root)
    context, primary, history, moves, baseline = frontier(ws, primary_id)
    assert MAX_CONSTRUCTIVE_CONTINUATIONS == 3
    assert _constructive_continuation_streak(context, history) == (frozenset({root}), 3)
    assert _constructive_continuation(context, ws, history, ()) is None
    assert all(not move.continue_construction for move in moves)
    assert baseline.operation == "synthesize" and baseline.target_entity_id == primary_id
    assert baseline.consumed_entity_ids == tuple(reversed(components))
    assert baseline.focus_obligation_id is None
    assert "testable candidate and/or explicit obligations" in baseline.rationale
    assert LegalResearchMove.from_choice(baseline) in moves
    assert any(move.operation == "develop" and move.develop_provenance == "frontier" for move in moves)
    assert frontier(ws, primary_id) == (context, primary, history, moves, baseline)
    # Reuse the existing primary-synthesis prompt and model routing.
    assert _research_prompt_sections(context, primary, baseline).stable_prefix == _research_prompt_sections(
        context, primary, replace(baseline, rationale="Combine the selected branch results.")
    ).stable_prefix
    assert all(context.attributes[i]["research_branch_status"] == "unresolved" for i in components)


def test_consolidation_uses_latest_live_route_artifacts_and_stable_order(construction):
    ws, primary_id, root = construction
    # Many unrelated artifacts share the contract input but have a distinct owner.
    other, = append_step(ws, primary_id)
    for _ in range(3):
        append_step(ws, primary_id, route=other, continuation=True)
    primed, = append_step(ws, primary_id, route=root)
    components = reach_checkpoint(ws, primary_id, root)
    with connect() as con:
        finding = add_entity(con, "Finding", "Current-route timing consequence", trust_state="quarantined")
        unrelated = add_entity(con, "Technique", "Newer unrelated construction", trust_state="quarantined")
        global_id = add_entity(con, "Finding", "Contract-global note", trust_state="quarantined")
        for i, owners, kind in ((finding, [root], "finding"), (unrelated, [other], "protocol_component"), (global_id, [], "finding")):
            link_workstream_entity(con, ws, i, "created")
            set_attribute(con, i, ROUTE_IDS, json.dumps(owners))
            set_attribute(con, i, "research_artifact_type", kind)
        set_attribute(con, components[-1], "research_attack_state", "challenged")
        set_attribute(con, components[-2], "research_branch_status", "blocked")
    context, primary, history, _, baseline = frontier(ws, primary_id)
    choice = _route_consolidation_choice(context, ws, primary_id, history, ())
    assert choice.consumed_entity_ids == (finding, components[1], primed, root)
    assert choice == _route_consolidation_choice(
        replace(context, entities=context.entities[::-1]), ws, primary_id, history, (),
    )
    assert all(set(json.loads(context.attributes[i][ROUTE_IDS])) == {root} for i in choice.consumed_entity_ids)


@pytest.mark.parametrize("operation", ["synthesize", "prove", "attack", "reframe"])
def test_other_operations_reset_streak(construction, operation):
    ws, primary_id, root = construction
    reach_checkpoint(ws, primary_id, root)
    append_step(ws, primary_id, route=root, operation=operation, artifact_type="finding")
    context, _, history, _, _ = frontier(ws, primary_id)
    assert _constructive_continuation_streak(context, history)[1] == 0
    assert _route_consolidation_choice(context, ws, primary_id, history, ()) is None
    append_step(ws, primary_id, route=root)
    context, _, history, _, baseline = frontier(ws, primary_id)
    assert baseline.continue_construction
    assert _constructive_continuation_streak(context, history)[1] == 0


@pytest.mark.parametrize("kind,status", [("proof_obligation", "unresolved"), ("proof_attempt", "unresolved"),
                                         ("protocol_component", "promising"), ("lemma", "promising")])
def test_candidate_or_obligation_creation_resets_streak(construction, kind, status):
    ws, primary_id, root = construction
    for _ in range(2):
        append_step(ws, primary_id, route=root, continuation=True)
    append_step(ws, primary_id, route=root, continuation=True, branch_status=status, extra_types=(kind,))
    context, _, history, _, _ = frontier(ws, primary_id)
    assert _constructive_continuation_streak(context, history)[1] == 0
    assert (_constructive_continuation(context, ws, history, ()) is not None) == (kind == "proof_obligation")


def test_route_change_starts_a_fresh_streak(construction):
    ws, primary_id, root = construction
    reach_checkpoint(ws, primary_id, root)
    other, = append_step(ws, primary_id)
    components = reach_checkpoint(ws, primary_id, other)
    context, _, history, _, baseline = frontier(ws, primary_id)
    assert _constructive_continuation_streak(context, history) == (frozenset({other}), 3)
    assert baseline.consumed_entity_ids == tuple(reversed(components))
    assert root not in baseline.consumed_entity_ids


def test_shared_ancestry_preserves_the_current_route_streak(construction):
    ws, primary_id, root = construction
    first, = append_step(ws, primary_id, route=root, continuation=True)
    second, = append_step(ws, primary_id, route=root, continuation=True)
    with connect() as con:
        set_attribute(con, first, ROUTE_IDS, json.dumps([root, 999]))
        set_attribute(con, second, ROUTE_IDS, json.dumps([root, 998]))
    context, _, history, _, baseline = frontier(ws, primary_id)
    assert _constructive_continuation_streak(context, history) == (frozenset({root}), 2)
    assert baseline.continue_construction


def test_ordinary_unfinished_develop_cannot_renew_an_exhausted_route_streak(construction):
    ws, primary_id, root = construction
    reach_checkpoint(ws, primary_id, root)
    append_step(ws, primary_id, route=root)
    context, _, history, moves, baseline = frontier(ws, primary_id)
    assert _constructive_continuation_streak(context, history) == (frozenset({root}), 3)
    assert all(not move.continue_construction for move in moves)
    assert baseline.operation == "synthesize"


def test_committed_route_checkpoint_defers_to_pending_attack(construction):
    ws, primary_id, root = construction
    components = reach_checkpoint(ws, primary_id, root)
    with connect() as con:
        obligation = add_entity(con, "OpenQuestion", "An unresolved timing premise", trust_state="quarantined")
        proof = add_entity(con, "ProofAttempt", "A precise timing argument", trust_state="quarantined")
        for i in (obligation, proof):
            link_workstream_entity(con, ws, i, "created")
            set_attribute(con, i, ROUTE_IDS, json.dumps([root]))
        set_attribute(con, obligation, "is_proof_obligation", "true")
        set_attribute(con, obligation, "research_obligation_state", "candidate_pending_attack")
        set_attribute(con, proof, "research_artifact_type", "proof_attempt")
        set_attribute(con, proof, "research_focus_obligation_id", str(obligation))
    context, _, history, moves, baseline = frontier(ws, primary_id)
    assert baseline.operation == "attack" and baseline.target_entity_id == proof
    choice = _route_consolidation_choice(context, ws, primary_id, history, (obligation,))
    assert choice.consumed_entity_ids == tuple(reversed(components))
    assert LegalResearchMove.from_choice(choice) not in moves
    assert {move.operation for move in moves} == {"attack"}
    assert LegalResearchMove.from_choice(baseline) in moves


@pytest.mark.parametrize("failure", ["error", "running"])
def test_failed_develops_do_not_advance_or_reset_streak(construction, failure):
    ws, primary_id, root = construction
    append_step(ws, primary_id, route=root, continuation=True)
    before = frontier(ws, primary_id)
    append_step(ws, primary_id, route=root, continuation=True, status=failure)
    context, _, history, moves, baseline = frontier(ws, primary_id)
    assert _constructive_continuation_streak(context, history) == (frozenset({root}), 1)
    assert moves == before[3] and baseline == before[4]
    assert _constructive_continuation(context, ws, history, ()) == baseline


def test_real_controller_resumes_idea_streak_and_executes_existing_synthesis(construction, monkeypatch):
    ws, primary_id, root = construction
    decisions = []

    def execute(decision, number):
        decisions.append(decision)
        number = len(decisions)
        if decision["operation"] == "develop":
            item = artifact("protocol_component", [
                "Allocate each epoch's signed messages to a bounded evidence buffer.",
                "On receiving a relay, propagate conflicting signatures before making a decision.",
                "At the final deadline, release only decisions consistent with all delivered echoes.",
            ][number - 1], f"echo_piece_{number}", [primary_id], branch_status="unresolved")
        else:
            assert number == 4 and decision["operation"] == "synthesize"
            item = artifact("proof_obligation", "Establish the final echo delivery deadline under the contract.", "echo_deadline",
                            [primary_id, *decision["required_consumed_entity_ids"]])
        return step_report(decision, [item])

    requests, _ = install_providers(monkeypatch, execute=execute)
    for _ in range(3):
        with connect() as con:
            con.execute("UPDATE workstreams SET status='active' WHERE id=?", (ws,))
        assert research(ws, strategy="off", max_calls=1).calls_made == 1
    before = frontier(ws, primary_id)
    assert before[-1].operation == "synthesize"
    assert frontier(ws, primary_id) == before
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (ws,))
    outcome = research(ws, strategy="off", max_calls=1)
    context = for_workstream(ws)
    assert _constructive_continuation_streak(context, _history(ws))[1] == 0
    assert json.loads(context.attributes[outcome.artifact_ids[0]][ROUTE_IDS]) == [root]
    assert [d["operation"] for d in decisions] == ["develop"] * 3 + ["synthesize"]
    assert decisions[-1]["required_consumed_entity_ids"] == list(before[-1].consumed_entity_ids)
    with connect() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM research_iterations WHERE workstream_id=? ORDER BY iteration_number", (ws,))]
        calls = [dict(r) for r in con.execute("SELECT purpose,model,status FROM api_calls ORDER BY id")]
    assert [r["selected_move_id"].endswith(":continue") for r in rows] == [False, True, True, True, False]
    assert [r["develop_provenance"] for r in rows[:4]] == ["idea"] * 4
    assert calls[-1]["purpose"] == "research:synthesize" and calls[-1]["status"] == "completed"
    assert calls[-1]["model"] == Config.load().research_synthesize_model
    assert len(requests) == 4
