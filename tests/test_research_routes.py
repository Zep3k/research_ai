"""Ordinary construction ownership survives route changes and controller resume."""
import json

import pytest

from theory.db import connect
from theory.graph import set_attribute
from theory.research import (
    LegalResearchMove, OperationChoice, ResearchSelection, ResearchStepReport,
    _start_iteration, _persist_step, _complete_iteration, build_progress_record,
    _history, _open_obligation_ids, build_research_state, choose_next_operation,
    generate_legal_research_moves, live_construction_route_ids, research,
)
from theory.research_context import for_workstream
from theory.research_report import build_research_report
from theory.research_routes import ROUTE_IDS, STARTED_AT, SUPERSEDED_BY
from test_research import (
    add_linked_research_entity, artifact, init_workspace, make_research_workstream,
    step_report,
)
from test_research_strategy import install_providers


def resume(ws, **kwargs):
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (ws,))
    return research(ws, max_calls=1, **kwargs)


def frontier(ws, primary_id):
    context = for_workstream(ws)
    primary = next(e for e in context.entities if e["id"] == primary_id)
    history = _history(ws)
    moves = generate_legal_research_moves(context, ws, primary, history)
    return context, primary, history, moves, choose_next_operation(context, ws, primary, history)


def historical_frontier_step(ws, primary_id, outputs):
    """Replay a pre-commitment root write receipt, without offering it as a legal move today."""
    context = for_workstream(ws)
    choice = OperationChoice("develop", primary_id, "Historical root construction receipt.",
                             develop_provenance="frontier")
    move = LegalResearchMove.from_choice(choice)
    selection = ResearchSelection(move, "deterministic_baseline", (move.move_id,), choice.rationale)
    iteration = _start_iteration(ws, choice, selection)
    report = ResearchStepReport.model_validate(step_report({
        "operation": "develop", "target_entity_id": primary_id, "required_consumed_entity_ids": [],
    }, outputs))
    persisted = _persist_step(iteration_id=iteration, workstream_id=ws, provider_name="openai",
                              model="gpt-6-sol", choice=choice, context=context, report=report)
    progress = build_progress_record(context_before=context, context_after=for_workstream(ws),
                                    workstream_id=ws, primary_id=primary_id, choice=choice,
                                    report=report, persisted=persisted)
    _complete_iteration(iteration, ws, choice, report, progress)
    return persisted


@pytest.fixture
def routes(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, primary_id = make_research_workstream()
    step = 0
    ids = {}

    def execute(decision, _):
        nonlocal step
        step += 1
        related = [decision["target_entity_id"]]
        if step == 1:
            outputs = [
                artifact("protocol_component", "The cutoff and seal construction collects locked certificates.",
                         "cutoff_root", related, branch_status="unresolved"),
                artifact("proof_obligation", "Exclude conflicts between sealed cutoff certificates.", "cutoff_a", related),
            ]
        elif step == 2:
            outputs = [artifact("proof_obligation", "Bound the maximum delayed seal acknowledgement.", "cutoff_child", related)]
        elif step == 3:
            outputs = [
                artifact("protocol_component", "A nine-round echo construction propagates conflicting evidence before output.",
                         "nine_round_root", related, branch_status="unresolved"),
                artifact("proof_obligation", "Show that the current nine-round evidence reaches every honest party in time.", "timing_b", related),
            ]
        elif step == 4:
            outputs = [artifact("proof_attempt", "Under the stated delivery bound, the final echo makes every honest decision compatible.",
                                "timing_candidate", [primary_id, ids["B"]], branch_status="promising")]
        else:
            assert decision["operation"] == "attack"
            assert decision["target_entity_id"] == ids["candidate"]
            return step_report(decision, [], attack_outcome="inconclusive", unresolved=["Check the boundary delivery schedule."])
        return step_report(decision, outputs,
                           addressed=[ids["B"]] if step == 4 else [])

    install_providers(monkeypatch, execute=execute)
    first = research(ws, strategy="off", max_calls=1)
    ids["old"], ids["A"] = first.artifact_ids
    # Direct obligation development records the same root on its descendant.
    with connect() as con:
        set_attribute(con, ids["old"], "research_branch_status", "promising")
        set_attribute(con, ids["old"], "precise_candidate", "false")
    second = resume(ws, strategy="off")
    ids["child"] = second.artifact_ids[0]
    # These old receipts deliberately left an unadjudicated route. New controllers
    # must still reconstruct their ownership, though that escape is no longer legal.
    third_report = execute({"operation": "develop", "target_entity_id": primary_id,
                            "required_consumed_entity_ids": []}, 0)
    third = historical_frontier_step(ws, primary_id, third_report["artifacts"])
    ids["new"], ids["B"] = third.artifact_ids
    fourth = resume(ws, strategy="off")
    ids["candidate"] = fourth.artifact_ids[0]
    return ws, primary_id, ids


def test_case02b_baseline_and_resume_use_only_current_route(routes):
    ws, primary_id, ids = routes
    context, primary, history, moves, baseline = frontier(ws, primary_id)
    assert _open_obligation_ids(context, ws, primary_id) == (ids["B"],)
    assert baseline.operation == "attack"
    assert (baseline.target_entity_id, baseline.focus_obligation_id) == (ids["candidate"], ids["B"])
    assert all(m.focus_obligation_id not in {ids["A"], ids["child"]} for m in moves)
    assert context.attributes[ids["A"]][ROUTE_IDS] == json.dumps([ids["old"]])
    assert context.attributes[ids["child"]][ROUTE_IDS] == json.dumps([ids["old"]])
    assert context.attributes[ids["candidate"]][ROUTE_IDS] == json.dumps([ids["new"]])
    assert json.loads(context.attributes[ids["old"]][SUPERSEDED_BY]) == [ids["new"]]
    assert live_construction_route_ids(context) == frozenset({ids["new"]})
    report = build_research_report(ws)
    obligations = {o.entity_id: o for o in report.obligations}
    for key in ("A", "child"):
        assert obligations[ids[key]].recorded_state == "open"
        assert obligations[ids[key]].route_inactive_reason == "no_live_owning_construction"
        assert obligations[ids[key]].owning_construction_route_ids == (ids["old"],)
    assert obligations[ids["B"]].live_owning_construction_route_ids == (ids["new"],)
    state = build_research_state(context, workstream_id=ws, primary=primary, history=history, legal_moves=moves)
    reloaded = frontier(ws, primary_id)
    assert reloaded == (context, primary, history, moves, baseline)
    assert state.open_obligations[0].id == ids["B"]
    outcome = resume(ws, strategy="off")
    with connect() as con:
        row = con.execute("SELECT * FROM research_iterations WHERE id=?", (outcome.iteration_ids[0],)).fetchone()
    assert row["selection_mode"] == "deterministic_baseline"
    assert row["selected_move_id"] == f"attack:{ids['candidate']}:{ids['B']}:none"


def test_shared_and_explicit_global_obligations_remain_active(routes):
    ws, primary_id, ids = routes
    shared = add_linked_research_entity(ws, "OpenQuestion", "A common delivery bound", proof_obligation=True)
    global_id = add_linked_research_entity(ws, "OpenQuestion", "Meet the problem contract", proof_obligation=True)
    with connect() as con:
        set_attribute(con, shared, ROUTE_IDS, json.dumps([ids["old"], ids["new"]]))
        set_attribute(con, shared, "related_entity_ids", json.dumps([ids["A"]]))
        set_attribute(con, global_id, ROUTE_IDS, "[]")
        set_attribute(con, global_id, "related_entity_ids", json.dumps([ids["A"]]))
    context = for_workstream(ws)
    assert _open_obligation_ids(context, ws, primary_id) == tuple(sorted([ids["B"], shared, global_id]))
    with connect() as con:
        set_attribute(con, ids["new"], "research_attack_state", "challenged")
    assert _open_obligation_ids(for_workstream(ws), ws, primary_id) == (global_id,)


@pytest.mark.parametrize("failure", ["challenged", "blocked", "inactive", "contradicted"])
def test_root_liveness_uses_persisted_structure(routes, failure):
    ws, primary_id, ids = routes
    with connect() as con:
        if failure == "challenged":
            set_attribute(con, ids["new"], "research_attack_state", failure)
        elif failure == "blocked":
            set_attribute(con, ids["new"], "research_branch_status", failure)
        elif failure == "inactive":
            con.execute("UPDATE entities SET status='abandoned' WHERE id=?", (ids["new"],))
        else:
            con.execute("UPDATE entities SET trust_state='contradicted' WHERE id=?", (ids["new"],))
    context, _, _, moves, baseline = frontier(ws, primary_id)
    assert _open_obligation_ids(context, ws, primary_id) == ()
    assert baseline.target_entity_id not in {ids["old"], ids["candidate"], ids["A"]}
    assert all(m.target_entity_id not in {ids["old"], ids["candidate"]} for m in moves)


@pytest.mark.parametrize("terminal", [False, True])
def test_v14_receipts_backfill_identical_ownership_and_frontier(routes, terminal):
    ws, primary_id, ids = routes
    if terminal:
        with connect() as con:
            set_attribute(con, ids["new"], "research_branch_status", "blocked")
    before = frontier(ws, primary_id)
    report = build_research_report(ws)
    with connect() as con:
        con.execute("DELETE FROM entity_attributes WHERE key LIKE 'research_construction_%'")
        con.execute("DELETE FROM schema_migrations WHERE version=15")
        con.execute("PRAGMA user_version=14")
    # The schema migration replays graph write receipts, never model text.
    after = frontier(ws, primary_id)
    assert after == before
    assert build_research_report(ws) == report
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM entity_attributes WHERE key=?", (STARTED_AT,)).fetchone()[0] == 2


@pytest.mark.parametrize("output", ["obligation_only", "duplicate", "failed_approach"])
def test_historical_frontier_selection_alone_cannot_supersede_current_route(routes, output):
    ws, primary_id, ids = routes

    def execute(decision, _):
        if output == "obligation_only":
            item = artifact("proof_obligation", "Bound the size of the alternative dissemination queue.", "alternate_queue", [primary_id])
        elif output == "duplicate":
            item = artifact("protocol_component", "Repeat the previously recorded echo procedure.", "nine_round_root", [primary_id], branch_status="unresolved")
        else:
            item = artifact("failed_approach", "The proposed star topology allows the faulty hub to suppress every echo.", "star_refuted", [primary_id], branch_status="refuted")
        return step_report(decision, [item])

    historical_frontier_step(ws, primary_id,
                             execute({"operation": "develop", "target_entity_id": primary_id,
                                      "required_consumed_entity_ids": []}, 0)["artifacts"])
    context = for_workstream(ws)
    assert SUPERSEDED_BY not in context.attributes[ids["new"]]
    assert ids["B"] in _open_obligation_ids(context, ws, primary_id)


def test_historical_new_frontier_supersedes_only_the_recorded_execution_route(routes):
    ws, primary_id, ids = routes
    independent = add_linked_research_entity(ws, "Technique", "An independent live construction")
    premise = add_linked_research_entity(ws, "OpenQuestion", "Bound the independent construction", proof_obligation=True)
    with connect() as con:
        set_attribute(con, independent, STARTED_AT, "1")
        set_attribute(con, independent, ROUTE_IDS, json.dumps([independent]))
        set_attribute(con, premise, ROUTE_IDS, json.dumps([independent]))

    outcome = historical_frontier_step(ws, primary_id, [
        artifact("protocol_component", "Collect signed epoch summaries through a rotating coordinator.", "epoch_route", [primary_id], branch_status="unresolved"),
    ])
    context = for_workstream(ws)
    new_root = outcome.artifact_ids[0]
    assert live_construction_route_ids(context) == frozenset({independent, new_root})
    assert json.loads(context.attributes[ids["new"]][SUPERSEDED_BY]) == [new_root]
    assert SUPERSEDED_BY not in context.attributes[independent]
    assert _open_obligation_ids(context, ws, primary_id) == (premise,)
