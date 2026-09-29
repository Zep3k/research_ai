"""Offline protocol milestones exercise production orchestration and persistence."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from eval.protocol_construction.cases import CASES
from eval.protocol_construction.runner import benchmark, run_case
from theory.graph import add_entity, add_relation, link_workstream_entity, set_attribute
from theory.db import connect
from theory.research import (
    LegalResearchMove, OperationChoice, _constructive_continuation,
    _research_prompt_sections, choose_next_operation, generate_legal_research_moves,
)
from theory.research_context import focus_research_context, for_workstream
from test_research import init_workspace, make_research_workstream


@pytest.fixture(scope="module")
def measured():
    return benchmark()


@pytest.mark.parametrize("index", range(len(CASES)), ids=[c.name for c in CASES])
def test_structural_milestones_and_real_call_ledger(measured, index):
    case, result = CASES[index], measured["results"][index]
    assert result["passed"], result
    assert not result["diagnostics"]
    assert result["execution_calls"] == len(case.stages)
    assert result["strategy_calls"] == (1 if case.strategy == "auto" else 0)
    assert result["logged_calls"] == result["execution_calls"] + result["strategy_calls"]
    assert result["stop_reason"] == "max_calls_exhausted"
    assert [t["operation"] for t in result["trace"]] == [s.operation for s in case.stages]
    assert all(t["selected_move"] in t["legal_moves"] for t in result["trace"])
    # Construction milestones are not automatic epistemic closure.
    assert not any(t["resolution_progress"] for t in result["trace"])


def test_obligation_reuse_and_failed_route_preservation(measured):
    queue = measured["results"][-1]
    assert [len(t["open_obligations"]) for t in queue["trace"]] == [1, 1, 1]
    assert [t["duplicate_count"] for t in queue["trace"]] == [0, 1, 0]
    assert queue["trace"][1]["duplicate_keys"] == ["queue_bound"]
    failed = measured["results"][2]
    assert "failed" in failed["trace"][-1]["branch_states"].values()


def test_negative_executor_controls_are_not_misreported_as_success(measured):
    assumption, fanout = measured["negative_controls"]
    assert not assumption["passed"]
    assert not assumption["checks"]["no_forbidden_assumptions"]
    assert assumption["checks"]["final_candidate_structure"]
    assert not fanout["passed"]
    assert not fanout["checks"]["bounded_obligations"]
    assert not fanout["checks"]["final_candidate_structure"]
    assert fanout["execution_calls"] == 3
    assert all(not t["selected_move"].endswith(":continue") for t in fanout["trace"])


def test_deterministic_benchmark_snapshot_and_workspace_isolation(measured, monkeypatch, tmp_path):
    sentinel = tmp_path / "do-not-touch"
    sentinel.write_text("existing workspace")
    monkeypatch.chdir(tmp_path)
    assert benchmark() == measured
    assert Path.cwd() == tmp_path
    assert list(tmp_path.iterdir()) == [sentinel]
    expected = json.loads((Path(__file__).parents[1] / "eval/protocol_construction/after.json").read_text())
    assert measured == expected


def construction_context(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, primary = make_research_workstream()
    with connect() as con:
        partial = add_entity(con, "Technique", "Unfinished receiver", trust_state="quarantined")
        link_workstream_entity(con, ws, partial, "created")
        for key, value in {"research_artifact_type": "protocol_component", "research_branch_status": "unresolved", "related_entity_ids": json.dumps([primary]), "precise_candidate": "true"}.items():
            set_attribute(con, partial, key, value)
    history = ({"status": "completed", "operation": "develop", "target_entity_id": primary,
                "focus_obligation_id": None, "material_progress": 1,
                "artifact_ids_json": json.dumps([partial])},)
    return ws, primary, partial, history


@pytest.mark.parametrize("reason", ["duplicate", "obligation_only", "finished", "challenged", "blocked", "inactive", "contradicted", "intervening_attack", "error", "focus_closed", "counterexample"])
def test_continuation_requires_new_live_unfinished_component(monkeypatch, tmp_path, reason):
    ws, primary, partial, history = construction_context(monkeypatch, tmp_path)
    context = for_workstream(ws)
    assert _constructive_continuation(context, ws, history, ()) is not None
    row = dict(history[0])
    with connect() as con:
        if reason == "duplicate":
            row.update(material_progress=0, artifact_ids_json="[]")
        elif reason == "obligation_only":
            set_attribute(con, partial, "research_artifact_type", "proof_obligation")
        elif reason == "finished":
            set_attribute(con, partial, "research_branch_status", "promising")
        elif reason == "challenged":
            set_attribute(con, partial, "research_attack_state", "challenged")
        elif reason == "blocked":
            set_attribute(con, partial, "research_branch_status", "blocked")
        elif reason in {"inactive", "contradicted"}:
            column, value = ("status", "abandoned") if reason == "inactive" else ("trust_state", "contradicted")
            con.execute(f"UPDATE entities SET {column}=? WHERE id=?", (value, partial))
        elif reason == "intervening_attack":
            row["operation"] = "attack"
        elif reason == "error":
            row["status"] = "error"
        elif reason == "focus_closed":
            row["focus_obligation_id"] = partial
        elif reason == "counterexample":
            failure = add_entity(con, "Counterexample", "Breaks receiver")
            link_workstream_entity(con, ws, failure, "created")
            add_relation(con, failure, "REFUTES", partial)
    assert _constructive_continuation(for_workstream(ws), ws, (row,), ()) is None


def test_continuation_has_distinct_move_and_stable_safe_prompt(monkeypatch, tmp_path):
    ws, primary, partial, history = construction_context(monkeypatch, tmp_path)
    with connect() as con:
        obligation = add_entity(con, "OpenQuestion", "Unproved invariant")
        link_workstream_entity(con, ws, obligation, "created")
        set_attribute(con, obligation, "is_proof_obligation", "true")
        set_attribute(con, obligation, "research_obligation_state", "open")
    context = for_workstream(ws)
    root = next(e for e in context.entities if e["id"] == primary)
    choice = choose_next_operation(context, ws, root, history)
    assert choice.continue_construction
    assert not choice.idea_origin
    assert choice.open_obligation_ids == (obligation,)
    moves = generate_legal_research_moves(context, ws, root, history)
    same_target = [m for m in moves if m.operation == "develop" and m.target_entity_id == primary]
    assert len(same_target) == 2  # Constructive continuation and a genuine branch escape.
    assert len({m.move_id for m in same_target}) == 2
    assert LegalResearchMove.from_choice(choice).to_operation_choice() == choice
    sections = _research_prompt_sections(context, root, choice)
    assert "Continue the same materially advancing protocol route" in sections.stable_prefix
    assert "Do not merely refine, rename, or continue" not in sections.stable_prefix
    assert "Do not hide" in sections.stable_prefix
    assert "no requirement to produce multiple branches" in sections.stable_prefix
    changed = replace(choice, open_obligation_ids=(555,), rationale="A later iteration")
    changed_context = replace(context, entities=tuple({**e, "body": "Changed graph data"} for e in context.entities))
    assert sections.stable_prefix == _research_prompt_sections(changed_context, root, changed).stable_prefix
    fallback = replace(choice, continue_construction=False)
    assert "Do not merely refine, rename, or continue" in _research_prompt_sections(context, root, fallback).stable_prefix


def test_continuation_carries_route_origin_without_idea_content(monkeypatch, tmp_path):
    ws, primary, _, history = construction_context(monkeypatch, tmp_path)
    context = for_workstream(ws)
    root = next(entity for entity in context.entities if entity["id"] == primary)
    ordinary = choose_next_operation(context, ws, root, history)
    idea = choose_next_operation(context, ws, root, ({**history[0], "idea_origin": 1},))

    assert ordinary.continue_construction and not ordinary.idea_origin
    assert idea.continue_construction and idea.idea_origin
    assert idea.idea is None and idea.ideation_call_id is None
    assert LegalResearchMove.from_choice(idea).to_operation_choice() == idea
    assert _research_prompt_sections(context, root, ordinary).render() == (
        _research_prompt_sections(context, root, idea).render()
    )


def test_failed_attempt_does_not_interrupt_scientific_construction(monkeypatch, tmp_path):
    ws, primary, partial, history = construction_context(monkeypatch, tmp_path)
    context = for_workstream(ws)
    root = next(entity for entity in context.entities if entity["id"] == primary)
    failed_attack = {"status": "error", "operation": "attack", "target_entity_id": partial}
    baseline = choose_next_operation(context, ws, root, history)
    moves = generate_legal_research_moves(context, ws, root, history)

    assert baseline.continue_construction
    assert choose_next_operation(context, ws, root, (*history, failed_attack)) == baseline
    assert generate_legal_research_moves(context, ws, root, (*history, failed_attack)) == moves


def test_dependency_closure_handles_cycles_without_reverse_branch_expansion(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, primary = make_research_workstream()
    with connect() as con:
        ids = [add_entity(con, "Technique", name, trust_state="quarantined") for name in ("storage", "merge", "send", "unrelated")]
        for entity in ids:
            link_workstream_entity(con, ws, entity, "created")
        storage, merge, send, unrelated = ids
        set_attribute(con, send, "related_entity_ids", json.dumps([merge]))
        set_attribute(con, merge, "related_entity_ids", json.dumps([storage]))
        set_attribute(con, storage, "related_entity_ids", json.dumps([merge, primary, 999999]))
        set_attribute(con, unrelated, "related_entity_ids", json.dumps([primary, storage]))
    full = for_workstream(ws)
    focused = focus_research_context(full, workstream_id=ws, primary_entity_id=primary, target_entity_id=send)
    assert {e["id"] for e in focused.entities} == {primary, storage, merge, send}
    assert focused == focus_research_context(full, workstream_id=ws, primary_entity_id=primary, target_entity_id=send)
    assert focused.epistemic["quarantined"].entities
    assert full == for_workstream(ws)


def test_strategy_can_continue_a_route_without_reframing():
    case = replace(CASES[-1], strategy="auto")
    result = run_case(case)
    assert result["passed"], result
    assert result["execution_calls"] == 3
    assert result["strategy_calls"] <= result["execution_calls"]
    assert all(t["selected_move"].endswith(":continue") for t in result["trace"][1:])
    assert result["logged_calls"] == result["execution_calls"] + result["strategy_calls"]


@pytest.mark.parametrize("kind", ["protocol_component", "proof_attempt"])
def test_completed_candidate_alongside_partial_component_does_not_force_continuation(monkeypatch, tmp_path, kind):
    ws, primary, partial, history = construction_context(monkeypatch, tmp_path)
    with connect() as con:
        candidate = add_entity(con, "Technique" if kind == "protocol_component" else "ProofAttempt", "Assembled candidate", trust_state="quarantined")
        link_workstream_entity(con, ws, candidate, "created")
        set_attribute(con, candidate, "research_artifact_type", kind)
        set_attribute(con, candidate, "research_branch_status", "promising")
    row = {**history[0], "artifact_ids_json": json.dumps([partial, candidate])}
    assert _constructive_continuation(for_workstream(ws), ws, (row,), ()) is None
