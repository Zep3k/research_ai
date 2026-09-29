"""Offline cross-branch hypothesis formation using the existing synthesis operation."""
import copy
import json

import pytest

from theory.db import connect
from theory.errors import ModelOutputError
from theory.graph import add_entity, link_workstream_entity, set_attribute
from theory.research import (
    LegalResearchMove, ResearchStepReport, _open_obligation_ids, _research_prompt,
    _validate_step_report, build_research_state, choose_next_operation,
    generate_legal_research_moves, research,
    _strategist_prompt,
)
from theory.research_context import focus_research_context, for_workstream
from test_research import (
    add_linked_research_entity, artifact, completed_history, decision_from_prompt,
    init_workspace, make_research_workstream, step_report,
)
from test_research_strategy import install_providers


CONTRACT = "Construct an object satisfying P without assuming the stronger invariant Q."


def generated(workstream, kind, statement, key, *, branch="unresolved", related=()):
    with connect() as con:
        entity_id = add_entity(con, kind, statement, body=statement,
                               trust_state="quarantined", generated_by_llm=True)
        link_workstream_entity(con, workstream, entity_id, "created")
        set_attribute(con, entity_id, "research_artifact_type", {
            "Obstruction": "obstruction", "Technique": "protocol_component", "Finding": "finding",
            "Lemma": "lemma", "ProofAttempt": "proof_attempt", "Counterexample": "counterexample",
        }[kind])
        set_attribute(con, entity_id, "research_material_key", key)
        set_attribute(con, entity_id, "research_branch_status", branch)
        set_attribute(con, entity_id, "related_entity_ids", json.dumps(related))
    return entity_id


@pytest.fixture
def branches(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Real provider forbidden"))
    workstream, primary = make_research_workstream(title="Construct P")
    with connect() as con:
        contract = add_entity(con, "Definition", "Exact contract", body=CONTRACT)
        link_workstream_entity(con, workstream, contract, "input")
    a = generated(workstream, "Obstruction", "A countermodel rules out the strong uniform invariant Q.",
                  "uniform_invariant_obstruction", branch="blocked", related=(primary,))
    b = generated(workstream, "Technique", "A staged local repair mechanism permits conditional construction.",
                  "staged_repair_mechanism", branch="promising", related=(primary,))
    c = generated(workstream, "Finding", "The aggregate resource bound imposes an independent feasibility constraint.",
                  "resource_feasibility", related=(primary,))
    return workstream, primary, contract, a, b, c


def planning(branches, history=(), *, strategy_enabled=True):
    workstream, primary_id, *_ = branches
    context = for_workstream(workstream)
    primary = next(e for e in context.entities if e["id"] == primary_id)
    moves = generate_legal_research_moves(context, workstream, primary, history,
                                          strategy_enabled=strategy_enabled)
    return context, primary, moves


def primary_moves(moves, primary):
    return [m for m in moves if m.operation == "synthesize" and m.target_entity_id == primary
            and m.focus_obligation_id is None]


def test_primary_synthesis_persists_joint_branch_without_mutating_inputs(branches, monkeypatch):
    workstream, primary_id, contract, a, b, c = branches
    context, primary, moves = planning(branches)
    baseline = choose_next_operation(context, workstream, primary, ())
    assert baseline.operation == "develop"
    assert moves[0] == LegalResearchMove.from_choice(baseline)
    bundles = primary_moves(moves, primary_id)
    assert 1 <= len(bundles) <= 3
    selected = next(m for m in bundles if m.consumed_entity_ids == (a, b))
    assert selected.move_id == f"synthesize:{primary_id}:none:{a},{b}"
    assert all(2 <= len(m.consumed_entity_ids) <= 4 for m in bundles)
    snapshot = copy.deepcopy(context.as_dict())
    state = build_research_state(context, workstream_id=workstream, primary=primary,
                                history=(), legal_moves=moves)
    assert {e.id: e.title for e in state.move_entities}[a] == next(e["title"] for e in context.entities if e["id"] == a)
    assert generate_legal_research_moves(context, workstream, primary, ()) == moves
    assert context.as_dict() == snapshot

    def execute(decision, _):
        assert decision["operation"] == "synthesize"
        assert decision["target_entity_id"] == primary_id
        assert decision["required_consumed_entity_ids"] == [a, b]
        assert decision["required_artifact_related_entity_ids"] == [a, b, primary_id]
        return step_report(decision, [artifact(
            "protocol_component", "Apply staged repair only outside the countermodel's forbidden uniformity regime.",
            "conditional_repair_route", [primary_id, a, b],
        ), artifact(
            "proof_obligation", "Establish that the conditional repair regime still satisfies the exact contract P.",
            "conditional_repair_premise", [primary_id, a, b], epistemic_status="unresolved",
        )])

    requests, _ = install_providers(monkeypatch,
        select=lambda _: {"selected_move_id": selected.move_id, "rationale": "Use the negative invariant result to constrain the repair mechanism."},
        execute=execute)
    outcome = research(workstream, max_calls=1)
    assert (outcome.calls_made, outcome.strategy_calls_made, outcome.total_api_calls_made) == (1, 1, 2)
    assert outcome.stop_reason == "max_calls_exhausted"
    execution = requests[1][1]
    assert (execution["model"], execution["effort"], execution["max_output_tokens"]) == ("gpt-6-sol", "high", 12_000)
    assert CONTRACT in execution["prompt"]
    assert "Synthesize the selected artifacts against the exact problem contract." in execution["prompt"]
    assert "Do not assume quarantined artifacts are true" in execution["prompt"]
    assert "attempt to close obligation" not in execution["prompt"]
    assert "Consider primary synthesis when existing branches contain complementary results" in requests[0][1]["prompt"]
    after = for_workstream(workstream)
    for entity_id in (a, b, c):
        assert next(e for e in after.entities if e["id"] == entity_id) == next(e for e in context.entities if e["id"] == entity_id)
        assert after.attributes[entity_id] == context.attributes[entity_id]
    assert after.relations == context.relations  # No closure/bypass/retirement relation.
    assert len(outcome.artifact_ids) == 2
    for entity_id in outcome.artifact_ids:
        assert set(json.loads(after.attributes[entity_id]["related_entity_ids"])) == {primary_id, a, b}
    assert _open_obligation_ids(after, workstream, primary_id) == (outcome.artifact_ids[1],)
    with connect() as con:
        history = tuple(dict(row) for row in con.execute("SELECT * FROM research_iterations"))
    assert history[0]["resolution_progress"] == 0
    assert history[0]["new_obligation_count"] == 1
    assert selected.move_id not in {m.move_id for m in planning(branches, history)[2]}


def test_primary_bundles_filter_duplicates_and_irrelevant_artifacts_and_stay_bounded(branches):
    workstream, primary_id, _, a, b, c = branches
    duplicate_key = generated(workstream, "Technique", "An alias for the same repair construction.", "staged_repair_mechanism")
    duplicate_text = generated(workstream, "Finding", "The aggregate resource bound imposes an independent feasibility constraint!", "paraphrased_resource")
    terminal = generated(workstream, "Technique", "This positive candidate was refuted.", "refuted_candidate", branch="refuted")
    inactive = generated(workstream, "Obstruction", "Archived negative evidence.", "archived_negative")
    contradicted = generated(workstream, "Counterexample", "A disproven counterexample.", "wrong_counterexample")
    nongenerated = add_linked_research_entity(workstream, "Technique", "Manually supplied material")
    with connect() as con:
        con.execute("UPDATE entities SET status='abandoned' WHERE id=?", (inactive,))
        con.execute("UPDATE entities SET trust_state='contradicted' WHERE id=?", (contradicted,))
    # A large frontier still generates only a fixed number of pairs, without combinations.
    for n in range(80):
        generated(workstream, "Finding", f"Independent constraint {n} with distinct parameter {n * 17}.", f"constraint_{n}", related=(1000 + n,))
    context, primary, moves = planning(branches)
    bundles = primary_moves(moves, primary_id)
    assert 1 <= len(bundles) <= 3
    assert len({m.move_id for m in moves}) == len(moves)
    used = {i for m in bundles for i in m.consumed_entity_ids}
    assert {duplicate_key, duplicate_text, terminal, inactive, contradicted, nongenerated}.isdisjoint(used)
    assert a in used and b in used
    history = tuple(completed_history("synthesize", primary_id, consumed_entity_ids=tuple(reversed(m.consumed_entity_ids))) for m in bundles)
    next_bundles = primary_moves(generate_legal_research_moves(context, workstream, primary, history), primary_id)
    assert len(next_bundles) <= 3
    assert {frozenset(m.consumed_entity_ids) for m in bundles}.isdisjoint(
        frozenset(m.consumed_entity_ids) for m in next_bundles)


def test_same_role_requires_distinct_recorded_branches(branches):
    workstream, primary_id, _, a, b, c = branches
    with connect() as con:
        con.execute("UPDATE entities SET status='abandoned' WHERE id IN (?,?)", (a, b))
    d = generated(workstream, "Finding", "An algebraic parity constraint narrows the construction.", "parity_constraint", related=(primary_id,))
    context, primary, moves = planning(branches)
    assert primary_moves(moves, primary_id) == []
    history = tuple({**completed_history("develop", primary_id), "artifact_ids_json": json.dumps([i])} for i in (c, d))
    assert [m.consumed_entity_ids for m in primary_moves(generate_legal_research_moves(context, workstream, primary, history), primary_id)] == [(c, d)]


@pytest.mark.parametrize("stage", ["obligation", "prove", "attack"])
def test_primary_synthesis_coexists_with_local_moves_and_escape(branches, stage):
    workstream, primary_id, _, _, b, c = branches
    target = add_linked_research_entity(workstream, "OpenQuestion" if stage == "obligation" else
        "Lemma" if stage == "prove" else "ProofAttempt", "Local target", proof_obligation=stage == "obligation")
    if stage == "obligation":
        with connect() as con:
            for entity_id in (b, c):
                set_attribute(con, entity_id, "related_entity_ids", json.dumps([target]))
    context, primary, moves = planning(branches)
    baseline = choose_next_operation(context, workstream, primary, ())
    assert baseline.operation == ("synthesize" if stage == "obligation" else stage)
    assert LegalResearchMove.from_choice(baseline) in moves
    assert any(m.move_id == f"develop:{primary_id}:none:none" for m in moves)
    assert primary_moves(moves, primary_id)
    if stage == "obligation":
        assert baseline.focus_obligation_id == target
        assert "attempt to close obligation" in _research_prompt(context, primary, baseline)
        assert any(m.operation == "reframe" for m in moves)


@pytest.mark.parametrize("provider,strategy", [("auto", "off"), ("openai", "auto"), ("anthropic", "auto")])
def test_primary_synthesis_is_strategy_only(branches, monkeypatch, provider, strategy):
    workstream, primary_id, *_ = branches
    context, primary, moves = planning(branches, strategy_enabled=False)
    assert moves == (LegalResearchMove.from_choice(choose_next_operation(context, workstream, primary, ())),)
    requests, _ = install_providers(monkeypatch)
    outcome = research(workstream, provider, strategy=strategy, max_calls=1)
    assert outcome.strategy_calls_made == 0 and len(requests) == 1
    assert decision_from_prompt(requests[0][1]["prompt"])["operation"] == "develop"


@pytest.mark.parametrize("invalid", ["consumed", "references", "addressed", "human"])
def test_primary_synthesis_validation_preserves_reference_and_judgment_gates(branches, invalid):
    workstream, primary_id, _, a, b, _ = branches
    obligation = add_linked_research_entity(workstream, "OpenQuestion", "An old premise", proof_obligation=True)
    context, primary, moves = planning(branches)
    choice = primary_moves(moves, primary_id)[0].to_operation_choice()
    decision = decision_from_prompt(_research_prompt(context, primary, choice))
    payload = step_report(decision, [artifact("lemma", "A conditional joint hypothesis.", "joint_hypothesis",
                                             [primary_id, *choice.consumed_entity_ids, obligation])])
    _validate_step_report(ResearchStepReport.model_validate(payload), context, choice)
    if invalid == "consumed":
        payload["consumed_entity_ids"] = [a]
    elif invalid == "references":
        payload["artifacts"][0]["related_entity_ids"] = [primary_id]
    elif invalid == "addressed":
        payload["addressed_obligation_ids"] = [obligation]
    else:
        payload["human_judgment_required"] = True
        payload["human_judgment_reason"] = "A candidate premise is missing."
    with pytest.raises(ModelOutputError):
        _validate_step_report(ResearchStepReport.model_validate(payload), context, choice)
    assert context.attributes[obligation].get("research_obligation_state", "open") == "open"


@pytest.mark.parametrize("kind", ["finding", "protocol_component", "lemma", "proof_attempt",
                                  "obstruction", "failed_approach", "proof_obligation"])
def test_primary_synthesis_accepts_joint_results_without_obligation_closure(branches, kind):
    _, primary_id, *_ = branches
    context, primary, moves = planning(branches)
    choice = primary_moves(moves, primary_id)[0].to_operation_choice()
    decision = decision_from_prompt(_research_prompt(context, primary, choice))
    payload = step_report(decision, [artifact(kind, "A joint consequence of the selected evidence.",
        "joint_evidence_result", [primary_id, *choice.consumed_entity_ids])])
    payload["consumed_entity_ids"].reverse()  # Exact set, independent of returned order.
    _validate_step_report(ResearchStepReport.model_validate(payload), context, choice)


def test_primary_synthesis_duplicate_output_does_not_close_old_obligations(branches, monkeypatch):
    workstream, primary_id, _, a, b, _ = branches
    obligation = add_linked_research_entity(workstream, "OpenQuestion", "Existing unresolved premise", proof_obligation=True)
    context, _, moves = planning(branches)
    selected = primary_moves(moves, primary_id)[0]
    requests, _ = install_providers(monkeypatch,
        select=lambda _: {"selected_move_id": selected.move_id, "rationale": "Reconcile complementary pieces."},
        execute=lambda decision, _: step_report(decision, [artifact(
            "protocol_component", "A rephrased existing repair mechanism.", "staged_repair_mechanism",
            [primary_id, *selected.consumed_entity_ids],
        )]))
    outcome = research(workstream, max_calls=1)
    assert outcome.artifact_ids == () and len(requests) == 2
    after = for_workstream(workstream)
    assert after.attributes[obligation] == context.attributes[obligation]
    assert _open_obligation_ids(after, workstream, primary_id) == (obligation,)
    with connect() as con:
        row = dict(con.execute("SELECT * FROM research_iterations").fetchone())
    assert row["duplicate_count"] == 1 and row["resolution_progress"] == 0
    assert row["progress_class"] == "duplicate_only"
    assert selected.move_id not in {m.move_id for m in planning(branches, (row,))[2]}


def test_planning_exact_statements_and_execution_full_consumed_bodies(branches):
    workstream, primary_id, contract, a, b, c = branches
    exact = "For every admissible object, Q fails on the following witness.\n" * 100
    a_body = exact + "\n\nReasoning: FULL_DERIVATION_A with all witness details."
    legacy_body = "The repair mechanism is conditional on a compatible boundary.\n" * 100 + "LEGACY_MECHANISM_DETAIL"
    unrelated = generated(workstream, "Finding", "Unrelated abandoned history", "unrelated_history")
    with connect() as con:
        con.execute("UPDATE entities SET title='Truncated obstruction',body=? WHERE id=?", (a_body, a))
        set_attribute(con, a, "research_statement", exact)
        con.execute("UPDATE entities SET title='Short mechanism title',body=? WHERE id=?", (legacy_body, b))
        con.execute("UPDATE entities SET status='abandoned',body=? WHERE id=?",
                    ("UNRELATED_HISTORICAL_BODY " * 1000, unrelated))
    context, primary, moves = planning(branches)
    selected = next(m for m in primary_moves(moves, primary_id) if m.consumed_entity_ids == (a, b))
    state = build_research_state(context, workstream_id=workstream, primary=primary, history=(), legal_moves=moves)
    represented = {e.id: e for e in state.move_entities}
    assert represented[a].statement == exact
    assert represented[b].statement == legacy_body
    assert next(e.body for e in state.problem_contract if e.id == contract) == CONTRACT
    prompt = _strategist_prompt(state)
    assert "FULL_DERIVATION_A" not in prompt  # The persisted statement takes precedence.
    assert "LEGACY_MECHANISM_DETAIL" in prompt
    assert "UNRELATED_HISTORICAL_BODY" not in prompt
    assert unrelated not in represented
    consumed_context = focus_research_context(context, workstream_id=workstream,
        primary_entity_id=primary_id, target_entity_id=primary_id,
        consumed_entity_ids=selected.consumed_entity_ids)
    execution_bodies = {e["id"]: e["body"] for e in consumed_context.entities}
    assert execution_bodies[a] == a_body and execution_bodies[b] == legacy_body
    assert execution_bodies[contract] == CONTRACT
    execution_prompt = _research_prompt(consumed_context, primary, selected.to_operation_choice())
    assert decision_from_prompt(execution_prompt)["required_consumed_entity_ids"] == [a, b]
    assert "FULL_DERIVATION_A" in execution_prompt and "LEGACY_MECHANISM_DETAIL" in execution_prompt


@pytest.mark.parametrize("kind,operation", [("Lemma", "prove"), ("ProofAttempt", "attack")])
def test_local_move_candidates_also_expose_exact_statements(branches, kind, operation):
    workstream, primary_id, *_ = branches
    candidate = generated(workstream, kind, "Short local candidate title", "local_candidate")
    statement = "Exact local candidate statement with essential quantifiers.\n" * 80
    with connect() as con:
        set_attribute(con, candidate, "research_statement", statement)
        con.execute("UPDATE entities SET body='FULL_LOCAL_REASONING' WHERE id=?", (candidate,))
    context, primary, moves = planning(branches)
    assert any(m.operation == operation and m.target_entity_id == candidate for m in moves)
    state = build_research_state(context, workstream_id=workstream, primary=primary, history=(), legal_moves=moves)
    assert next(e.statement for e in state.move_entities if e.id == candidate) == statement
    assert "FULL_LOCAL_REASONING" not in _strategist_prompt(state)
