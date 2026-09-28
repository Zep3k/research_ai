"""Human judgment is limited to ambiguity in an input contract, not failed routes."""
import json

import pytest

from theory.db import connect
from theory.errors import ModelOutputError
from theory.graph import add_entity, link_workstream_entity
from theory.research import (
    OperationChoice, ResearchStepReport, _research_prompt, _validate_step_report, research,
)
from theory.research_context import focus_research_context, for_workstream
from test_research import (
    DynamicProvider, add_linked_research_entity, artifact, current_choice,
    decision_from_prompt, init_workspace, make_research_workstream, step_report,
)


CONTRACT = (
    "Synchronous authenticated network. In each round, at most d incoming and d outgoing "
    "links per party may be corrupted; the set of corrupted links may change between rounds."
)
MISSING_PREMISE = (
    "The generated two-hop cross-round debit argument requires corruption persistence "
    "between rounds, which the supplied model does not assume."
)


@pytest.fixture
def wa(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, primary = make_research_workstream(title="Weak Agreement")
    with connect() as con:
        model = add_entity(con, "Model", "Round-dependent link corruption", body=CONTRACT)
        link_workstream_entity(con, workstream, model, "input")
    obligation = add_linked_research_entity(
        workstream, "OpenQuestion", "Establish a weak-consistency bound", proof_obligation=True,
    )
    candidate = add_linked_research_entity(
        workstream, "Lemma", "Two-hop cross-round debit bound",
        related_entity_ids=(obligation,),
    )
    monkeypatch.setattr("theory.research.get_provider", lambda *_: pytest.fail("Real provider forbidden"))
    return workstream, primary, model, obligation, candidate


def failed_route_report(decision, wa, *, human=False):
    _, _, model, obligation, candidate = wa
    return step_report(
        decision,
        [
            artifact("failed_approach", MISSING_PREMISE, "invalid_cross_round_debit",
                     [candidate, obligation, model], branch_status="failed"),
            artifact("open_question", "Can a round-local counting argument handle changing corrupt links?",
                     "round_local_counting_alternative", [candidate, obligation, model]),
        ],
        unresolved=[MISSING_PREMISE], human=human,
        human_reason="Please assume cross-round persistence to rescue this proof." if human else None,
    )


@pytest.mark.parametrize("operation", ["prove", "attack", "synthesize", "reframe"])
@pytest.mark.parametrize("target_is_input", [False, True])
def test_non_develop_cannot_request_human_judgment_even_on_input(wa, operation, target_is_input):
    workstream, primary, _, _, candidate = wa
    target = primary if target_is_input else candidate
    choice = OperationChoice(operation, target, "Test judgment boundary")
    raw = step_report(
        {"operation": operation, "target_entity_id": target, "required_consumed_entity_ids": []},
        [], human=True, human_reason=MISSING_PREMISE,
    )
    with pytest.raises(ModelOutputError, match="Human judgment is permitted only for develop"):
        _validate_step_report(ResearchStepReport.model_validate(raw), for_workstream(workstream), choice)


def test_generated_obligation_develop_must_encode_missing_premises(wa):
    workstream, primary_id, _, obligation, _ = wa
    context = for_workstream(workstream)
    primary = next(e for e in context.entities if e["id"] == primary_id)
    choice = OperationChoice("develop", obligation, "Explore this branch",
                             open_obligation_ids=(obligation,), focus_obligation_id=obligation)
    prompt = _research_prompt(context, primary, choice)
    assert "human_judgment_required MUST be false" in prompt
    raw = step_report(
        decision_from_prompt(prompt),
        [artifact("obstruction", MISSING_PREMISE, "conditional_branch_missing_premise", [obligation],
                  branch_status="blocked")],
        human=True, human_reason=MISSING_PREMISE,
    )
    with pytest.raises(ModelOutputError, match="Human judgment is permitted only for develop"):
        _validate_step_report(ResearchStepReport.model_validate(raw), context, choice)
    raw.update(human_judgment_required=False, human_judgment_reason=None)
    _validate_step_report(ResearchStepReport.model_validate(raw), context, choice)


def test_wa_invalid_prove_human_request_fails_without_blocking_as_human_judgment(wa, monkeypatch):
    workstream, primary, _, _, candidate = wa
    assert current_choice(workstream, primary).operation == "prove"
    provider = DynamicProvider(lambda decision, _: failed_route_report(decision, wa, human=True))
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    with pytest.raises(ModelOutputError, match="record candidate-specific missing premises in artifacts"):
        research(workstream, max_calls=3, strategy="off")
    assert len(provider.calls) == 1
    with connect() as con:
        row = dict(con.execute("SELECT * FROM research_iterations").fetchone())
        assert (row["operation"], row["target_entity_id"], row["status"]) == ("prove", candidate, "error")
        assert row["stop_reason"] != "human_judgment_required"
        assert json.loads(row["artifact_ids_json"]) == []
        assert con.execute("SELECT status FROM workstreams WHERE id=?", (workstream,)).fetchone()[0] == "error"
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 1


def test_wa_failed_proof_route_continues_to_next_operation_without_extra_calls(wa, monkeypatch):
    workstream, primary_id, model, obligation, candidate = wa
    context = for_workstream(workstream)
    primary = next(e for e in context.entities if e["id"] == primary_id)
    choice = current_choice(workstream, primary_id)
    assert (choice.operation, choice.target_entity_id) == ("prove", candidate)
    focused = focus_research_context(
        context, workstream_id=workstream, primary_entity_id=primary_id,
        target_entity_id=candidate, focus_obligation_id=obligation,
    )
    prompt = _research_prompt(focused, primary, choice)
    assert CONTRACT in prompt
    assert "human_judgment_required MUST be false" in prompt
    assert "Never ask the human to strengthen the contract to save a candidate" in prompt
    _validate_step_report(
        ResearchStepReport.model_validate(failed_route_report(decision_from_prompt(prompt), wa)),
        focused, choice,
    )
    operations = []
    def respond(decision, _):
        operations.append(decision["operation"])
        if len(operations) == 1:
            assert decision["target_entity_id"] == candidate
            return failed_route_report(decision, wa)
        assert decision["operation"] == "synthesize"
        assert decision["target_entity_id"] == obligation
        return step_report(decision, [artifact(
            "obstruction", "A round-local replacement bound still requires a reduction controlling changing corrupt links.",
            "round_local_evidence_gap", [obligation, *decision["required_consumed_entity_ids"]],
            branch_status="unresolved",
        )])
    provider = DynamicProvider(respond)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    outcome = research(workstream, max_calls=2, strategy="off")
    assert operations == ["prove", "synthesize"]
    assert (outcome.calls_made, outcome.strategy_calls_made, outcome.total_api_calls_made) == (2, 0, 2)
    assert len(provider.calls) == 2
    assert outcome.stop_reason == "max_calls_exhausted" and outcome.final_status == "completed"
    with connect() as con:
        rows = con.execute("SELECT status,stop_reason,artifact_ids_json FROM research_iterations ORDER BY id").fetchall()
        assert len(rows) == 2 and all(row["status"] == "completed" for row in rows)
        assert all(row["stop_reason"] != "human_judgment_required" for row in rows)
        first_artifacts = json.loads(rows[0]["artifact_ids_json"])
        kinds = [con.execute("SELECT entity_type FROM entities WHERE id=?", (i,)).fetchone()[0]
                 for i in first_artifacts]
        assert kinds == ["FailedApproach", "OpenQuestion"]
        assert con.execute("SELECT body FROM entities WHERE id=?", (model,)).fetchone()[0] == CONTRACT
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 2
