import json

import pytest
from typer.testing import CliRunner

from theory.cli import app
from theory.config import Config
from theory.db import connect, initialize, monthly_spend
from theory.errors import BudgetExceededError, ModelOutputError, TheoryError
from theory.graph import (
    add_entity,
    create_workstream,
    link_workstream_entity,
    set_attribute,
)
from theory.models import ModelResult
from theory.research import (
    OperationChoice,
    RESEARCH_MAX_OUTPUT_TOKENS,
    ResearchArtifact,
    ResearchStepReport,
    _relevant_synthesis_inputs,
    _research_prompt,
    _validate_step_report,
    choose_next_operation,
    research,
)
from theory.research_context import for_workstream


def decision_from_prompt(prompt: str) -> dict:
    payload = prompt.split("CONTROLLER DECISION\n", 1)[1].split(
        "\n\nGRAPH CONTEXT", 1
    )[0]
    return json.loads(payload)


def artifact(
    artifact_type: str,
    statement: str,
    material_key: str,
    related_entity_ids: list[int],
    *,
    branch_status=None,
    epistemic_status="speculation",
):
    return {
        "artifact_type": artifact_type,
        "statement": statement,
        "reasoning_summary": f"Technical reasoning for {material_key}.",
        "material_key": material_key,
        "epistemic_status": epistemic_status,
        "related_entity_ids": related_entity_ids,
        "source_ids": [],
        "branch_status": branch_status,
    }


def step_report(
    decision: dict,
    artifacts: list[dict],
    *,
    addressed=None,
    attack_outcome="not_applicable",
    unresolved=None,
    human=False,
    human_reason=None,
):
    return {
        "operation": decision["operation"],
        "target_entity_id": decision["target_entity_id"],
        "summary": f"Completed bounded {decision['operation']} operation.",
        "artifacts": artifacts,
        "consumed_entity_ids": decision["required_consumed_entity_ids"],
        "addressed_obligation_ids": addressed or [],
        "attack_outcome": attack_outcome,
        "could_not_determine": unresolved or [],
        "human_judgment_required": human,
        "human_judgment_reason": human_reason,
    }


class DynamicProvider:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responder(decision_from_prompt(kwargs["prompt"]), len(self.calls))
        return ModelResult(
            text=json.dumps(response),
            input_tokens=500,
            output_tokens=250,
            cost_usd=0.007,
        )


class RaisingProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        raise TimeoutError("mock controller timeout")


def init_workspace(monkeypatch, tmp_path, *, budget=100.0):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    initialize("Research controller tests")
    Config(monthly_budget_usd=budget).save()


def make_research_workstream(entity_type="ResearchIdea", title="Promising direction"):
    with connect() as con:
        target = add_entity(con, entity_type, title)
        workstream = create_workstream(con, "research", "Advance and test the direction")
        link_workstream_entity(con, workstream, target, "input")
    return workstream, target


def completed_history(operation, target_entity_id, *, consumed_entity_ids=()):
    return {
        "status": "completed",
        "operation": operation,
        "target_entity_id": target_entity_id,
        "consumed_entity_ids_json": json.dumps(consumed_entity_ids),
    }


def current_choice(workstream, target, history=()):
    context = for_workstream(workstream)
    primary = next(entity for entity in context.entities if int(entity["id"]) == target)
    return choose_next_operation(context, workstream, primary, tuple(history))


def add_linked_research_entity(
    workstream,
    entity_type,
    title,
    *,
    role="created",
    related_entity_ids=(),
    proof_obligation=False,
):
    with connect() as con:
        entity_id = add_entity(con, entity_type, title)
        link_workstream_entity(con, workstream, entity_id, role)
        if related_entity_ids:
            set_attribute(
                con,
                entity_id,
                "related_entity_ids",
                json.dumps(related_entity_ids),
            )
        if proof_obligation:
            set_attribute(con, entity_id, "is_proof_obligation", "true")
    return entity_id


class MinimalResearchContext:
    sources = ()

    def __init__(self, *entity_ids):
        self.entities = tuple({"id": value} for value in (entity_ids or (1,)))

    def as_model_payload(self):
        return {}


@pytest.mark.parametrize("operation", ["develop", "synthesize", "prove"])
def test_non_attack_prompts_require_not_applicable_attack_outcome(operation):
    prompt = _research_prompt(
        MinimalResearchContext(),
        {"id": 1},
        OperationChoice(operation, 1, "Regression test"),
    )

    assert (
        '- For non-attack operations, attack_outcome MUST be exactly "not_applicable".'
        in prompt
    )
    assert '"attack_outcome": "not_applicable"' in prompt
    assert "critical_issue|no_critical_issue|inconclusive" not in prompt


def test_attack_prompt_excludes_not_applicable_attack_outcome():
    prompt = _research_prompt(
        MinimalResearchContext(),
        {"id": 1},
        OperationChoice("attack", 1, "Regression test"),
    )

    assert (
        'attack_outcome MUST be "critical_issue", "no_critical_issue", or "inconclusive"; '
        'it MUST NOT be "not_applicable".' in prompt
    )
    assert (
        '"attack_outcome": "critical_issue|no_critical_issue|inconclusive"' in prompt
    )
    assert '"attack_outcome": "not_applicable"' not in prompt


def test_synthesize_prompt_requires_controller_consumed_ids_in_any_order():
    prompt = _research_prompt(
        MinimalResearchContext(1, 4, 8, 12, 17),
        {"id": 1},
        OperationChoice(
            "synthesize",
            17,
            "Regression test",
            consumed_entity_ids=(4, 8, 12),
        ),
    )
    decision = decision_from_prompt(prompt)

    assert (
        "consumed_entity_ids MUST contain exactly the controller-selected IDs [4, 8, 12], "
        "in any order" in prompt
    )
    assert decision["required_consumed_entity_ids"] == [4, 8, 12]
    assert decision["required_artifact_related_entity_ids"] == [4, 8, 12, 17]
    assert '"consumed_entity_ids": [4, 8, 12]' in prompt
    assert '"related_entity_ids": [4, 8, 12, 17]' in prompt
    assert "EVERY artifact emitted by this synthesis operation MUST include EVERY" in prompt
    assert "minimum required related_entity_ids are exactly: [4, 8, 12, 17]" in prompt
    assert "successful synthesis artifacts AND failed_approach or obstruction" in prompt
    assert "they must also occur in each" in prompt
    assert "ADDRESSED OBLIGATION RULES" in prompt
    assert "ONLY if THIS response also emits" in prompt
    assert "artifact_type lemma or proof_attempt" in prompt
    assert "does NOT by itself count as addressing an obligation" in prompt
    assert "Never claim an obligation is addressed merely because it is the selected target" in prompt
    assert "The selected target obligation is #17" in prompt
    assert "SUCCESS CASE" in prompt
    assert "Then and only then include #17 in addressed_obligation_ids" in prompt
    assert "FAILURE / PARTIAL-PROGRESS CASE" in prompt
    assert "set addressed_obligation_ids to []" in prompt
    assert 'Do not call partial progress "addressed"' in prompt
    assert "Otherwise leave addressed_obligation_ids empty" in prompt
    assert '"addressed_obligation_ids": []' in prompt


@pytest.mark.parametrize("operation", ["develop", "attack", "prove"])
def test_non_synthesis_prompts_require_empty_consumed_ids(operation):
    prompt = _research_prompt(
        MinimalResearchContext(),
        {"id": 1},
        OperationChoice(operation, 1, "Regression test"),
    )

    decision = decision_from_prompt(prompt)

    assert "For non-synthesis operations, consumed_entity_ids MUST be []." in prompt
    assert decision["required_consumed_entity_ids"] == []
    assert decision["required_artifact_related_entity_ids"] == [1]
    assert '"consumed_entity_ids": []' in prompt
    assert '"related_entity_ids": [1]' in prompt
    assert "minimum required related_entity_ids" not in prompt


def test_research_prompt_and_schema_bound_artifact_output():
    prompt = _research_prompt(
        MinimalResearchContext(),
        {"id": 1},
        OperationChoice("develop", 1, "Regression test"),
    )

    assert "Return at most 4 substantive artifacts" in prompt
    assert "Keep each reasoning_summary concise and technical" in prompt
    assert (
        ResearchStepReport.model_json_schema()["properties"]["artifacts"]["maxItems"]
        == 4
    )


def test_research_prompt_explains_branch_status_compatibility():
    prompt = _research_prompt(
        MinimalResearchContext(),
        {"id": 1},
        OperationChoice("develop", 1, "Regression test"),
    )

    assert "BRANCH STATUS RULES" in prompt
    assert '"blocked" is legal ONLY for obstruction or failed_approach' in prompt
    assert '"failed" and "refuted" are legal ONLY for failed_approach' in prompt
    assert "do NOT mark that substantive artifact blocked" in prompt
    assert "emit a separate obstruction artifact" in prompt
    assert "represent that failure as a failed_approach" in prompt


def report_with_artifact_status(artifact_type, branch_status):
    decision = {
        "operation": "develop",
        "target_entity_id": 1,
        "required_consumed_entity_ids": [],
    }
    return ResearchStepReport.model_validate(
        step_report(
            decision,
            [
                artifact(
                    artifact_type,
                    f"Artifact of type {artifact_type} for branch-status validation.",
                    f"{artifact_type}_branch_status_validation",
                    [1],
                    branch_status=branch_status,
                )
            ],
        )
    )


@pytest.mark.parametrize(
    ("artifact_type", "branch_status"),
    [
        ("parameter_analysis", "blocked"),
        ("obstruction", "failed"),
    ],
)
def test_artifact_union_rejects_incompatible_branch_status(
    artifact_type, branch_status
):
    with pytest.raises(ValueError):
        report_with_artifact_status(artifact_type, branch_status)


@pytest.mark.parametrize(
    ("artifact_type", "branch_status"),
    [
        ("parameter_analysis", "unresolved"),
        ("parameter_analysis", None),
        ("obstruction", "blocked"),
        ("failed_approach", "blocked"),
        ("failed_approach", "failed"),
        ("failed_approach", "refuted"),
    ],
)
def test_artifact_union_accepts_compatible_branch_status(
    artifact_type, branch_status
):
    parsed = report_with_artifact_status(artifact_type, branch_status).artifacts[0]

    assert isinstance(parsed, ResearchArtifact)
    assert parsed.artifact_type == artifact_type
    assert parsed.branch_status == branch_status


def test_research_artifact_base_keeps_defensive_branch_status_validator():
    payload = artifact(
        "parameter_analysis",
        "A parameter analysis that reveals a separate blocker.",
        "defensive_branch_status_validation",
        [1],
        branch_status="blocked",
    )

    with pytest.raises(
        ValueError,
        match="blocked branches must be obstruction or failed_approach artifacts",
    ):
        ResearchArtifact.model_validate(payload)


def test_substantive_artifact_plus_separate_blocked_obstruction_validates():
    decision = {
        "operation": "develop",
        "target_entity_id": 1,
        "required_consumed_entity_ids": [],
    }
    report = ResearchStepReport.model_validate(
        step_report(
            decision,
            [
                artifact(
                    "parameter_analysis",
                    "The threshold inequality fails in the boundary regime.",
                    "boundary_threshold_analysis",
                    [1],
                    branch_status="unresolved",
                ),
                artifact(
                    "obstruction",
                    "The boundary regime blocks the proposed construction.",
                    "boundary_regime_obstruction",
                    [1],
                    branch_status="blocked",
                ),
            ],
        )
    )

    _validate_step_report(
        report,
        MinimalResearchContext(),
        OperationChoice("develop", 1, "Regression test"),
    )
    assert [item.artifact_type for item in report.artifacts] == [
        "parameter_analysis",
        "obstruction",
    ]


@pytest.mark.parametrize(
    ("operation", "attack_outcome", "error"),
    [
        ("develop", "critical_issue", "Only attack may return an attack outcome"),
        ("synthesize", "no_critical_issue", "Only attack may return an attack outcome"),
        ("prove", "inconclusive", "Only attack may return an attack outcome"),
        ("attack", "not_applicable", "Attack output must state its bounded attack outcome"),
    ],
)
def test_attack_outcome_validation_remains_operation_specific(
    operation, attack_outcome, error
):
    choice = OperationChoice(operation, 1, "Regression test")
    report = ResearchStepReport(
        operation=operation,
        target_entity_id=1,
        summary="Regression test report.",
        artifacts=[],
        consumed_entity_ids=[],
        addressed_obligation_ids=[],
        attack_outcome=attack_outcome,
        could_not_determine=[],
        human_judgment_required=False,
        human_judgment_reason=None,
    )

    with pytest.raises(ModelOutputError, match=error):
        _validate_step_report(report, MinimalResearchContext(), choice)


@pytest.mark.parametrize(
    ("operation", "target_entity_id", "error"),
    [
        ("attack", 1, "Research output chose 'attack', expected 'develop'"),
        ("develop", 2, "Research output targeted entity #2, expected #1"),
    ],
)
def test_semantic_validation_rejects_incorrect_operation_or_target(
    operation, target_entity_id, error
):
    report = ResearchStepReport(
        operation=operation,
        target_entity_id=target_entity_id,
        summary="Invalid semantic report.",
        artifacts=[],
        consumed_entity_ids=[],
        addressed_obligation_ids=[],
        attack_outcome="inconclusive" if operation == "attack" else "not_applicable",
        could_not_determine=[],
        human_judgment_required=False,
        human_judgment_reason=None,
    )

    with pytest.raises(ModelOutputError, match=error):
        _validate_step_report(
            report,
            MinimalResearchContext(1, 2),
            OperationChoice("develop", 1, "Regression test"),
        )


def synthesis_report(consumed_entity_ids):
    return ResearchStepReport(
        operation="synthesize",
        target_entity_id=1,
        summary="Regression test synthesis.",
        artifacts=[
            artifact(
                "failed_approach",
                "The selected artifacts cannot be combined.",
                "consumed_id_validation_failure",
                [1, 2, 3],
                branch_status="failed",
            )
        ],
        consumed_entity_ids=consumed_entity_ids,
        addressed_obligation_ids=[],
        attack_outcome="not_applicable",
        could_not_determine=[],
        human_judgment_required=False,
        human_judgment_reason=None,
    )


def synthesis_reference_report(artifact_type, related_entity_ids):
    successful = artifact_type == "proof_attempt"
    branch_status = {
        "failed_approach": "failed",
        "obstruction": "blocked",
    }.get(artifact_type)
    return ResearchStepReport(
        operation="synthesize",
        target_entity_id=17,
        summary="Regression test synthesis references.",
        artifacts=[
            artifact(
                artifact_type,
                f"Synthesis artifact of type {artifact_type}.",
                f"{artifact_type}_synthesis_reference_validation",
                related_entity_ids,
                branch_status=branch_status,
            )
        ],
        consumed_entity_ids=[4, 8, 12],
        addressed_obligation_ids=[17] if successful else [],
        attack_outcome="not_applicable",
        could_not_determine=[],
        human_judgment_required=False,
        human_judgment_reason=None,
    )


SYNTHESIS_REFERENCE_CHOICE = OperationChoice(
    "synthesize",
    17,
    "Regression test",
    consumed_entity_ids=(4, 8, 12),
    open_obligation_ids=(17,),
)


def addressed_obligation_report(artifact_type, related_entity_ids):
    return ResearchStepReport(
        operation="develop",
        target_entity_id=1,
        summary="Regression test addressed obligation.",
        artifacts=[
            artifact(
                artifact_type,
                f"Candidate artifact of type {artifact_type}.",
                f"{artifact_type}_addressed_obligation_validation",
                related_entity_ids,
            )
        ],
        consumed_entity_ids=[],
        addressed_obligation_ids=[8],
        attack_outcome="not_applicable",
        could_not_determine=[],
        human_judgment_required=False,
        human_judgment_reason=None,
    )


ADDRESSED_OBLIGATION_CHOICE = OperationChoice(
    "develop",
    1,
    "Regression test",
    open_obligation_ids=(8,),
)


@pytest.mark.parametrize("artifact_type", ["synthesis", "finding"])
def test_non_proof_artifact_cannot_address_obligation(artifact_type):
    with pytest.raises(
        ModelOutputError,
        match="Addressed obligation #8 lacks a lemma or proof attempt",
    ):
        _validate_step_report(
            addressed_obligation_report(artifact_type, [1, 8]),
            MinimalResearchContext(1, 8),
            ADDRESSED_OBLIGATION_CHOICE,
        )


@pytest.mark.parametrize("artifact_type", ["lemma", "proof_attempt"])
def test_proof_artifact_can_address_referenced_open_obligation(artifact_type):
    _validate_step_report(
        addressed_obligation_report(artifact_type, [1, 8]),
        MinimalResearchContext(1, 8),
        ADDRESSED_OBLIGATION_CHOICE,
    )


def test_lemma_cannot_address_obligation_it_does_not_reference():
    with pytest.raises(
        ModelOutputError,
        match="Addressed obligation #8 lacks a lemma or proof attempt",
    ):
        _validate_step_report(
            addressed_obligation_report("lemma", [1]),
            MinimalResearchContext(1, 8),
            ADDRESSED_OBLIGATION_CHOICE,
        )


def successful_addressed_synthesis_report(related_entity_ids):
    return ResearchStepReport(
        operation="synthesize",
        target_entity_id=8,
        summary="Concrete candidate proof of obligation eight.",
        artifacts=[
            artifact(
                "lemma",
                "The consumed artifacts imply the target obligation.",
                "successful_synthesis_lemma",
                related_entity_ids,
            )
        ],
        consumed_entity_ids=[2, 4, 7],
        addressed_obligation_ids=[8],
        attack_outcome="not_applicable",
        could_not_determine=[],
        human_judgment_required=False,
        human_judgment_reason=None,
    )


SUCCESSFUL_SYNTHESIS_CHOICE = OperationChoice(
    "synthesize",
    8,
    "Regression test",
    consumed_entity_ids=(2, 4, 7),
    open_obligation_ids=(8,),
)


def test_successful_synthesis_lemma_addresses_target_obligation():
    _validate_step_report(
        successful_addressed_synthesis_report([2, 4, 7, 8]),
        MinimalResearchContext(2, 4, 7, 8),
        SUCCESSFUL_SYNTHESIS_CHOICE,
    )


def test_successful_synthesis_lemma_still_requires_every_consumed_reference():
    with pytest.raises(
        ModelOutputError,
        match="did not reference every consumed artifact and the target obligation",
    ):
        _validate_step_report(
            successful_addressed_synthesis_report([2, 4, 8]),
            MinimalResearchContext(2, 4, 7, 8),
            SUCCESSFUL_SYNTHESIS_CHOICE,
        )


@pytest.mark.parametrize(
    ("related_entity_ids", "accepted"),
    [
        ([17], False),
        ([4, 8, 12, 17], True),
        ([17, 12, 4, 8], True),
        ([4, 8, 17], False),
    ],
)
def test_synthesis_artifact_requires_all_consumed_and_target_references(
    related_entity_ids, accepted
):
    report = synthesis_reference_report("proof_attempt", related_entity_ids)
    context = MinimalResearchContext(4, 8, 12, 17)

    if accepted:
        _validate_step_report(report, context, SYNTHESIS_REFERENCE_CHOICE)
    else:
        with pytest.raises(
            ModelOutputError,
            match="did not reference every consumed artifact and the target obligation",
        ):
            _validate_step_report(report, context, SYNTHESIS_REFERENCE_CHOICE)


@pytest.mark.parametrize("artifact_type", ["failed_approach", "obstruction"])
def test_unsuccessful_synthesis_artifacts_require_all_references(artifact_type):
    context = MinimalResearchContext(4, 8, 12, 17)
    complete_report = synthesis_reference_report(
        artifact_type, [4, 8, 12, 17]
    )

    assert complete_report.addressed_obligation_ids == []
    _validate_step_report(
        complete_report,
        context,
        SYNTHESIS_REFERENCE_CHOICE,
    )
    with pytest.raises(
        ModelOutputError,
        match="did not reference every consumed artifact and the target obligation",
    ):
        _validate_step_report(
            synthesis_reference_report(artifact_type, [4, 8, 17]),
            context,
            SYNTHESIS_REFERENCE_CHOICE,
        )


def test_synthesize_accepts_reordered_controller_consumed_ids():
    choice = OperationChoice(
        "synthesize", 1, "Regression test", consumed_entity_ids=(2, 3)
    )

    _validate_step_report(
        synthesis_report([3, 2]), MinimalResearchContext(1, 2, 3), choice
    )


@pytest.mark.parametrize(
    ("consumed_entity_ids", "context_ids", "error"),
    [
        ([2], (1, 2, 3), "exactly the controller-selected consumed_entity_ids"),
        ([2, 3, 4], (1, 2, 3, 4), "exactly the controller-selected consumed_entity_ids"),
        ([2, 3, 99], (1, 2, 3), "unknown/out-of-context entity IDs: 99"),
    ],
)
def test_synthesize_rejects_invalid_consumed_id_sets(
    consumed_entity_ids, context_ids, error
):
    choice = OperationChoice(
        "synthesize", 1, "Regression test", consumed_entity_ids=(2, 3)
    )

    with pytest.raises(ModelOutputError, match=error):
        _validate_step_report(
            synthesis_report(consumed_entity_ids),
            MinimalResearchContext(*context_ids),
            choice,
        )


def test_synthesize_rejects_duplicate_consumed_ids():
    with pytest.raises(ValueError, match="IDs must not be duplicated"):
        synthesis_report([2, 2])


@pytest.mark.parametrize("operation", ["develop", "attack", "prove"])
def test_non_synthesis_rejects_non_empty_consumed_ids(operation):
    report = ResearchStepReport(
        operation=operation,
        target_entity_id=1,
        summary="Regression test report.",
        artifacts=[],
        consumed_entity_ids=[2],
        addressed_obligation_ids=[],
        attack_outcome="inconclusive" if operation == "attack" else "not_applicable",
        could_not_determine=[],
        human_judgment_required=False,
        human_judgment_reason=None,
    )

    with pytest.raises(
        ModelOutputError,
        match="Only synthesize may return non-empty consumed_entity_ids",
    ):
        _validate_step_report(
            report,
            MinimalResearchContext(1, 2),
            OperationChoice(operation, 1, "Regression test"),
        )


def test_open_obligation_prioritizes_newest_relevant_unattacked_proof_attempt(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Close the remaining inequality",
        proof_obligation=True,
    )
    older_attempt = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "First candidate proof",
        related_entity_ids=(obligation,),
    )
    newer_attempt = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "Second candidate proof",
        related_entity_ids=(obligation,),
    )

    choice = current_choice(workstream, target)

    assert choice.operation == "attack"
    assert choice.target_entity_id == newer_attempt
    assert choice.target_entity_id != older_attempt
    assert choice.open_obligation_ids == (obligation,)


def test_proof_attempt_is_never_selected_as_a_prove_target(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Prove the remaining claim",
        proof_obligation=True,
    )
    attempt = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "Already attempted proof",
        related_entity_ids=(obligation,),
    )
    history = [completed_history("attack", attempt)]

    choice = current_choice(workstream, target, history)

    assert choice.operation == "develop"
    assert not (
        choice.operation == "prove" and choice.target_entity_id == attempt
    )


def test_attacked_attempt_allows_non_attempt_lemma_to_be_proved(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Finish the composition proof",
        proof_obligation=True,
    )
    attempt = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "Attempt needing critique",
        related_entity_ids=(obligation,),
    )
    lemma = add_linked_research_entity(
        workstream,
        "Lemma",
        "A precise composition lemma",
    )
    context = for_workstream(workstream)
    consumed = _relevant_synthesis_inputs(context, workstream, obligation)
    history = [
        completed_history("attack", attempt),
        completed_history(
            "synthesize", obligation, consumed_entity_ids=consumed
        ),
    ]

    choice = current_choice(workstream, target, history)

    assert choice.operation == "prove"
    assert choice.target_entity_id == lemma
    assert choice.target_entity_id != attempt


def test_synthesis_repeats_only_after_consumed_input_set_changes(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Combine the available bounds",
        proof_obligation=True,
    )
    attempt = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "An attacked candidate proof",
        related_entity_ids=(obligation,),
    )
    add_linked_research_entity(workstream, "Finding", "First relevant bound")
    attacked_history = [completed_history("attack", attempt)]
    first_choice = current_choice(workstream, target, attacked_history)
    assert first_choice.operation == "synthesize"

    same_inputs_history = [
        *attacked_history,
        completed_history(
            "synthesize",
            obligation,
            consumed_entity_ids=first_choice.consumed_entity_ids,
        ),
    ]
    repeated_choice = current_choice(workstream, target, same_inputs_history)
    assert repeated_choice.operation == "develop"

    new_input = add_linked_research_entity(
        workstream, "Finding", "A genuinely new relevant bound"
    )
    changed_choice = current_choice(workstream, target, same_inputs_history)
    assert changed_choice.operation == "synthesize"
    assert new_input in changed_choice.consumed_entity_ids
    assert set(changed_choice.consumed_entity_ids) != set(
        first_choice.consumed_entity_ids
    )


def test_unrelated_unattacked_proof_attempt_is_not_prioritized(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    current_obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Current open obligation",
        proof_obligation=True,
    )
    other_obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Already addressed historical obligation",
        proof_obligation=True,
    )
    unrelated_attempt = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "Attempt for a different obligation",
        related_entity_ids=(other_obligation,),
    )
    with connect() as con:
        set_attribute(
            con,
            unrelated_attempt,
            "addresses_obligation_ids",
            json.dumps([other_obligation]),
        )

    choice = current_choice(workstream, target)

    assert choice.open_obligation_ids == (current_obligation,)
    assert not (
        choice.operation == "attack"
        and choice.target_entity_id == unrelated_attempt
    )


def test_completed_no_issue_attack_does_not_address_open_obligation(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Still requires a constructive proof",
        proof_obligation=True,
    )
    attempt = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "Candidate that survived one attack",
        related_entity_ids=(obligation,),
    )
    attack = completed_history("attack", attempt)
    attack["attack_outcome"] = "no_critical_issue"

    choice = current_choice(workstream, target, [attack])

    assert obligation in choice.open_obligation_ids


def test_prove_cannot_emit_free_standing_attempt_while_obligation_stays_open():
    report = ResearchStepReport(
        operation="prove",
        target_entity_id=1,
        summary="Another attempt without an obligation transition.",
        artifacts=[
            artifact(
                "proof_attempt",
                "A descendant proof attempt repeats the unresolved route.",
                "descendant_unresolved_proof_attempt",
                [1, 8],
            )
        ],
        consumed_entity_ids=[],
        addressed_obligation_ids=[],
        attack_outcome="not_applicable",
        could_not_determine=[],
        human_judgment_required=False,
        human_judgment_reason=None,
    )

    with pytest.raises(
        ModelOutputError,
        match="Prove with open obligations must address one",
    ):
        _validate_step_report(
            report,
            MinimalResearchContext(1, 8),
            OperationChoice(
                "prove", 1, "Regression test", open_obligation_ids=(8,)
            ),
        )


def test_controller_adapts_develop_synthesize_attack_and_stops_successfully(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    with connect() as con:
        add_entity(con, "ResearchIdea", "UNRELATED PROJECT HISTORY SENTINEL")

    def responder(decision, call_number):
        selected = decision["target_entity_id"]
        if call_number == 1:
            assert decision["operation"] == "develop"
            return step_report(
                decision,
                [
                    artifact(
                        "lemma",
                        "Every decision certificate contains an honest signer.",
                        "honest_signer_lemma",
                        [selected],
                        epistemic_status="inference",
                    ),
                    artifact(
                        "protocol_component",
                        "Aggregate threshold certificates before the decision step.",
                        "threshold_certificate_branch",
                        [selected],
                        branch_status="promising",
                    ),
                    artifact(
                        "proof_obligation",
                        "Prove certificate availability despite t silent processes.",
                        "certificate_availability_obligation",
                        [selected],
                        epistemic_status="unresolved",
                    ),
                    artifact(
                        "failed_approach",
                        "Digest-only compression loses signer attribution.",
                        "digest_only_failed_branch",
                        [selected],
                        branch_status="failed",
                        epistemic_status="inference",
                    ),
                ],
            )
        if call_number == 2:
            assert decision["operation"] == "synthesize"
            consumed = decision["required_consumed_entity_ids"]
            assert len(consumed) >= 2
            related = [selected, *consumed]
            return step_report(
                decision,
                [
                    artifact(
                        "proof_attempt",
                        "Combine honest overlap with threshold aggregation to construct a certificate.",
                        "certificate_availability_proof_attempt",
                        related,
                        epistemic_status="inference",
                    )
                ],
                addressed=[selected],
            )
        assert call_number == 3
        assert decision["operation"] == "attack"
        return step_report(
            decision,
            [],
            attack_outcome="no_critical_issue",
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    def network_must_not_run(*args, **kwargs):
        raise AssertionError("research controller must not retrieve from the network")

    monkeypatch.setattr("theory.openalex.search_works", network_must_not_run)
    monkeypatch.setattr("theory.workflows.search_works", network_must_not_run)

    outcome = research(workstream, "openai", max_calls=6)

    assert outcome.calls_made == 3
    assert outcome.stop_reason == "candidate_survived_attack"
    assert outcome.final_status == "completed"
    assert len(provider.calls) == 3
    assert all(
        call["max_output_tokens"] == RESEARCH_MAX_OUTPUT_TOKENS
        for call in provider.calls
    )
    assert all(call["response_model"] is ResearchStepReport for call in provider.calls)
    assert all(
        "UNRELATED PROJECT HISTORY SENTINEL" not in call["prompt"]
        for call in provider.calls
    )

    with connect() as con:
        iterations = con.execute(
            "SELECT * FROM research_iterations WHERE workstream_id=? ORDER BY iteration_number",
            (workstream,),
        ).fetchall()
        generated = con.execute(
            """
            SELECT e.* FROM workstream_entities we
            JOIN entities e ON e.id=we.entity_id
            WHERE we.workstream_id=? AND we.role='created' ORDER BY e.id
            """,
            (workstream,),
        ).fetchall()
        calls = con.execute(
            "SELECT * FROM api_calls WHERE workstream_id=? ORDER BY id", (workstream,)
        ).fetchall()
        attempts = con.execute(
            "SELECT * FROM relations WHERE relation_type='ATTEMPTS'"
        ).fetchall()
        review = con.execute(
            "SELECT * FROM reviews WHERE workstream_id=?", (workstream,)
        ).fetchone()
        status = con.execute(
            "SELECT status FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0]

    assert [row["operation"] for row in iterations] == [
        "develop",
        "synthesize",
        "attack",
    ]
    assert all(row["rationale"].strip() for row in iterations)
    assert all(row["status"] == "completed" for row in iterations)
    assert iterations[-1]["stop_reason"] == "candidate_survived_attack"
    assert [row["purpose"] for row in calls] == [
        "research:develop",
        "research:synthesize",
        "research:attack",
    ]
    assert all(row["status"] == "completed" for row in calls)
    assert all(row["trust_state"] == "quarantined" for row in generated)
    assert {row["entity_type"] for row in generated} >= {
        "Lemma",
        "Technique",
        "OpenQuestion",
        "FailedApproach",
        "ProofAttempt",
    }
    assert len(attempts) == 1
    assert attempts[0]["trust_state"] == "quarantined"
    assert review["result"] == "no_flaw_found"
    assert status == "completed"


def test_prove_is_selected_only_for_a_precise_candidate(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream("Conjecture", "Precise quorum claim")
    with connect() as con:
        set_attribute(con, target, "precise_candidate", "true")
        obligation = add_entity(con, "OpenQuestion", "Establish the quorum inequality")
        set_attribute(con, obligation, "is_proof_obligation", "true")
        link_workstream_entity(con, workstream, obligation, "created")

    def responder(decision, _):
        assert decision["operation"] == "prove"
        assert decision["target_entity_id"] == target
        return step_report(
            decision,
            [
                artifact(
                    "proof_attempt",
                    "For quorums Q1 and Q2, inclusion-exclusion gives the required overlap.",
                    "quorum_overlap_proof_attempt",
                    [target, obligation],
                    epistemic_status="inference",
                )
            ],
            addressed=[obligation],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=1)

    assert outcome.stop_reason == "max_calls_exhausted"
    assert len(provider.calls) == 1
    with connect() as con:
        iteration = con.execute(
            "SELECT operation,target_entity_id FROM research_iterations"
        ).fetchone()
    assert (iteration["operation"], iteration["target_entity_id"]) == ("prove", target)


def test_proving_candidate_creates_attempt_that_is_attacked_not_proved(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    lemma = add_linked_research_entity(
        workstream,
        "Lemma",
        "Candidate lemma A",
    )
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Open obligation for candidate A",
        proof_obligation=True,
    )

    def responder(decision, call_number):
        if call_number == 1:
            assert decision["operation"] == "prove"
            assert decision["target_entity_id"] == lemma
            return step_report(
                decision,
                [
                    artifact(
                        "proof_attempt",
                        "Candidate A reduces the goal to one explicit missing implication.",
                        "candidate_a_reduction_attempt",
                        [lemma, obligation],
                        epistemic_status="inference",
                    ),
                    artifact(
                        "proof_obligation",
                        "Prove the remaining implication exposed by candidate A.",
                        "candidate_a_remaining_implication",
                        [lemma],
                        epistemic_status="unresolved",
                    ),
                ],
            )
        assert call_number == 2
        assert decision["operation"] == "attack"
        return step_report(decision, [], attack_outcome="inconclusive")

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=2)

    assert outcome.calls_made == 2
    with connect() as con:
        iterations = con.execute(
            "SELECT operation,target_entity_id FROM research_iterations ORDER BY iteration_number"
        ).fetchall()
        proof_attempt_id = con.execute(
            "SELECT id FROM entities WHERE entity_type='ProofAttempt' ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    assert [(row["operation"], row["target_entity_id"]) for row in iterations] == [
        ("prove", lemma),
        ("attack", proof_attempt_id),
    ]
    assert proof_attempt_id != target


def test_synthesize_targets_blocked_obligation_and_consumes_two_artifacts(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream("Conjecture", "Concrete candidate")
    with connect() as con:
        lemma = add_entity(con, "Lemma", "First supporting lemma")
        technique = add_entity(con, "Technique", "Second supporting construction")
        blocker = add_entity(con, "Obstruction", "Missing composition argument")
        link_workstream_entity(con, workstream, lemma, "evidence")
        link_workstream_entity(con, workstream, technique, "evidence")
        link_workstream_entity(con, workstream, blocker, "blocked_by")

    def responder(decision, _):
        assert decision["operation"] == "synthesize"
        assert decision["target_entity_id"] == blocker
        assert {lemma, technique} <= set(decision["required_consumed_entity_ids"])
        related = [blocker, *decision["required_consumed_entity_ids"]]
        return step_report(
            decision,
            [
                artifact(
                    "failed_approach",
                    "The two ingredients use incompatible round boundaries.",
                    "composition_round_boundary_failure",
                    related,
                    branch_status="failed",
                    epistemic_status="inference",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=1)

    assert outcome.calls_made == 1
    assert len(provider.calls) == 1
    with connect() as con:
        iteration = con.execute(
            """
            SELECT operation,target_entity_id,consumed_entity_ids_json
            FROM research_iterations
            """
        ).fetchone()
    assert (iteration["operation"], iteration["target_entity_id"]) == (
        "synthesize",
        blocker,
    )
    assert set(json.loads(iteration["consumed_entity_ids_json"])) >= {
        lemma,
        technique,
    }


@pytest.mark.parametrize(
    ("entity_type", "precise", "expected_operation"),
    [
        ("ResearchIdea", False, "develop"),
        ("Theorem", False, "attack"),
    ],
)
def test_attack_is_reserved_for_concrete_candidates(
    monkeypatch, tmp_path, entity_type, precise, expected_operation
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream(entity_type, "Candidate statement")
    if precise:
        with connect() as con:
            set_attribute(con, target, "precise_candidate", "true")

    def responder(decision, _):
        assert decision["operation"] == expected_operation
        if expected_operation == "attack":
            return step_report(decision, [], attack_outcome="inconclusive")
        return step_report(
            decision,
            [
                artifact(
                    "finding",
                    "The model must first specify a concrete invariant.",
                    "missing_concrete_invariant",
                    [target],
                    epistemic_status="unresolved",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    research(workstream, "openai", max_calls=1)
    assert len(provider.calls) == 1


def test_all_terminal_branches_stop_before_a_model_call(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    with connect() as con:
        blocker = add_entity(con, "Obstruction", "Every branch hits the lower bound")
        set_attribute(con, blocker, "research_branch_status", "blocked")
        link_workstream_entity(con, workstream, blocker, "created")
    provider = DynamicProvider(lambda *_: pytest.fail("provider must not be called"))
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=5)

    assert outcome.calls_made == 0
    assert outcome.stop_reason == "all_branches_blocked_or_refuted"
    assert outcome.final_status == "blocked"
    assert provider.calls == []


def test_duplicate_outputs_are_rejected_and_two_stagnant_iterations_stop(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    with connect() as con:
        existing = add_entity(
            con,
            "Finding",
            "Existing count",
            body="The quorum overlap has size at least n minus two t.",
        )
        set_attribute(con, existing, "research_material_key", "quorum_overlap_count")
        link_workstream_entity(con, workstream, existing, "created")

    def responder(decision, _):
        assert decision["operation"] == "develop"
        return step_report(
            decision,
            [
                artifact(
                    "parameter_analysis",
                    "The quorum overlap has size at least n minus two t.",
                    "quorum_overlap_count",
                    [target],
                    epistemic_status="inference",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=6)

    assert outcome.calls_made == 2
    assert outcome.stop_reason == "stagnation"
    assert outcome.artifact_ids == ()
    with connect() as con:
        iterations = con.execute(
            "SELECT material_progress,duplicate_count FROM research_iterations ORDER BY id"
        ).fetchall()
        count = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    assert [(row[0], row[1]) for row in iterations] == [(0, 1), (0, 1)]
    assert count == 2


def test_max_calls_is_a_hard_bound_and_new_material_is_persisted(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()

    def responder(decision, call_number):
        return step_report(
            decision,
            [
                artifact(
                    "finding",
                    f"Distinct technical consequence number {call_number}.",
                    f"distinct_consequence_{call_number}",
                    [target],
                    epistemic_status="inference",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=2)

    assert outcome.calls_made == 2
    assert len(provider.calls) == 2
    assert len(outcome.artifact_ids) == 2
    assert outcome.stop_reason == "max_calls_exhausted"


def test_max_calls_summary_reports_remaining_open_obligations(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Unresolved proof obligation",
        proof_obligation=True,
    )

    def responder(decision, _):
        assert decision["operation"] == "develop"
        return step_report(
            decision,
            [
                artifact(
                    "finding",
                    "A useful bound that does not yet close the obligation.",
                    "useful_nonclosing_bound",
                    [target],
                    epistemic_status="inference",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=1)

    assert outcome.stop_reason == "max_calls_exhausted"
    assert outcome.final_status == "completed"
    with connect() as con:
        summary = con.execute(
            "SELECT summary FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0]
    assert "max_calls_exhausted" in summary
    assert "1 proof obligation remains open" in summary


def test_human_judgment_request_stops_controller_as_blocked(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()

    def responder(decision, _):
        return step_report(
            decision,
            [
                artifact(
                    "open_question",
                    "Choose whether adaptive corruptions belong in the intended model.",
                    "adaptive_corruption_model_choice",
                    [target],
                    epistemic_status="unresolved",
                )
            ],
            human=True,
            human_reason="The graph does not specify the intended adversary model.",
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=5)

    assert outcome.calls_made == 1
    assert outcome.stop_reason == "human_judgment_required"
    assert outcome.final_status == "blocked"


def test_budget_refusal_makes_no_iteration_or_state_change(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path, budget=0.0001)
    workstream, _ = make_research_workstream()
    monkeypatch.setattr(
        "theory.research.get_provider",
        lambda _: (_ for _ in ()).throw(AssertionError("provider must not be created")),
    )

    with pytest.raises(BudgetExceededError, match="cannot fit"):
        research(workstream, "openai", max_calls=3)

    with connect() as con:
        status = con.execute(
            "SELECT status FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0]
        iterations = con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0]
        calls = con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
    assert status == "active"
    assert iterations == 0
    assert calls == 0


def test_provider_and_validation_failures_are_recorded_consistently(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    provider = RaisingProvider()
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    with pytest.raises(TheoryError, match="mock controller timeout"):
        research(workstream, "anthropic", max_calls=2)

    with connect() as con:
        iteration = con.execute("SELECT * FROM research_iterations").fetchone()
        call = con.execute("SELECT * FROM api_calls").fetchone()
        status = con.execute(
            "SELECT status FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0]
    assert provider.calls == 1
    assert iteration["status"] == "error"
    assert "mock controller timeout" in iteration["error_message"]
    assert call["status"] == "failed"
    assert status == "error"

    with connect() as con:
        con.execute(
            "UPDATE workstreams SET status='active',summary='' WHERE id=?", (workstream,)
        )

    invalid_provider = DynamicProvider(lambda *_: {"unexpected": True})
    monkeypatch.setattr("theory.research.get_provider", lambda _: invalid_provider)
    with pytest.raises(ModelOutputError, match="did not match"):
        research(workstream, "openai", max_calls=1)

    with connect() as con:
        latest_iteration = con.execute(
            "SELECT * FROM research_iterations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        latest_call = con.execute(
            "SELECT * FROM api_calls ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert latest_iteration["status"] == "error"
    assert latest_call["status"] == "completed"


def test_incomplete_openai_response_is_not_parsed_and_retains_usage(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()

    class IncompleteProvider:
        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return ModelResult(
                text='{"operation": "develop", "summary": "cut off',
                input_tokens=1_234,
                output_tokens=31_999,
                cost_usd=0.644916,
                response_status="incomplete",
                incomplete_reason="max_output_tokens",
            )

    provider = IncompleteProvider()
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    def parsing_must_not_run(*args, **kwargs):
        raise AssertionError("incomplete output must not reach JSON parsing")

    monkeypatch.setattr("theory.research.parse_json_model", parsing_must_not_run)

    with pytest.raises(
        ModelOutputError,
        match="OpenAI response was incomplete: max_output_tokens exhausted",
    ):
        research(workstream, "openai", max_calls=1)

    assert len(provider.calls) == 1
    assert provider.calls[0]["max_output_tokens"] == 32_000
    assert provider.calls[0]["response_model"] is ResearchStepReport
    with connect() as con:
        call = con.execute("SELECT * FROM api_calls").fetchone()
        iteration = con.execute("SELECT * FROM research_iterations").fetchone()
        artifact_count = con.execute(
            "SELECT COUNT(*) FROM workstream_entities WHERE workstream_id=? AND role='created'",
            (workstream,),
        ).fetchone()[0]
    assert call["status"] == "failed"
    assert call["input_tokens"] == 1_234
    assert call["output_tokens"] == 31_999
    assert call["cost_usd"] == pytest.approx(0.644916)
    assert call["response_text"].endswith('"cut off')
    assert "max_output_tokens exhausted" in call["error_message"]
    assert monthly_spend() == pytest.approx(0.644916)
    assert iteration["status"] == "error"
    assert artifact_count == 0


def test_anthropic_research_keeps_json_parsing_path(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()

    def responder(decision, _):
        return step_report(
            decision,
            [
                artifact(
                    "parameter_analysis",
                    "The parameter boundary exposes a separate blocker.",
                    "anthropic_parameter_analysis",
                    [target],
                    branch_status="unresolved",
                ),
                artifact(
                    "obstruction",
                    "The exposed boundary blocks this construction.",
                    "anthropic_blocked_obstruction",
                    [target],
                    branch_status="blocked",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "anthropic", max_calls=1)

    assert outcome.calls_made == 1
    assert provider.calls[0]["response_model"] is None
    with connect() as con:
        branch_statuses = con.execute(
            """
            SELECT a.value FROM workstream_entities we
            JOIN entity_attributes a ON a.entity_id=we.entity_id
            WHERE we.workstream_id=? AND we.role='created'
              AND a.key='research_branch_status'
            ORDER BY a.entity_id
            """,
            (workstream,),
        ).fetchall()
    assert [row["value"] for row in branch_statuses] == ["unresolved", "blocked"]


def test_reactivated_research_retries_operation_after_errored_iteration(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream("Theorem", "Concrete theorem")

    def invalid_responder(decision, _):
        assert decision["operation"] == "attack"
        response = step_report(decision, [], attack_outcome="inconclusive")
        response["operation"] = "develop"
        return response

    first_provider = DynamicProvider(invalid_responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: first_provider)
    with pytest.raises(ModelOutputError, match="expected 'attack'"):
        research(workstream, "openai", max_calls=1)

    with connect() as con:
        con.execute(
            "UPDATE workstreams SET status='active',summary='' WHERE id=?",
            (workstream,),
        )

    def valid_responder(decision, _):
        assert decision["operation"] == "attack"
        return step_report(decision, [], attack_outcome="inconclusive")

    second_provider = DynamicProvider(valid_responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: second_provider)

    outcome = research(workstream, "openai", max_calls=1)

    assert outcome.calls_made == 1
    assert len(second_provider.calls) == 1
    with connect() as con:
        iterations = con.execute(
            "SELECT operation,status FROM research_iterations ORDER BY iteration_number"
        ).fetchall()
    assert [(row["operation"], row["status"]) for row in iterations] == [
        ("attack", "error"),
        ("attack", "completed"),
    ]


def test_research_cli_and_workstream_show_make_no_extra_call(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()

    def responder(decision, call_number):
        return step_report(
            decision,
            [
                artifact(
                    "finding",
                    "A new invariant candidate relates delivery and certificate formation.",
                    f"delivery_certificate_invariant_{call_number}",
                    [target],
                    epistemic_status="inference",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["research", str(workstream), "--provider", "openai", "--max-calls", "1"],
    )

    assert result.exit_code == 0, result.output
    assert "Research controller stopped" in result.output
    assert "max_calls_exhausted" in result.output
    assert "iteration 1: develop" in result.output
    assert "why:" in result.output
    assert "research:develop" in result.output
    assert len(provider.calls) == 1

    shown = runner.invoke(app, ["workstream", "show", str(workstream)])
    assert shown.exit_code == 0, shown.output
    assert "iteration 1: develop" in shown.output
    assert "max_calls_exhausted" in shown.output
    assert len(provider.calls) == 1
