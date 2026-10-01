import json

import pytest
from typer.testing import CliRunner

from theory.cli import app
from theory.config import Config
from theory.db import connect, initialize, monthly_spend
from theory.errors import BudgetExceededError, ModelOutputError, TheoryError
from theory.graph import (
    add_entity,
    add_relation,
    create_workstream,
    link_workstream_entity,
    set_attribute,
)
from theory.models import ModelResult
from theory.prompts import render_prompt
from theory.research import (
    OperationChoice,
    RESEARCH_MAX_OUTPUT_TOKENS,
    ResearchArtifact,
    ResearchStepReport,
    ResearchAttackResponse,
    _open_obligation_ids,
    _relevant_synthesis_inputs,
    _research_prompt,
    _select_focus_obligation,
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
        kwargs["prompt"] = render_prompt(kwargs["prompt"])
        self.calls.append(kwargs)
        response = self.responder(decision_from_prompt(kwargs["prompt"]), len(self.calls))
        if kwargs["response_model"].__name__ == "ResearchAttackResponse":
            response = {"report": response}
        return ModelResult(
            text=json.dumps(response),
            input_tokens=500,
            uncached_input_tokens=500,
            output_tokens=250,
            cost_usd=0.007,
            output_cost_usd=0.007,
        )


class RaisingProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        kwargs["prompt"] = render_prompt(kwargs["prompt"])
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
    branch_status=None,
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
        if branch_status is not None:
            set_attribute(con, entity_id, "research_branch_status", branch_status)
    return entity_id


def mark_candidate_attempt(candidate_id, obligation_id, *, addressed=True):
    with connect() as con:
        add_relation(con, candidate_id, "ATTEMPTS", obligation_id)
        if addressed:
            set_attribute(
                con,
                candidate_id,
                "addresses_obligation_ids",
                json.dumps([obligation_id]),
            )


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

    assert "ATTACK OUTCOME PRECEDENCE" in prompt
    assert '1. CONCRETE DEFECT FOUND: use "critical_issue".' in prompt
    assert '2. NO CONCRETE DEFECT, BUT MATERIAL UNCERTAINTY REMAINS: use "inconclusive".' in prompt
    assert '3. NEITHER A CONCRETE DEFECT NOR MATERIAL UNCERTAINTY: use "no_critical_issue".' in prompt
    assert "could_not_determine non-empty" in prompt
    assert '"attack_outcome": "no_critical_issue"' in prompt
    assert '"artifacts": []' in prompt
    assert "critical_issue|no_critical_issue|inconclusive" not in prompt
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
    if operation == "attack":
        assert '"artifacts": []' in prompt
    else:
        assert '"related_entity_ids": [1]' in prompt
    assert "minimum required related_entity_ids" not in prompt


def test_focused_prove_prompt_requires_target_and_obligation_references():
    choice = OperationChoice(
        "prove",
        27,
        "Prove the obligation-specific candidate",
        open_obligation_ids=(9,),
        focus_obligation_id=9,
    )

    prompt = _research_prompt(
        MinimalResearchContext(9, 27),
        {"id": 1},
        choice,
    )
    decision = decision_from_prompt(prompt)

    assert decision["target_entity_id"] == 27
    assert decision["focus_obligation_id"] == 9
    assert decision["required_artifact_related_entity_ids"] == [27, 9]
    assert "every lemma or proof_attempt MUST include both target entity #27" in prompt
    assert "focus obligation #9" in prompt
    assert "does not by itself justify adding the obligation" in prompt


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
    ("attack_outcome", "has_critical_artifact", "has_uncertainty"),
    [
        (outcome, critical, uncertainty)
        for outcome in ("critical_issue", "inconclusive", "no_critical_issue")
        for critical in (False, True)
        for uncertainty in (False, True)
    ],
)
def test_attack_output_contract_all_artifact_uncertainty_combinations(
    attack_outcome, has_critical_artifact, has_uncertainty
):
    report = ResearchStepReport(
        operation="attack",
        target_entity_id=1,
        summary="Bounded adversarial pass.",
        artifacts=(
            [
                artifact(
                    "obstruction",
                    "A concrete invalid step blocks this candidate.",
                    "attack_contract_concrete_obstruction",
                    [1],
                    branch_status="blocked",
                )
            ]
            if has_critical_artifact
            else []
        ),
        consumed_entity_ids=[],
        addressed_obligation_ids=[],
        attack_outcome=attack_outcome,
        could_not_determine=(
            ["The boundary case remains materially unresolved."]
            if has_uncertainty
            else []
        ),
        human_judgment_required=False,
        human_judgment_reason=None,
    )
    valid = (
        (attack_outcome == "critical_issue" and has_critical_artifact)
        or (
            attack_outcome == "inconclusive"
            and not has_critical_artifact
            and has_uncertainty
        )
        or (
            attack_outcome == "no_critical_issue"
            and not has_critical_artifact
            and not has_uncertainty
        )
    )

    if valid:
        _validate_step_report(
            report,
            MinimalResearchContext(),
            OperationChoice("attack", 1, "Regression test"),
        )
        return

    if attack_outcome == "critical_issue":
        error = "critical_issue requires a concrete"
    elif attack_outcome == "inconclusive":
        error = (
            "Critical attack artifacts require"
            if has_critical_artifact
            else "inconclusive requires non-empty could_not_determine"
        )
    else:
        error = "no_critical_issue cannot accompany"
    with pytest.raises(ModelOutputError, match=error):
        _validate_step_report(
            report,
            MinimalResearchContext(),
            OperationChoice("attack", 1, "Regression test"),
        )


@pytest.mark.parametrize(
    ("artifact_type", "branch_status"),
    [
        ("counterexample", None),
        ("obstruction", "blocked"),
        ("failed_approach", "failed"),
    ],
)
def test_critical_issue_accepts_each_concrete_defect_type(
    artifact_type, branch_status
):
    report = ResearchStepReport(
        operation="attack",
        target_entity_id=1,
        summary="A concrete defect was established.",
        artifacts=[
            artifact(
                artifact_type,
                f"Concrete {artifact_type} found by the bounded attack.",
                f"attack_contract_{artifact_type}",
                [1],
                branch_status=branch_status,
            )
        ],
        consumed_entity_ids=[],
        addressed_obligation_ids=[],
        attack_outcome="critical_issue",
        could_not_determine=[],
        human_judgment_required=False,
        human_judgment_reason=None,
    )

    _validate_step_report(
        report,
        MinimalResearchContext(),
        OperationChoice("attack", 1, "Regression test"),
    )


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


def test_blocked_obstruction_is_not_an_open_proof_obligation(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obstruction = add_linked_research_entity(
        workstream,
        "Obstruction",
        "This branch cannot satisfy the overlap bound",
        role="blocked_by",
        branch_status="blocked",
    )

    context = for_workstream(workstream)

    assert obstruction not in _open_obligation_ids(context, workstream, target)


def test_explicit_open_question_is_an_open_proof_obligation(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Prove the remaining overlap inequality",
    )
    with connect() as con:
        set_attribute(con, obligation, "research_artifact_type", "proof_obligation")

    context = for_workstream(workstream)

    assert _open_obligation_ids(context, workstream, target) == (obligation,)


def test_lower_id_obligation_cannot_starve_later_open_obligation(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    lower_id = add_linked_research_entity(
        workstream, "OpenQuestion", "First obligation", proof_obligation=True
    )
    later_id = add_linked_research_entity(
        workstream, "OpenQuestion", "Later obligation", proof_obligation=True
    )
    history = [
        completed_history("develop", lower_id),
        completed_history("develop", lower_id),
    ]

    choice = current_choice(workstream, target, history)

    assert choice.focus_obligation_id == later_id
    assert choice.target_entity_id == later_id


def test_never_focused_obligation_beats_repeatedly_focused_obligation(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    never_focused = add_linked_research_entity(
        workstream, "OpenQuestion", "Neglected obligation", proof_obligation=True
    )
    repeatedly_focused = add_linked_research_entity(
        workstream, "OpenQuestion", "Repeated obligation", proof_obligation=True
    )
    history = [
        completed_history("develop", repeatedly_focused),
        completed_history("synthesize", repeatedly_focused),
    ]

    choice = current_choice(workstream, target, history)

    assert choice.focus_obligation_id == never_focused


def test_parent_obligation_is_deferred_while_explicit_child_is_open(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    parent = add_linked_research_entity(
        workstream, "OpenQuestion", "Parent obligation", proof_obligation=True
    )
    child = add_linked_research_entity(
        workstream, "OpenQuestion", "Refined child obligation", proof_obligation=True
    )
    with connect() as con:
        set_attribute(con, child, "research_focus_obligation_id", str(parent))
        add_relation(con, parent, "DEPENDS_ON", child)

    choice = current_choice(workstream, target)

    assert choice.focus_obligation_id == child
    assert choice.focus_obligation_id != parent


def test_parent_obligation_becomes_eligible_after_related_child_is_inactive(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    parent = add_linked_research_entity(
        workstream, "OpenQuestion", "Parent obligation", proof_obligation=True
    )
    child = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Child linked through related IDs",
        related_entity_ids=(parent,),
        proof_obligation=True,
    )
    with connect() as con:
        set_attribute(con, child, "research_obligation_state", "blocked")

    choice = current_choice(workstream, target)

    assert choice.open_obligation_ids == (parent,)
    assert choice.focus_obligation_id == parent


def test_repeated_selection_eventually_focuses_every_open_leaf(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligations = tuple(
        add_linked_research_entity(
            workstream,
            "OpenQuestion",
            f"Leaf obligation {index}",
            proof_obligation=True,
        )
        for index in range(3)
    )
    context = for_workstream(workstream)
    history: list[dict] = []
    selected: list[int] = []
    for _ in range(6):
        focus_id = _select_focus_obligation(context, tuple(history), obligations)
        selected.append(focus_id)
        history.append(completed_history("develop", focus_id))

    assert selected == [*obligations, *obligations]


def test_focus_obligation_selection_is_deterministic_and_excludes_inactive(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    selectable = add_linked_research_entity(
        workstream, "OpenQuestion", "Selectable obligation", proof_obligation=True
    )
    blocked = add_linked_research_entity(
        workstream, "OpenQuestion", "Blocked obligation", proof_obligation=True
    )
    resolved = add_linked_research_entity(
        workstream, "OpenQuestion", "Resolved candidate", proof_obligation=True
    )
    with connect() as con:
        set_attribute(con, blocked, "research_obligation_state", "blocked")
        set_attribute(con, resolved, "research_obligation_state", "resolved_candidate")
    context = for_workstream(workstream)
    open_ids = _open_obligation_ids(context, workstream, target)

    first = _select_focus_obligation(context, (), open_ids)
    second = _select_focus_obligation(context, (), open_ids)

    assert open_ids == (selectable,)
    assert first == second == selectable
    assert blocked not in open_ids
    assert resolved not in open_ids


def test_legacy_candidate_survived_attack_state_remains_actionable(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "A candidate survived, but the obligation is not controller-complete",
        proof_obligation=True,
    )
    with connect() as con:
        set_attribute(
            con,
            obligation,
            "research_obligation_state",
            "candidate_survived_attack",
        )

    context = for_workstream(workstream)

    assert _open_obligation_ids(context, workstream, target) == (obligation,)


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
    with connect() as con:
        add_relation(con, newer_attempt, "ATTEMPTS", obligation)

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


def test_unrelated_lemma_is_not_selected_for_open_obligation(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Resolve the conflict-selection step",
        proof_obligation=True,
    )
    unrelated_lemma = add_linked_research_entity(
        workstream,
        "Lemma",
        "A relay dissemination bound",
    )

    choice = current_choice(workstream, target)

    assert choice.operation == "develop"
    assert choice.target_entity_id == obligation
    assert choice.target_entity_id != unrelated_lemma


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
        related_entity_ids=(obligation,),
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
    add_linked_research_entity(
        workstream,
        "Finding",
        "First relevant bound",
        related_entity_ids=(obligation,),
    )
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

    unrelated_input = add_linked_research_entity(
        workstream, "Finding", "A new but unrelated dissemination fact"
    )
    unrelated_choice = current_choice(workstream, target, same_inputs_history)
    assert unrelated_choice.operation == "develop"
    assert unrelated_input not in unrelated_choice.consumed_entity_ids

    new_input = add_linked_research_entity(
        workstream,
        "Finding",
        "A genuinely new relevant bound",
        related_entity_ids=(obligation,),
    )
    changed_choice = current_choice(workstream, target, same_inputs_history)
    assert changed_choice.operation == "synthesize"
    assert new_input in changed_choice.consumed_entity_ids
    assert set(changed_choice.consumed_entity_ids) != set(
        first_choice.consumed_entity_ids
    )


def test_attempts_relation_does_not_close_obligation_or_misroute_attack(
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
        add_relation(con, unrelated_attempt, "ATTEMPTS", other_obligation)
        set_attribute(
            con,
            unrelated_attempt,
            "addresses_obligation_ids",
            json.dumps([other_obligation]),
        )

    choice = current_choice(workstream, target)

    assert choice.open_obligation_ids == (current_obligation, other_obligation)
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


def test_critical_attack_keeps_obligation_actionable(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Close the conflict-selection argument",
        proof_obligation=True,
    )
    attempt = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "Candidate conflict-selection argument",
        related_entity_ids=(obligation,),
    )

    def responder(decision, _):
        assert decision["operation"] == "attack"
        assert decision["target_entity_id"] == attempt
        assert decision["focus_obligation_id"] == obligation
        return step_report(
            decision,
            [
                artifact(
                    "obstruction",
                    "The candidate assumes an unavailable common conflict view.",
                    "missing_common_conflict_view",
                    [attempt],
                    branch_status="blocked",
                    epistemic_status="inference",
                )
            ],
            attack_outcome="critical_issue",
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=1)
    context = for_workstream(workstream)

    assert outcome.stop_reason == "max_calls_exhausted"
    assert _open_obligation_ids(context, workstream, target) == (obligation,)
    assert context.attributes[obligation]["research_obligation_state"] == "challenged"
    assert context.attributes[attempt]["research_attack_state"] == "challenged"


def test_multiple_candidate_attack_histories_remain_separate(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Prove obligation fourteen",
        proof_obligation=True,
    )
    candidate_c = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "Candidate C",
        related_entity_ids=(obligation,),
    )
    candidate_b = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "Candidate B",
        related_entity_ids=(obligation,),
    )
    candidate_a = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "Candidate A",
        related_entity_ids=(obligation,),
    )
    for candidate_id in (candidate_a, candidate_b, candidate_c):
        mark_candidate_attempt(candidate_id, obligation)

    def responder(decision, _):
        candidate_id = decision["target_entity_id"]
        assert decision["operation"] == "attack"
        assert decision["focus_obligation_id"] == obligation
        if candidate_id in {candidate_a, candidate_b}:
            return step_report(
                decision,
                [
                    artifact(
                        "obstruction",
                        f"Candidate {candidate_id} has a critical gap.",
                        f"candidate_{candidate_id}_critical_gap",
                        [candidate_id],
                        branch_status="blocked",
                        epistemic_status="inference",
                    )
                ],
                attack_outcome="critical_issue",
            )
        assert candidate_id == candidate_c
        return step_report(decision, [], attack_outcome="no_critical_issue")

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=4)
    context = for_workstream(workstream)

    assert outcome.calls_made == 3
    assert outcome.stop_reason == "candidate_survived_attack"
    assert context.attributes[candidate_a]["research_attack_state"] == "challenged"
    assert context.attributes[candidate_b]["research_attack_state"] == "challenged"
    assert context.attributes[candidate_c]["research_attack_state"] == "survived_attack"
    assert context.attributes[obligation]["research_obligation_state"] == "resolved_candidate"
    assert context.attributes[obligation]["research_surviving_candidate_id"] == str(
        candidate_c
    )
    assert "verified" not in {
        context.attributes[candidate_a]["research_attack_state"],
        context.attributes[candidate_b]["research_attack_state"],
        context.attributes[candidate_c]["research_attack_state"],
        context.attributes[obligation]["research_obligation_state"],
    }
    with connect() as con:
        iterations = con.execute(
            """
            SELECT target_entity_id,attack_outcome FROM research_iterations
            ORDER BY iteration_number
            """
        ).fetchall()
        reviews = con.execute(
            "SELECT result,issues FROM reviews ORDER BY id"
        ).fetchall()
    assert [(row["target_entity_id"], row["attack_outcome"]) for row in iterations] == [
        (candidate_a, "critical_issue"),
        (candidate_b, "critical_issue"),
        (candidate_c, "no_critical_issue"),
    ]
    assert [row["result"] for row in reviews] == [
        "issue_found",
        "issue_found",
        "no_flaw_found",
    ]
    assert f"entity #{candidate_a}" in reviews[0]["issues"]
    assert f"entity #{candidate_b}" in reviews[1]["issues"]
    assert f"entity #{candidate_c}" in reviews[2]["issues"]


def test_surviving_structurally_incomplete_candidate_does_not_close_obligation(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Complete the remaining proof obligation",
        proof_obligation=True,
    )
    candidate = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "Candidate without the strong addressed marker",
        related_entity_ids=(obligation,),
    )
    mark_candidate_attempt(candidate, obligation, addressed=False)

    provider = DynamicProvider(
        lambda decision, _: step_report(
            decision, [], attack_outcome="no_critical_issue"
        )
    )
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=1)
    context = for_workstream(workstream)

    assert outcome.stop_reason == "max_calls_exhausted"
    assert context.attributes[candidate]["research_attack_state"] == "survived_attack"
    assert context.attributes[obligation]["research_obligation_state"] == "open"
    assert "research_surviving_candidate_id" not in context.attributes[obligation]
    assert _open_obligation_ids(context, workstream, target) == (obligation,)


def test_global_survived_attack_stop_requires_every_obligation_complete(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    first_obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "First proof obligation",
        proof_obligation=True,
    )
    second_obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Second proof obligation",
        proof_obligation=True,
    )
    first_candidate = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "First complete candidate",
        related_entity_ids=(first_obligation,),
    )
    second_candidate = add_linked_research_entity(
        workstream,
        "ProofAttempt",
        "Second complete candidate",
        related_entity_ids=(second_obligation,),
    )
    mark_candidate_attempt(first_candidate, first_obligation)
    mark_candidate_attempt(second_candidate, second_obligation)

    provider = DynamicProvider(
        lambda decision, _: step_report(
            decision, [], attack_outcome="no_critical_issue"
        )
    )
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=3)
    context = for_workstream(workstream)

    assert outcome.calls_made == 2
    assert outcome.stop_reason == "candidate_survived_attack"
    assert [
        decision_from_prompt(call["prompt"])["focus_obligation_id"]
        for call in provider.calls
    ] == [first_obligation, second_obligation]
    for obligation_id, candidate_id in (
        (first_obligation, first_candidate),
        (second_obligation, second_candidate),
    ):
        assert (
            context.attributes[obligation_id]["research_obligation_state"]
            == "resolved_candidate"
        )
        assert context.attributes[obligation_id][
            "research_surviving_candidate_id"
        ] == str(candidate_id)


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


def test_focused_prove_rejects_proof_artifact_missing_obligation_reference():
    report = ResearchStepReport(
        operation="prove",
        target_entity_id=27,
        summary="Candidate proof omitted its focus provenance.",
        artifacts=[
            artifact(
                "proof_attempt",
                "A candidate argument for the target lemma.",
                "candidate_missing_focus_reference",
                [27],
            )
        ],
        consumed_entity_ids=[],
        addressed_obligation_ids=[],
        attack_outcome="not_applicable",
        could_not_determine=[],
        human_judgment_required=False,
        human_judgment_reason=None,
    )

    with pytest.raises(ModelOutputError, match="did not reference obligation #9"):
        _validate_step_report(
            report,
            MinimalResearchContext(9, 27),
            OperationChoice(
                "prove",
                27,
                "Regression test",
                open_obligation_ids=(9,),
                focus_obligation_id=9,
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
            assert decision["operation"] == "develop"
            assert decision["focus_obligation_id"] == selected
            return step_report(
                decision,
                [
                    artifact(
                        "lemma",
                        "The obligation-specific overlap lemma supplies an honest signer.",
                        "focused_honest_overlap_lemma",
                        [selected],
                        epistemic_status="inference",
                    ),
                    artifact(
                        "protocol_component",
                        "Use an obligation-specific threshold certificate construction.",
                        "focused_threshold_certificate_component",
                        [selected],
                        branch_status="promising",
                    ),
                ],
            )
        if call_number == 3:
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
        assert call_number == 4
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

    assert outcome.calls_made == 4
    assert outcome.stop_reason == "candidate_survived_attack"
    assert outcome.final_status == "completed"
    assert len(provider.calls) == 4
    assert all(
        call["max_output_tokens"] == RESEARCH_MAX_OUTPUT_TOKENS
        for call in provider.calls
    )
    assert all(call["response_model"] is (ResearchAttackResponse if decision_from_prompt(call["prompt"])["operation"] == "attack" else ResearchStepReport) for call in provider.calls)
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
        obligation_row = con.execute(
            """
            SELECT e.id,ea.value FROM entity_attributes ea
            JOIN entities e ON e.id=ea.entity_id
            WHERE e.entity_type='OpenQuestion'
              AND ea.key='research_obligation_state'
            """
        ).fetchone()
        proof_attempt_id = con.execute(
            "SELECT id FROM entities WHERE entity_type='ProofAttempt'"
        ).fetchone()[0]
        surviving_candidate_id = con.execute(
            """
            SELECT value FROM entity_attributes
            WHERE entity_id=? AND key='research_surviving_candidate_id'
            """,
            (obligation_row["id"],),
        ).fetchone()[0]
        candidate_attack_state = con.execute(
            """
            SELECT value FROM entity_attributes
            WHERE entity_id=? AND key='research_attack_state'
            """,
            (proof_attempt_id,),
        ).fetchone()[0]
        summary = con.execute(
            "SELECT summary FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0]

    assert [row["operation"] for row in iterations] == [
        "develop",
        "develop",
        "synthesize",
        "attack",
    ]
    assert all(row["rationale"].strip() for row in iterations)
    assert all(row["status"] == "completed" for row in iterations)
    assert iterations[-1]["stop_reason"] == "candidate_survived_attack"
    assert [row["purpose"] for row in calls] == [
        "research:develop",
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
    assert obligation_row["value"] == "resolved_candidate"
    assert surviving_candidate_id == str(proof_attempt_id)
    assert candidate_attack_state == "survived_attack"
    assert "not proof verification" in summary
    assert "verified" not in obligation_row["value"]
    assert "verified" not in candidate_attack_state


def test_prove_is_selected_only_for_a_precise_candidate(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream("Conjecture", "Precise quorum claim")
    with connect() as con:
        set_attribute(con, target, "precise_candidate", "true")
        obligation = add_entity(con, "OpenQuestion", "Establish the quorum inequality")
        set_attribute(con, obligation, "is_proof_obligation", "true")
        set_attribute(con, obligation, "related_entity_ids", json.dumps([target]))
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
        related_entity_ids=(lemma,),
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
                    )
                ],
                addressed=[obligation],
            )
        assert call_number == 2
        assert decision["operation"] == "attack"
        return step_report(
            decision,
            [],
            attack_outcome="inconclusive",
            unresolved=["The remaining implication could not be determined."],
        )

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
    context = for_workstream(workstream)
    assert context.attributes[obligation]["research_obligation_state"] == "challenged"
    assert context.attributes[proof_attempt_id]["research_attack_state"] == "inconclusive"
    assert obligation in _open_obligation_ids(context, workstream, target)


def test_synthesize_targets_explicit_obligation_not_blocked_obstruction(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream("Conjecture", "Concrete candidate")
    with connect() as con:
        lemma = add_entity(con, "Lemma", "First supporting lemma")
        technique = add_entity(con, "Technique", "Second supporting construction")
        blocker = add_entity(con, "Obstruction", "Missing composition argument")
        obligation = add_entity(con, "OpenQuestion", "Resolve the composition gap")
        set_attribute(con, blocker, "research_branch_status", "blocked")
        set_attribute(con, obligation, "is_proof_obligation", "true")
        set_attribute(con, lemma, "related_entity_ids", json.dumps([obligation]))
        set_attribute(con, technique, "related_entity_ids", json.dumps([obligation]))
        link_workstream_entity(con, workstream, lemma, "evidence")
        link_workstream_entity(con, workstream, technique, "evidence")
        link_workstream_entity(con, workstream, blocker, "blocked_by")
        link_workstream_entity(con, workstream, obligation, "created")

    def responder(decision, _):
        assert decision["operation"] == "synthesize"
        assert decision["target_entity_id"] == obligation
        assert {lemma, technique} <= set(decision["required_consumed_entity_ids"])
        related = [obligation, *decision["required_consumed_entity_ids"]]
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
        obligation,
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
            return step_report(
                decision,
                [],
                attack_outcome="inconclusive",
                unresolved=["The boundary behavior remains undetermined."],
            )
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
        route = add_entity(con, "Technique", "Established protocol route")
        set_attribute(con, route, "research_artifact_type", "protocol_component")
        set_attribute(con, route, "research_construction_started_iteration_id", "1")
        set_attribute(con, route, "research_construction_route_ids", json.dumps([route]))
        link_workstream_entity(con, workstream, route, "created")
        blocker = add_entity(con, "Obstruction", "Every branch hits the lower bound")
        set_attribute(con, blocker, "research_branch_status", "blocked")
        link_workstream_entity(con, workstream, blocker, "created")
        add_relation(con, blocker, "BLOCKS", route)
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


def test_reactivated_stagnated_workstream_gets_a_fresh_stagnation_window(
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

    def duplicate_responder(decision, _):
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

    first_provider = DynamicProvider(duplicate_responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: first_provider)

    first_outcome = research(workstream, "openai", max_calls=6)

    assert first_outcome.calls_made == 2
    assert first_outcome.stop_reason == "stagnation"
    with connect() as con:
        historical_rows = [
            tuple(row)
            for row in con.execute(
                "SELECT * FROM research_iterations ORDER BY id"
            ).fetchall()
        ]
        con.execute(
            "UPDATE workstreams SET status='active',summary='' WHERE id=?",
            (workstream,),
        )

    second_provider = DynamicProvider(duplicate_responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: second_provider)

    second_outcome = research(workstream, "openai", max_calls=6)

    assert second_outcome.calls_made == 2
    assert len(second_provider.calls) == 2
    assert second_outcome.stop_reason == "stagnation"
    with connect() as con:
        all_rows = [
            tuple(row)
            for row in con.execute(
                "SELECT * FROM research_iterations ORDER BY id"
            ).fetchall()
        ]
        iteration_states = con.execute(
            """
            SELECT iteration_number,status,material_progress,duplicate_count,stop_reason
            FROM research_iterations ORDER BY iteration_number
            """
        ).fetchall()
        call_count = con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
    assert all_rows[:2] == historical_rows
    assert len(all_rows) == 4
    assert [tuple(row) for row in iteration_states] == [
        (1, "completed", 0, 1, None),
        (2, "completed", 0, 1, "stagnation"),
        (3, "completed", 0, 1, None),
        (4, "completed", 0, 1, "stagnation"),
    ]
    assert call_count == 4


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
    obligation = add_linked_research_entity(
        workstream,
        "OpenQuestion",
        "Unresolved proof obligation",
        proof_obligation=True,
    )
    add_linked_research_entity(
        workstream,
        "Obstruction",
        "A different branch is terminally blocked",
        role="blocked_by",
        branch_status="blocked",
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
                    [obligation],
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
    contract = (
        "Use either fixed corrupted links or fresh corrupted links in each round. "
        "Determine the exact optimal threshold for the chosen model. Model choice is unspecified."
    )
    with connect() as con:
        con.execute("UPDATE entities SET body=? WHERE id=?", (contract, target))

    def responder(decision, _):
        assert decision["operation"] == "develop" and decision["target_entity_id"] == target
        return step_report(
            decision,
            [
                artifact(
                    "open_question",
                    "Choose between the explicitly supplied fixed-link and changing-link models.",
                    "adaptive_corruption_model_choice",
                    [target],
                    epistemic_status="unresolved",
                )
            ],
            human=True,
            human_reason=(
                "The supplied contract explicitly offers fixed or fresh corrupted links and "
                "requires the exact optimal threshold for the chosen model. These are different "
                "optimization problems; a conservative bound cannot settle which optimum is requested."
            ),
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=5)

    prompt = provider.calls[0]["prompt"]
    assert contract in prompt
    assert "no conservative route can proceed without choosing one" in prompt
    assert "Candidate uncertainty, failed derivations" in prompt
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
            kwargs["prompt"] = render_prompt(kwargs["prompt"])
            self.calls.append(kwargs)
            return ModelResult(
                text='{"operation": "develop", "summary": "cut off',
                input_tokens=1_234,
                uncached_input_tokens=1_234,
                output_tokens=31_999,
                cost_usd=0.644916,
                output_cost_usd=0.644916,
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
    assert provider.calls[0]["max_output_tokens"] == RESEARCH_MAX_OUTPUT_TOKENS
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
    assert provider.calls[0]["response_model"] is ResearchStepReport
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
    with pytest.raises(ModelOutputError, match="Input should be 'attack'"):
        research(workstream, "openai", max_calls=1)

    with connect() as con:
        con.execute(
            "UPDATE workstreams SET status='active',summary='' WHERE id=?",
            (workstream,),
        )

    def valid_responder(decision, _):
        assert decision["operation"] == "attack"
        return step_report(
            decision,
            [],
            attack_outcome="inconclusive",
            unresolved=["The retry still cannot determine the boundary case."],
        )

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
    assert "Proof obligations" in result.output
    assert "Execution accounting" in result.output
    assert len(provider.calls) == 1

    shown = runner.invoke(app, ["workstream", "show", str(workstream)])
    assert shown.exit_code == 0, shown.output
    assert "iteration 1: develop" in shown.output
    assert "max_calls_exhausted" in shown.output
    assert len(provider.calls) == 1
