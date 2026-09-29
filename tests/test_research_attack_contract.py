"""The wire contract narrows outcomes without changing scientific validation."""
import json

import pytest
from pydantic import ValidationError

import theory.research as controller
from theory.db import connect
from theory.errors import ModelOutputError
from theory.providers import _response_format
from theory.research import (
    FlatAttackReport, ResearchAttackResponse, ResearchStepReport,
    _execution_response_model, _parse_execution_report, research,
)
from theory.research_context import for_workstream
from test_research import (
    DynamicProvider, add_linked_research_entity, artifact, init_workspace,
    make_research_workstream, step_report,
)


def attack_payload(outcome, critical, uncertainty):
    return step_report(
        {"operation": "attack", "target_entity_id": 1, "required_consumed_entity_ids": []},
        [artifact("counterexample", "Concrete failing schedule.", "failing_schedule", [1])]
        if critical else [],
        attack_outcome=outcome, unresolved=uncertainty,
    )


@pytest.mark.parametrize("outcome", ("critical_issue", "inconclusive", "no_critical_issue"))
@pytest.mark.parametrize("critical", (False, True))
@pytest.mark.parametrize("uncertainty", ([], ["Timing is unresolved."], ["   "]))
def test_attack_variants_match_existing_outcome_semantics(outcome, critical, uncertainty):
    payload = attack_payload(outcome, critical, uncertainty)
    valid = uncertainty != ["   "] and (
        (outcome == "critical_issue" and critical)
        or (outcome == "inconclusive" and not critical and bool(uncertainty))
        or (outcome == "no_critical_issue" and not critical and not uncertainty)
    )
    if valid:
        report = ResearchAttackResponse.model_validate({"report": payload}).report
        assert report.model_dump() == ResearchStepReport.model_validate(payload).model_dump()
    else:
        with pytest.raises(ValidationError):
            ResearchAttackResponse.model_validate({"report": payload})


def test_critical_variant_requires_critical_not_just_nonempty_artifacts():
    payload = attack_payload("critical_issue", False, [])
    payload["artifacts"] = [artifact("finding", "Only an observation.", "observation", [1])]
    with pytest.raises(ValidationError, match="requires a concrete"):
        ResearchAttackResponse.model_validate({"report": payload})


def test_openai_attack_schema_keeps_outcome_variants():
    assert _execution_response_model("attack", "openai") is ResearchAttackResponse
    schema = _response_format("openai", ResearchAttackResponse)["schema"]
    assert schema["type"] == "object"
    assert len(schema["properties"]["report"]["anyOf"]) == 3
    text = json.dumps(schema)
    for outcome in ("critical_issue", "inconclusive", "no_critical_issue"):
        assert outcome in text
    assert '"not_applicable", "critical_issue"' not in text


def test_anthropic_attack_schema_has_flat_artifacts_and_no_outcome_union():
    assert _execution_response_model("attack", "anthropic") is FlatAttackReport
    schema = _response_format("anthropic", FlatAttackReport)["schema"]
    assert schema["type"] == "object"
    assert "report" not in schema["properties"]
    assert "anyOf" not in schema["properties"]["artifacts"]["items"]
    assert "anyOf" not in schema["properties"]["attack_outcome"]
    assert set(schema["properties"]["attack_outcome"]["enum"]) == {
        "critical_issue", "inconclusive", "no_critical_issue",
    }
    for field in ("consumed_entity_ids", "addressed_obligation_ids",
                  "necessity_contract_entity_ids"):
        assert FlatAttackReport.model_json_schema()["properties"][field]["maxItems"] == 0
        assert schema["properties"][field]["enum"] == [[]]
    assert schema["properties"]["operation"]["enum"] == ["attack"]
    assert schema["properties"]["necessity_outcome"]["enum"] == ["not_applicable"]
    assert schema["properties"]["necessity_audit"]["type"] == "null"
    assert schema["properties"]["human_judgment_required"]["enum"] == [False]
    assert schema["properties"]["human_judgment_reason"]["type"] == "null"


@pytest.mark.parametrize(("field", "value"), (
    ("operation", "develop"),
    ("consumed_entity_ids", [1]),
    ("addressed_obligation_ids", [1]),
    ("necessity_outcome", "alternative_route_found"),
    ("necessity_contract_entity_ids", [1]),
    ("necessity_audit", {"argument": "Not an attack field."}),
    ("human_judgment_required", True),
    ("human_judgment_reason", "Ask the human to resolve this."),
))
def test_flat_attack_rejects_structurally_illegal_fields(field, value):
    payload = attack_payload("inconclusive", False, ["Boundary remains open."])
    payload[field] = value
    with pytest.raises(ValidationError):
        FlatAttackReport.model_validate(payload)


@pytest.mark.parametrize("provider", ("openai", "anthropic"))
def test_attack_parser_requires_the_provider_shape(provider):
    report = attack_payload("no_critical_issue", False, [])
    model = _execution_response_model("attack", provider)
    expected = {"report": report} if provider == "openai" else report
    wrong = report if provider == "openai" else {"report": report}
    assert _parse_execution_report(json.dumps(expected), model).model_dump() == (
        ResearchStepReport.model_validate(report).model_dump()
    )
    with pytest.raises(ModelOutputError):
        _parse_execution_report(json.dumps(wrong), model)
    if provider == "anthropic":
        with pytest.raises(ModelOutputError):
            _parse_execution_report(json.dumps(report) + " trailing text", model)


@pytest.mark.parametrize("operation", ("develop", "synthesize", "prove", "reframe"))
@pytest.mark.parametrize("provider", ("openai", "anthropic"))
def test_non_attack_response_model_is_unchanged(operation, provider):
    assert _execution_response_model(operation, provider) is ResearchStepReport


@pytest.mark.parametrize("provider_name", ("openai", "anthropic"))
@pytest.mark.parametrize("outcome", ("critical_issue", "inconclusive", "no_critical_issue"))
def test_provider_attack_shape_runs_shared_semantic_validation(monkeypatch, tmp_path, provider_name, outcome):
    init_workspace(monkeypatch, tmp_path)
    ws, primary = make_research_workstream()
    add_linked_research_entity(ws, "ProofAttempt", "Precise argument to test",
                               related_entity_ids=[primary])
    def respond(decision, _):
        critical = [artifact("counterexample", "Concrete failing schedule.",
                             "failing_schedule", [decision["target_entity_id"]])]
        return step_report(
            decision, critical if outcome == "critical_issue" else [],
            attack_outcome=outcome,
            unresolved=["Boundary remains open."] if outcome == "inconclusive" else [],
        )

    provider = DynamicProvider(respond)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    original = controller._validate_step_report
    validated = []

    def validate(report, context, choice):
        validated.append((type(report), report.attack_outcome, choice.operation))
        return original(report, context, choice)

    monkeypatch.setattr(controller, "_validate_step_report", validate)
    result = research(ws, provider_name, max_calls=1)
    assert result.calls_made == 1
    assert validated and all(observed == outcome and operation == "attack"
                             for _, observed, operation in validated)
    expected_report = (FlatAttackReport if provider_name == "anthropic" else {
        "critical_issue": controller.CriticalAttackReport,
        "inconclusive": controller.InconclusiveAttackReport,
        "no_critical_issue": controller.NoCriticalIssueAttackReport,
    }[outcome])
    assert {model for model, _, _ in validated} == {expected_report}
    request = provider.calls[0]
    assert request["response_model"] is _execution_response_model("attack", provider_name)
    example = json.loads(request["prompt"].split("Return ONLY strict JSON with exactly this shape:\n", 1)[1])
    assert ("report" in example) == (provider_name == "openai")
    assert "consumed_entity_ids MUST be [] and addressed_obligation_ids MUST be []" in request["prompt"]
    with connect() as con:
        call = con.execute("SELECT status,purpose FROM api_calls").fetchone()
    assert tuple(call) == ("completed", "research:attack")


@pytest.mark.parametrize("field", ("consumed_entity_ids", "addressed_obligation_ids"))
def test_anthropic_nonempty_attack_ids_fail_before_scientific_persistence(monkeypatch, tmp_path, field):
    init_workspace(monkeypatch, tmp_path)
    ws, primary = make_research_workstream()
    add_linked_research_entity(ws, "ProofAttempt", "Precise argument to test",
                               related_entity_ids=[primary])

    def respond(decision, _):
        report = step_report(decision, [], attack_outcome="inconclusive",
                             unresolved=["Boundary remains open."])
        report[field] = [primary]
        return report

    provider = DynamicProvider(respond)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    before = for_workstream(ws)
    with pytest.raises(ModelOutputError, match="FlatAttackReport") as error:
        research(ws, "anthropic", max_calls=1)
    assert field in str(error.value)
    after = for_workstream(ws)
    assert before.entities == after.entities
    assert before.relations == after.relations
    assert len(provider.calls) == 1
    with connect() as con:
        call = con.execute("SELECT * FROM api_calls").fetchone()
        iteration = con.execute("SELECT * FROM research_iterations").fetchone()
        assert con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0
    assert call["status"] == "failed" and call["input_tokens"] == 500
    assert json.loads(call["response_text"])[field] == [primary]
    assert iteration["status"] == "error"


@pytest.mark.parametrize("fault", ("empty_uncertainty", "blank_uncertainty", "missing_critical", "critical_mismatch", "unknown_reference"))
@pytest.mark.parametrize("provider_name", ("openai", "anthropic"))
def test_invalid_attack_is_auditable_without_retry_or_graph_write(monkeypatch, tmp_path, fault, provider_name):
    init_workspace(monkeypatch, tmp_path)
    ws, primary = make_research_workstream()
    add_linked_research_entity(ws, "ProofAttempt", "Precise argument to test", related_entity_ids=[primary])
    def respond(decision, _):
        if fault in {"empty_uncertainty", "blank_uncertainty"}:
            return step_report(decision, [], attack_outcome="inconclusive",
                               unresolved=["   "] if fault == "blank_uncertainty" else [])
        if fault == "missing_critical":
            return step_report(decision, [], attack_outcome="critical_issue")
        refs = [decision["target_entity_id"]]
        if fault == "unknown_reference":
            refs.append(999999)
        return step_report(decision, [artifact("counterexample", "Concrete bad execution.",
            "bad_execution", refs)], attack_outcome="critical_issue" if fault == "unknown_reference" else "no_critical_issue")
    provider = DynamicProvider(respond)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    before = for_workstream(ws)
    with pytest.raises(ModelOutputError):
        research(ws, provider_name, max_calls=1)
    after = for_workstream(ws)
    assert before.entities == after.entities
    assert before.relations == after.relations
    assert len(provider.calls) == 1
    assert provider.calls[0]["response_model"] is _execution_response_model("attack", provider_name)
    with connect() as con:
        call = con.execute("SELECT * FROM api_calls").fetchone()
        iteration = con.execute("SELECT * FROM research_iterations").fetchone()
        assert con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0
    assert call["status"] == "failed" and call["error_message"]
    assert call["cost_usd"] == 0.007 and call["input_tokens"] == 500
    assert call["provider"] == provider_name and call["model"] and call["created_at"]
    assert call["purpose"] == "research:attack" and call["workstream_id"] == ws
    raw = json.loads(call["response_text"])
    assert (raw["report"] if provider_name == "openai" else raw) == respond(
        {"operation": "attack", "target_entity_id": iteration["target_entity_id"], "required_consumed_entity_ids": []}, 1)
    assert iteration["status"] == "error" and iteration["error_message"]


def test_inconclusive_schema_requires_nonblank_uncertainty():
    schema = ResearchAttackResponse.model_json_schema()
    uncertainty = schema["$defs"]["InconclusiveAttackReport"]["properties"]["could_not_determine"]
    assert uncertainty["minItems"] == 1
    assert uncertainty["items"]["pattern"] == r"\S"
