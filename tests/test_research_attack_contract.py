"""The wire contract narrows outcomes without changing scientific validation."""
import json

import pytest
from pydantic import ValidationError

from theory.db import connect
from theory.errors import ModelOutputError
from theory.providers import _response_format
from theory.research import ResearchAttackResponse, ResearchStepReport, research
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


@pytest.mark.parametrize("provider", ("openai", "anthropic"))
def test_provider_attack_schema_keeps_outcome_variants(provider):
    schema = _response_format(provider, ResearchAttackResponse)["schema"]
    assert schema["type"] == "object"
    assert len(schema["properties"]["report"]["anyOf"]) == 3
    text = json.dumps(schema)
    for outcome in ("critical_issue", "inconclusive", "no_critical_issue"):
        assert outcome in text
    assert '"not_applicable", "critical_issue"' not in text


@pytest.mark.parametrize("fault", ("empty_uncertainty", "blank_uncertainty", "missing_critical", "critical_mismatch", "unknown_reference"))
def test_invalid_attack_is_auditable_without_retry_or_graph_write(monkeypatch, tmp_path, fault):
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
        research(ws, "openai", max_calls=1)
    after = for_workstream(ws)
    assert before.entities == after.entities
    assert before.relations == after.relations
    assert len(provider.calls) == 1
    assert provider.calls[0]["response_model"] is ResearchAttackResponse
    with connect() as con:
        call = con.execute("SELECT * FROM api_calls").fetchone()
        iteration = con.execute("SELECT * FROM research_iterations").fetchone()
        assert con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0
    assert call["status"] == "failed" and call["error_message"]
    assert call["cost_usd"] == 0.007 and call["input_tokens"] == 500
    assert call["provider"] == "openai" and call["model"] and call["created_at"]
    assert call["purpose"] == "research:attack" and call["workstream_id"] == ws
    assert json.loads(call["response_text"])["report"] == respond(
        {"operation": "attack", "target_entity_id": iteration["target_entity_id"], "required_consumed_entity_ids": []}, 1)
    assert iteration["status"] == "error" and iteration["error_message"]


def test_inconclusive_schema_requires_nonblank_uncertainty():
    schema = ResearchAttackResponse.model_json_schema()
    uncertainty = schema["$defs"]["InconclusiveAttackReport"]["properties"]["could_not_determine"]
    assert uncertainty["minItems"] == 1
    assert uncertainty["items"]["pattern"] == r"\S"
