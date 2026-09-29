import json
from types import SimpleNamespace

import pytest

from theory.jsonutil import parse_json_model
from theory.providers import AnthropicProvider, OpenAIProvider
from theory.research import FlatAttackReport, ResearchStepReport, _parse_execution_report


class CaptureCreate:
    def __init__(self, result):
        self.result = result
        self.kwargs = None
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return self.result


def test_openai_adapter_passes_reasoning_and_hard_output_cap():
    create = CaptureCreate(
        SimpleNamespace(
            output_text="answer",
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )
    )
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.client = SimpleNamespace(responses=create)

    result = provider.complete(
        model="gpt-5.6-sol", prompt="question", effort="high", max_output_tokens=1234
    )

    assert create.kwargs["reasoning"] == {"effort": "high"}
    assert create.kwargs["max_output_tokens"] == 1234
    assert result.input_tokens == 10
    assert result.output_tokens == 20


def test_openai_adapter_reports_incomplete_response_with_usage():
    create = CaptureCreate(
        SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output_text='{"operation": "develop", "summary": "cut off',
            usage=SimpleNamespace(input_tokens=321, output_tokens=654),
        )
    )
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.client = SimpleNamespace(responses=create)

    result = provider.complete(
        model="gpt-5.6-sol", prompt="question", effort="high", max_output_tokens=32_000
    )

    assert result.response_status == "incomplete"
    assert result.incomplete_reason == "max_output_tokens"
    assert result.input_tokens == 321
    assert result.output_tokens == 654
    assert result.cost_usd > 0


def test_openai_adapter_uses_strict_schema_for_complete_research_report():
    report = {
        "operation": "attack",
        "target_entity_id": 1,
        "summary": "The bounded attack was inconclusive.",
        "artifacts": [],
        "consumed_entity_ids": [],
        "addressed_obligation_ids": [],
        "attack_outcome": "inconclusive",
        "could_not_determine": [],
        "human_judgment_required": False,
        "human_judgment_reason": None,
    }
    create = CaptureCreate(
        SimpleNamespace(
            status="completed",
            incomplete_details=None,
            output_text=json.dumps(report),
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )
    )
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.client = SimpleNamespace(responses=create)

    result = provider.complete(
        model="gpt-5.6-sol",
        prompt="question",
        effort="high",
        max_output_tokens=32_000,
        response_model=ResearchStepReport,
    )

    output_format = create.kwargs["text"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    assert output_format["name"] == "ResearchStepReport"
    assert output_format["schema"]["additionalProperties"] is False
    artifact_schema = output_format["schema"]["properties"]["artifacts"]["items"]
    assert "anyOf" in artifact_schema
    assert "oneOf" not in artifact_schema
    variant_names = {
        item["$ref"].rsplit("/", 1)[-1] for item in artifact_schema["anyOf"]
    }
    assert variant_names == {
        "GeneralResearchArtifact",
        "ObstructionResearchArtifact",
        "FailedApproachResearchArtifact",
    }
    definitions = output_format["schema"]["$defs"]

    def branch_statuses(name):
        alternatives = definitions[name]["properties"]["branch_status"]["anyOf"]
        return next(set(item["enum"]) for item in alternatives if "enum" in item)

    assert branch_statuses("GeneralResearchArtifact") == {"promising", "unresolved"}
    assert branch_statuses("ObstructionResearchArtifact") == {
        "promising",
        "blocked",
        "unresolved",
    }
    assert branch_statuses("FailedApproachResearchArtifact") == {
        "promising",
        "blocked",
        "failed",
        "refuted",
        "unresolved",
    }
    assert parse_json_model(result.text, ResearchStepReport).operation == "attack"


def test_anthropic_adapter_passes_effort_and_hard_output_cap():
    create = CaptureCreate(
        SimpleNamespace(
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="answer")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )
    )
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.client = SimpleNamespace(messages=create)

    result = provider.complete(
        model="claude-opus-5-5",
        prompt="question",
        effort="high",
        max_output_tokens=1234,
        response_model=None,
    )

    assert create.kwargs["output_config"] == {"effort": "high"}
    assert create.kwargs["max_tokens"] == 1234
    assert "text" not in create.kwargs
    assert result.text == "answer"


@pytest.mark.parametrize("response_model", (ResearchStepReport, FlatAttackReport))
def test_anthropic_structured_report_preserves_selected_schema(response_model):
    report = {
        "operation": "attack", "target_entity_id": 1, "summary": "Bounded attack.",
        "artifacts": [], "consumed_entity_ids": [], "addressed_obligation_ids": [],
        "attack_outcome": "no_critical_issue", "could_not_determine": [],
        "human_judgment_required": False, "human_judgment_reason": None,
    }
    if response_model is FlatAttackReport:
        report.pop("attack_outcome")
    capture = CaptureCreate(SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(report))],
        usage=SimpleNamespace(input_tokens=100, output_tokens=200),
    ))
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.client = SimpleNamespace(messages=capture)

    result = provider.complete(
        model="claude-opus-5-5", prompt="question", effort="medium",
        max_output_tokens=12_000, response_model=response_model,
    )

    assert capture.calls == 1
    assert capture.kwargs["output_config"]["effort"] == "medium"
    assert capture.kwargs["max_tokens"] == 12_000
    output_format = capture.kwargs["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    schema = output_format["schema"]
    assert schema["additionalProperties"] is False
    assert "report" not in schema["properties"]
    if response_model is FlatAttackReport:
        assert "anyOf" not in schema["properties"]["artifacts"]["items"]
        assert "attack_outcome" not in schema["properties"]
        assert set(schema["$defs"]) == {"ResearchArtifact"}
    else:
        assert "anyOf" in schema["properties"]["artifacts"]["items"]
        for variant, tag in (
            ("ObstructionResearchArtifact", "obstruction"),
            ("FailedApproachResearchArtifact", "failed_approach"),
        ):
            assert schema["$defs"][variant]["properties"]["artifact_type"]["enum"] == [tag]
            assert schema["$defs"][variant]["additionalProperties"] is False
    assert _parse_execution_report(result.text, response_model).attack_outcome == "no_critical_issue"
    assert result.cost_usd == pytest.approx(0.0044)


@pytest.mark.parametrize("provider_class,module,constructor,key", [
    (OpenAIProvider, "openai", "OpenAI", "OPENAI_API_KEY"),
    (AnthropicProvider, "anthropic", "Anthropic", "ANTHROPIC_API_KEY"),
])
def test_provider_clients_disable_automatic_retries(
    monkeypatch, provider_class, module, constructor, key
):
    calls = []
    monkeypatch.setenv(key, "offline-test-key")
    monkeypatch.setattr(f"{module}.{constructor}", lambda **kwargs: calls.append(kwargs))
    provider_class()
    assert calls == [{"max_retries": 0}]
