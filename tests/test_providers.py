import json
from types import SimpleNamespace

from theory.jsonutil import parse_json_model
from theory.providers import AnthropicProvider, OpenAIProvider
from theory.research import ResearchStepReport


class CaptureCreate:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def create(self, **kwargs):
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
        response_model=ResearchStepReport,
    )

    assert create.kwargs["output_config"] == {"effort": "high"}
    assert create.kwargs["max_tokens"] == 1234
    assert "text" not in create.kwargs
    assert result.text == "answer"
