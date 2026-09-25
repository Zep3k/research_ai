from types import SimpleNamespace

from theory.providers import AnthropicProvider, OpenAIProvider


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
        model="claude-opus-5-5", prompt="question", effort="high", max_output_tokens=1234
    )

    assert create.kwargs["output_config"] == {"effort": "high"}
    assert create.kwargs["max_tokens"] == 1234
    assert result.text == "answer"
