from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dotenv import load_dotenv
from .errors import ConfigurationError
from .models import ModelResult

load_dotenv()

@dataclass(frozen=True)
class ModelSpec:
    provider: str
    input_usd_per_million: float
    output_usd_per_million: float


# Verified 2026-09-25 against the official provider pages:
# https://platform.openai.com/docs/models and
# https://www.anthropic.com/claude/opus
# These are standard, uncached, global API rates. Update this table deliberately;
# an unknown model must never be treated as free.
MODEL_SPECS = {
    "gpt-5.6-sol": ModelSpec("openai", 4.0, 20.0),
    "gpt-5.6-terra": ModelSpec("openai", 2.0, 12.0),
    "gpt-5.6-luna": ModelSpec("openai", 0.20, 1.20),
    "claude-opus-5-5": ModelSpec("anthropic", 4.0, 20.0),
}


def get_model_spec(model: str, provider: str | None = None) -> ModelSpec:
    try:
        spec = MODEL_SPECS[model]
    except KeyError as exc:
        raise ConfigurationError(
            f"No trusted pricing is configured for model {model!r}; refusing an unpriced call."
        ) from exc
    if provider is not None and spec.provider != provider:
        raise ConfigurationError(
            f"Model {model!r} belongs to {spec.provider}, not configured provider {provider}."
        )
    return spec


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts cannot be negative")
    spec = get_model_spec(model)
    return (
        input_tokens / 1_000_000 * spec.input_usd_per_million
        + output_tokens / 1_000_000 * spec.output_usd_per_million
    )


def conservative_call_cost(model: str, prompt: str, max_output_tokens: int) -> float:
    """Upper-bound a normal text call locally for budget admission.

    UTF-8 bytes are used as a deliberately conservative input-token bound. The
    provider-side output cap supplies the output-token bound.
    """
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    input_token_bound = max(1, len(prompt.encode("utf-8")))
    return estimate_cost(model, input_token_bound, max_output_tokens)


class Provider(ABC):
    name: str

    @abstractmethod
    def complete(
        self, *, model: str, prompt: str, effort: str = "high", max_output_tokens: int
    ) -> ModelResult:
        raise NotImplementedError


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigurationError("OPENAI_API_KEY is not set.")
        self.client = OpenAI()

    def complete(
        self, *, model: str, prompt: str, effort: str = "high", max_output_tokens: int
    ) -> ModelResult:
        get_model_spec(model, self.name)
        response = self.client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": effort},
            max_output_tokens=max_output_tokens,
        )
        usage = getattr(response, "usage", None)
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        return ModelResult(
            text=response.output_text or "",
            input_tokens=inp,
            output_tokens=out,
            cost_usd=estimate_cost(model, inp, out),
        )


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ConfigurationError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic()

    def complete(
        self, *, model: str, prompt: str, effort: str = "high", max_output_tokens: int
    ) -> ModelResult:
        get_model_spec(model, self.name)
        message = self.client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            output_config={"effort": effort},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "\n".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(message, "usage", None)
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        return ModelResult(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            cost_usd=estimate_cost(model, inp, out),
        )


def get_provider(name: str) -> Provider:
    if name == "openai":
        return OpenAIProvider()
    if name == "anthropic":
        return AnthropicProvider()
    raise ConfigurationError(f"Unknown provider: {name}")
