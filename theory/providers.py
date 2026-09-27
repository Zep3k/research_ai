from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel
from .errors import ConfigurationError
from .models import ModelResult

load_dotenv()

@dataclass(frozen=True)
class ModelSpec:
    provider: str
    input_usd_per_million: float
    output_usd_per_million: float


# Verified 2026-09-27 against the official provider pages:
# https://platform.openai.com/docs/models and
# https://www.anthropic.com/claude/opus
# These are standard, uncached, global API rates. Update this table deliberately;
# an unknown model must never be treated as free.
MODEL_SPECS = {
    "gpt-6-luna": ModelSpec("openai", 0.10, 0.50),
    "gpt-6-sol": ModelSpec("openai", 2.0, 10.0),
    "gpt-6-astra": ModelSpec("openai", 10.0, 50.0),
    "gpt-5.6-sol": ModelSpec("openai", 4.0, 20.0),
    "gpt-5.6-terra": ModelSpec("openai", 2.0, 12.0),
    "gpt-5.6-luna": ModelSpec("openai", 0.20, 1.20),
    "claude-opus-5-5": ModelSpec("anthropic", 4.0, 20.0),
    "claude-sonnet-5": ModelSpec("anthropic", 2.0, 10.0),
    "claude-fable-5-1": ModelSpec("anthropic", 10.0, 50.0),
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
        self,
        *,
        model: str,
        prompt: str,
        effort: str = "high",
        max_output_tokens: int,
        response_model: type[BaseModel] | None = None,
    ) -> ModelResult:
        raise NotImplementedError


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigurationError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(max_retries=0)

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        effort: str = "high",
        max_output_tokens: int,
        response_model: type[BaseModel] | None = None,
    ) -> ModelResult:
        get_model_spec(model, self.name)
        request = {
            "model": model,
            "input": prompt,
            "reasoning": {"effort": effort},
            "max_output_tokens": max_output_tokens,
        }
        if response_model is not None:
            from openai.lib._parsing._responses import type_to_text_format_param

            request["text"] = {"format": type_to_text_format_param(response_model)}
        response = self.client.responses.create(
            **request,
        )
        usage = getattr(response, "usage", None)
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        status = str(getattr(response, "status", "completed") or "completed")
        incomplete_details = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete_details, "reason", None)
        return ModelResult(
            text=response.output_text or "",
            input_tokens=inp,
            output_tokens=out,
            cost_usd=estimate_cost(model, inp, out),
            response_status=status,
            incomplete_reason=incomplete_reason,
        )


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ConfigurationError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(max_retries=0)

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        effort: str = "high",
        max_output_tokens: int,
        response_model: type[BaseModel] | None = None,
    ) -> ModelResult:
        get_model_spec(model, self.name)
        output_config = {"effort": effort}
        if response_model is not None:
            from anthropic import transform_schema

            # Use the SDK's supported schema subset, then validate the full
            # Pydantic contract locally after usage has been recorded.
            # The SDK moves `const` to a description; preserve single-literal
            # artifact tags as supported one-value enums before conversion.
            schema = response_model.model_json_schema()

            def normalize_literals(node):
                if isinstance(node, dict):
                    if "const" in node:
                        node["enum"] = [node.pop("const")]
                    for value in node.values():
                        normalize_literals(value)
                elif isinstance(node, list):
                    for value in node:
                        normalize_literals(value)

            normalize_literals(schema)
            output_config["format"] = {
                "type": "json_schema",
                "schema": transform_schema(schema),
            }
        message = self.client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            output_config=output_config,
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
