from __future__ import annotations
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from dotenv import load_dotenv
from pydantic import BaseModel
from .errors import ConfigurationError, TelemetryError
from .models import ModelResult

load_dotenv()

@dataclass(frozen=True)
class ModelSpec:
    provider: str
    input_usd_per_million: float
    output_usd_per_million: float
    cache_read_usd_per_million: float
    cache_write_5m_usd_per_million: float
    cache_write_1h_usd_per_million: float | None


# Verified 2026-09-27 against the official provider pages:
# https://platform.openai.com/docs/models and
# https://www.anthropic.com/claude/opus
# These are standard global API rates. OpenAI's 5m bucket is an accounting
# convention for its cache writes, not an Anthropic-style TTL claim.
# Update this table deliberately;
# an unknown model must never be treated as free.
MODEL_SPECS = {
    "gpt-6-luna": ModelSpec("openai", 0.10, 0.50, 0.01, 0.125, None),
    "gpt-6-sol": ModelSpec("openai", 2.0, 10.0, 0.20, 2.50, None),
    "gpt-6-astra": ModelSpec("openai", 10.0, 50.0, 1.0, 12.50, None),
    "gpt-5.6-sol": ModelSpec("openai", 4.0, 20.0, 0.40, 5.0, None),
    "gpt-5.6-terra": ModelSpec("openai", 2.0, 12.0, 0.20, 2.50, None),
    "gpt-5.6-luna": ModelSpec("openai", 0.20, 1.20, 0.02, 0.25, None),
    "claude-opus-5-5": ModelSpec("anthropic", 4.0, 20.0, 0.20, 5.0, 8.0),
    "claude-sonnet-5": ModelSpec("anthropic", 2.0, 10.0, 0.20, 2.50, 4.0),
    "claude-fable-5-1": ModelSpec("anthropic", 10.0, 50.0, 0.25, 12.50, 20.0),
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
    """Compatibility convenience: all input uncached."""
    return estimate_usage_cost(
        model, uncached_input_tokens=input_tokens, output_tokens=output_tokens
    ).cost_usd


@dataclass(frozen=True)
class UsageCost:
    uncached_input_cost_usd: float
    cache_read_cost_usd: float
    cache_write_cost_usd: float
    output_cost_usd: float

    @property
    def cost_usd(self) -> float:
        return (self.uncached_input_cost_usd + self.cache_read_cost_usd
                + self.cache_write_cost_usd + self.output_cost_usd)


def estimate_usage_cost(
    model: str, *, uncached_input_tokens: int, output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_write_5m_input_tokens: int = 0,
    cache_write_1h_input_tokens: int = 0,
) -> UsageCost:
    counts = (uncached_input_tokens, cache_read_input_tokens, cache_write_5m_input_tokens,
              cache_write_1h_input_tokens, output_tokens)
    if any(type(count) is not int or count < 0 for count in counts):
        raise TelemetryError("Token counts must be non-negative integers.")
    spec = get_model_spec(model)
    if cache_write_1h_input_tokens and spec.cache_write_1h_usd_per_million is None:
        raise TelemetryError(f"No trusted 1h cache-write price for model {model!r}.")
    return UsageCost(
        uncached_input_tokens / 1_000_000 * spec.input_usd_per_million,
        cache_read_input_tokens / 1_000_000 * spec.cache_read_usd_per_million,
        cache_write_5m_input_tokens / 1_000_000 * spec.cache_write_5m_usd_per_million
        + cache_write_1h_input_tokens / 1_000_000 * (spec.cache_write_1h_usd_per_million or 0),
        output_tokens / 1_000_000 * spec.output_usd_per_million,
    )


def _usage_field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _usage_count(obj, key, *, required=False) -> int:
    value = _usage_field(obj, key)
    if value is None and not required:
        return 0
    if type(value) is not int or value < 0:
        raise TelemetryError(f"Provider usage {key} must be a non-negative integer.")
    return value


def _normalized_usage(model: str, usage, *, provider: str) -> dict:
    inp = _usage_count(usage, "input_tokens", required=True)
    out = _usage_count(usage, "output_tokens", required=True)
    if provider == "openai":
        details = _usage_field(usage, "input_tokens_details")
        read = _usage_count(details, "cached_tokens")
        write5 = _usage_count(details, "cache_write_tokens")
        write1 = 0
        uncached = inp - read - write5
        if uncached < 0:
            raise TelemetryError("OpenAI cache usage exceeds total input_tokens.")
        reasoning_key = "reasoning_tokens"
    else:
        uncached = inp
        read = _usage_count(usage, "cache_read_input_tokens")
        writes = _usage_count(usage, "cache_creation_input_tokens")
        breakdown = _usage_field(usage, "cache_creation")
        if breakdown is None and writes:
            raise TelemetryError("Anthropic cache writes require a TTL breakdown to price accurately.")
        write5 = _usage_count(breakdown, "ephemeral_5m_input_tokens", required=breakdown is not None)
        write1 = _usage_count(breakdown, "ephemeral_1h_input_tokens", required=breakdown is not None)
        if writes != write5 + write1:
            raise TelemetryError("Anthropic cache_creation_input_tokens does not match TTL breakdown.")
        reasoning_key = "thinking_tokens"
    output_details = _usage_field(usage, "output_tokens_details")
    reasoning = (None if _usage_field(output_details, reasoning_key) is None
                 else _usage_count(output_details, reasoning_key))
    cost = estimate_usage_cost(
        model, uncached_input_tokens=uncached, cache_read_input_tokens=read,
        cache_write_5m_input_tokens=write5, cache_write_1h_input_tokens=write1,
        output_tokens=out,
    )
    return dict(
        input_tokens=uncached + read + write5 + write1,
        uncached_input_tokens=uncached, cache_read_input_tokens=read,
        cache_write_input_tokens=write5 + write1,
        cache_write_5m_input_tokens=write5, cache_write_1h_input_tokens=write1,
        output_tokens=out, reasoning_tokens=reasoning,
        **asdict(cost), cost_usd=cost.cost_usd,
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
        status = str(getattr(response, "status", "completed") or "completed")
        incomplete_details = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete_details, "reason", None)
        return ModelResult(
            text=response.output_text or "",
            **_normalized_usage(model, usage, provider=self.name),
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
        stop_reason = getattr(message, "stop_reason", None)
        return ModelResult(
            text=text,
            **_normalized_usage(model, usage, provider=self.name),
            response_status="incomplete" if stop_reason == "max_tokens" else "completed",
            incomplete_reason="max_output_tokens" if stop_reason == "max_tokens" else None,
        )


def get_provider(name: str) -> Provider:
    if name == "openai":
        return OpenAIProvider()
    if name == "anthropic":
        return AnthropicProvider()
    raise ConfigurationError(f"Unknown provider: {name}")
