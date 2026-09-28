import sqlite3
from types import SimpleNamespace as NS

import pytest
from typer.testing import CliRunner

from theory.cli import app
from theory.config import Config
from theory.db import CALL_TELEMETRY_COLUMNS, SCHEMA, connect, monthly_spend, utcnow
from theory.errors import ModelOutputError, TelemetryError, TheoryError
from theory.model_calls import budget_guard, call_model
from theory.models import ModelResult
from theory.providers import (
    AnthropicProvider, OpenAIProvider, estimate_cost, estimate_usage_cost,
)
from test_providers import CaptureCreate
from test_research import init_workspace, make_research_workstream


def adapter(provider_name, usage, *, incomplete=False):
    if provider_name == "openai":
        provider = OpenAIProvider.__new__(OpenAIProvider)
        capture = CaptureCreate(NS(
            output_text="response", usage=usage,
            status="incomplete" if incomplete else "completed",
            incomplete_details=NS(reason="max_output_tokens") if incomplete else None,
        ))
        provider.client = NS(responses=capture)
    else:
        provider = AnthropicProvider.__new__(AnthropicProvider)
        capture = CaptureCreate(NS(
            content=[NS(type="text", text="response")], usage=usage,
            stop_reason="max_tokens" if incomplete else "end_turn",
        ))
        provider.client = NS(messages=capture)
    return provider, capture


def complete(provider_name, usage, *, model=None):
    provider, _ = adapter(provider_name, usage)
    return provider.complete(
        model=model or ("gpt-6-luna" if provider_name == "openai" else "claude-sonnet-5"),
        prompt="test", max_output_tokens=1000,
    )


@pytest.mark.parametrize("read,write,reasoning", [(0, 0, None), (600, 0, None), (0, 300, None), (600, 300, 55)])
def test_openai_normalizes_usage_and_optional_details(read, write, reasoning):
    usage = NS(input_tokens=1000, output_tokens=100)
    if read or write:
        usage.input_tokens_details = NS(cached_tokens=read, cache_write_tokens=write)
    if reasoning is not None:
        usage.output_tokens_details = NS(reasoning_tokens=reasoning)
    result = complete("openai", usage)
    assert result.input_tokens == 1000
    assert result.uncached_input_tokens == 1000 - read - write
    assert result.cache_read_input_tokens == read
    assert result.cache_write_input_tokens == result.cache_write_5m_input_tokens == write
    assert result.cache_write_1h_input_tokens == 0
    assert result.reasoning_tokens == reasoning
    assert result.input_tokens == result.uncached_input_tokens + read + write
    assert result.cost_usd == pytest.approx((1000-read-write)*0.10/1e6 + read*0.01/1e6 + write*0.125/1e6 + 100*0.50/1e6)


@pytest.mark.parametrize("read,write5,write1,thinking", [
    (0, 0, 0, None), (600, 0, 0, None), (0, 300, 0, None),
    (0, 0, 300, None), (600, 200, 300, 55),
])
def test_anthropic_normalizes_total_and_ttl_usage(read, write5, write1, thinking):
    usage = NS(input_tokens=100, output_tokens=100)
    if read:
        usage.cache_read_input_tokens = read
    if write5 or write1:
        usage.cache_creation_input_tokens = write5 + write1
        usage.cache_creation = NS(ephemeral_5m_input_tokens=write5, ephemeral_1h_input_tokens=write1)
    if thinking is not None:
        usage.output_tokens_details = NS(thinking_tokens=thinking)
    result = complete("anthropic", usage)
    assert result.input_tokens == 100 + read + write5 + write1
    assert result.uncached_input_tokens == 100
    assert result.cache_read_input_tokens == read
    assert result.cache_write_input_tokens == write5 + write1
    assert result.cache_write_5m_input_tokens == write5
    assert result.cache_write_1h_input_tokens == write1
    assert result.reasoning_tokens == thinking
    assert result.cost_usd == pytest.approx((100*2 + read*0.20 + write5*2.5 + write1*4 + 100*10)/1e6)


@pytest.mark.parametrize("provider_name,usage,error", [
    ("openai", NS(input_tokens=5, output_tokens=1, input_tokens_details=NS(cached_tokens=6)), "exceeds"),
    ("openai", NS(input_tokens=-1, output_tokens=1), "non-negative"),
    ("openai", NS(input_tokens=1, output_tokens=1, output_tokens_details=NS(reasoning_tokens=-1)), "non-negative"),
    ("openai", None, "input_tokens"),
    ("anthropic", NS(input_tokens=1, output_tokens=1, cache_creation_input_tokens=10), "TTL breakdown"),
    ("anthropic", NS(input_tokens=1, output_tokens=1, cache_creation_input_tokens=10,
                     cache_creation=NS(ephemeral_5m_input_tokens=5, ephemeral_1h_input_tokens=6)), "does not match"),
    ("anthropic", NS(input_tokens=1, output_tokens=1, cache_read_input_tokens=-1), "non-negative"),
])
def test_impossible_or_unpriced_usage_fails_closed(provider_name, usage, error):
    with pytest.raises(TelemetryError, match=error):
        complete(provider_name, usage)


@pytest.mark.parametrize("model,uncached,read,write5,write1,output", [
    ("gpt-6-luna", .10, .01, .125, None, .50),
    ("gpt-6-sol", 2, .20, 2.50, None, 10),
    ("gpt-6-astra", 10, 1, 12.50, None, 50),
    ("claude-sonnet-5", 2, .20, 2.50, 4, 10),
    ("claude-opus-5-5", 4, .20, 5, 8, 20),
    ("claude-fable-5-1", 10, .25, 12.50, 20, 50),
    ("gpt-5.6-sol", 4, .40, 5, None, 20),
    ("gpt-5.6-terra", 2, .20, 2.50, None, 12),
    ("gpt-5.6-luna", .20, .02, .25, None, 1.20),
])
def test_precise_cache_rates_and_uncached_compatibility(model, uncached, read, write5, write1, output):
    cost = estimate_usage_cost(
        model, uncached_input_tokens=1_000_000, cache_read_input_tokens=1_000_000,
        cache_write_5m_input_tokens=1_000_000,
        cache_write_1h_input_tokens=1_000_000 if write1 is not None else 0,
        output_tokens=1_000_000,
    )
    assert cost.uncached_input_cost_usd == uncached
    assert cost.cache_read_cost_usd == read
    assert cost.cache_write_cost_usd == write5 + (write1 or 0)
    assert cost.output_cost_usd == output
    assert cost.cost_usd == uncached + read + write5 + (write1 or 0) + output
    assert estimate_cost(model, 1_000_000, 1_000_000) == uncached + output


def test_unpriced_cache_bucket_and_negative_counts_are_rejected():
    with pytest.raises(TelemetryError, match="1h cache-write price"):
        estimate_usage_cost("gpt-6-sol", uncached_input_tokens=0, output_tokens=0, cache_write_1h_input_tokens=1)
    with pytest.raises(TelemetryError, match="non-negative"):
        estimate_usage_cost("gpt-6-sol", uncached_input_tokens=0, output_tokens=0, cache_read_input_tokens=-1)


@pytest.mark.parametrize("kwargs,error", [
    ({"input_tokens": 1}, "input_tokens"),
    ({"input_tokens": 1, "cache_write_input_tokens": 1}, "accounting buckets"),
    ({"cost_usd": .1}, "four cost components"),
    ({"reasoning_tokens": -1}, "greater than or equal"),
])
def test_model_result_enforces_accounting_identities(kwargs, error):
    with pytest.raises(ValueError, match=error):
        ModelResult(text="test", **kwargs)


@pytest.mark.parametrize("provider_name,incomplete", [
    ("openai", False), ("openai", True), ("anthropic", False), ("anthropic", True),
])
def test_receipts_preserve_detailed_usage_even_on_incomplete_response(monkeypatch, tmp_path, provider_name, incomplete):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    usage = (NS(input_tokens=1000, output_tokens=100,
                input_tokens_details=NS(cached_tokens=600, cache_write_tokens=300),
                output_tokens_details=NS(reasoning_tokens=50)) if provider_name == "openai"
             else NS(input_tokens=100, output_tokens=100, cache_read_input_tokens=600,
                     cache_creation_input_tokens=300,
                     cache_creation=NS(ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=200),
                     output_tokens_details=NS(thinking_tokens=50)))
    provider, capture = adapter(provider_name, usage, incomplete=incomplete)
    model = "gpt-6-luna" if provider_name == "openai" else "claude-opus-5-5"
    prompt = "Bound ∑ α — 数学"
    expected = complete(provider_name, usage, model=model)
    kwargs = dict(run_id=None, workstream_id=workstream, provider=provider,
                  provider_name=provider_name, model=model, purpose="research:attack",
                  prompt=prompt, max_output_tokens=1000, estimated_max_cost_usd=.5)
    if incomplete:
        with pytest.raises(ModelOutputError, match="incomplete"):
            call_model(**kwargs)
    else:
        call_model(**kwargs)
    assert capture.calls == 1
    with connect() as con:
        row = con.execute("SELECT * FROM api_calls").fetchone()
    for key in CALL_TELEMETRY_COLUMNS:
        assert row[key] == (len(prompt.encode("utf-8")) if key == "prompt_utf8_bytes" else getattr(expected, key))
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 100
    assert row["cost_usd"] == expected.cost_usd == monthly_spend()
    assert row["status"] == ("failed" if incomplete else "completed")
    assert row["response_text"] == "response"


def test_no_usage_failure_preserves_unknown_details(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    def fail(**kwargs):
        raise TimeoutError("offline timeout")
    with pytest.raises(TheoryError, match="offline timeout"):
        call_model(run_id=None, provider=NS(complete=fail), provider_name="openai",
                   model="gpt-6-luna", purpose="test", prompt="α", max_output_tokens=100,
                   estimated_max_cost_usd=.1)
    with connect() as con:
        row = con.execute("SELECT * FROM api_calls").fetchone()
    assert row["status"] == "failed"
    assert row["prompt_utf8_bytes"] == 2
    assert all(row[key] is None for key in CALL_TELEMETRY_COLUMNS if key != "prompt_utf8_bytes")
    assert monthly_spend() == 0


def test_budget_admission_covers_cold_cache_writes(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    prompt = "α" * 100
    maximum = budget_guard(Config(), model="gpt-6-luna", prompt=prompt,
                           max_output_tokens=1000, purpose="test")
    assert maximum == estimate_usage_cost(
        "gpt-6-luna", uncached_input_tokens=0, cache_write_5m_input_tokens=200,
        output_tokens=1000,
    ).cost_usd
    cached = estimate_usage_cost("gpt-6-luna", uncached_input_tokens=0,
                                cache_read_input_tokens=200, output_tokens=1000)
    assert maximum > cached.cost_usd


def test_v7_migration_keeps_historical_receipt_details_unknown(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    # Reconstruct v7 DDL in a temporary test workspace, with an actual old receipt.
    old_schema = "\n".join(line for line in SCHEMA.splitlines()
                           if not any(line.strip().startswith(key + " ") for key in CALL_TELEMETRY_COLUMNS))
    with sqlite3.connect(tmp_path / ".theory" / "research.db") as con:
        con.executescript(old_schema)
        con.execute("INSERT INTO projects VALUES(1,'Legacy','','t')")
        con.execute("""INSERT INTO api_calls(provider,model,purpose,input_tokens,output_tokens,cost_usd,created_at)
                       VALUES('anthropic','claude-opus-5-5','old',100,50,1.25,?)""", (utcnow(),))
        original = con.execute("SELECT * FROM api_calls").fetchone()
        old_columns = [row[1] for row in con.execute("PRAGMA table_info(api_calls)")]
        con.execute("PRAGMA user_version=7")
    for _ in range(2):
        with connect() as con:
            assert con.execute("PRAGMA user_version").fetchone()[0] == 11
            assert con.execute("SELECT name FROM schema_migrations WHERE version=8").fetchone()[0] == "normalized_call_telemetry"
            migrated = con.execute("SELECT * FROM api_calls").fetchone()
        assert tuple(migrated[key] for key in old_columns) == original
        assert all(migrated[key] is None for key in CALL_TELEMETRY_COLUMNS)
    assert monthly_spend() == 1.25


def test_workstream_show_displays_details_only_when_known(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    provider, _ = adapter("openai", NS(input_tokens=6240, output_tokens=410,
        input_tokens_details=NS(cached_tokens=5120), output_tokens_details=NS(reasoning_tokens=180)))
    call_model(run_id=None, workstream_id=workstream, provider=provider, provider_name="openai",
               model="gpt-6-luna", purpose="research:develop", prompt="α",
               max_output_tokens=1000, estimated_max_cost_usd=.1)
    with connect() as con:
        con.execute("""INSERT INTO api_calls(workstream_id,provider,model,purpose,created_at)
                       VALUES(?,'openai','legacy','old',?)""", (workstream, utcnow()))
    result = CliRunner().invoke(app, ["workstream", "show", str(workstream)])
    assert result.exit_code == 0, result.output
    assert "6,240 = 1,120 uncached + 5,120 cache-read + 0 cache-write" in result.output
    assert "output 410 (reasoning 180)" in result.output
    assert "prompt: 2 bytes" in result.output
    assert "cache-read" not in result.output.split("model call: openai / legacy", 1)[1]
