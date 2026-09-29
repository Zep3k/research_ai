"""Offline SDK transport, cache boundaries, admission, and receipt regressions."""
from dataclasses import FrozenInstanceError
import json

import anthropic
import httpx
import httpx2
import openai
import pytest

from theory.config import Config
from theory.db import CALL_TELEMETRY_COLUMNS, connect, monthly_spend
from theory.errors import BudgetExceededError
from theory.model_calls import budget_guard, call_model
from theory.prompts import PromptContent
from theory.providers import (
    AnthropicProvider, OpenAIProvider, conservative_call_cost, estimate_usage_cost,
    get_model_spec,
)
from theory.research import ResearchStepReport, research
from theory.research_prompts import build_research_sections, build_strategist_sections
from test_research import (
    add_linked_research_entity, artifact, decision_from_prompt, init_workspace,
    make_research_workstream, step_report,
)
from test_research_prompts import CASES, inputs, strategist_state


def sdk_provider(name, requests, responder=lambda body: "answer", *, hit=False):
    transport_module = httpx if name == "openai" else httpx2

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        text = responder(body)
        if name == "openai":
            response = {
                "id": "resp_offline", "object": "response", "created_at": 1,
                "status": "completed", "model": body["model"],
                "output": [{"id": "msg_offline", "type": "message", "role": "assistant",
                            "status": "completed", "content": [
                                {"type": "output_text", "text": text, "annotations": []}]}],
                "usage": {"input_tokens": 2400, "output_tokens": 100, "total_tokens": 2500,
                          "input_tokens_details": {"cached_tokens": 2000 if hit else 0,
                                                   "cache_write_tokens": 0 if hit else 2000},
                          "output_tokens_details": {"reasoning_tokens": 20}},
            }
        else:
            response = {
                "id": "msg_offline", "type": "message", "role": "assistant", "model": body["model"],
                "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 400, "output_tokens": 100,
                          "cache_read_input_tokens": 2000 if hit else 0,
                          "cache_creation_input_tokens": 0 if hit else 2000,
                          "cache_creation": {"ephemeral_5m_input_tokens": 0 if hit else 2000,
                                             "ephemeral_1h_input_tokens": 0}},
            }
        return transport_module.Response(200, json=response)

    client = transport_module.Client(transport=transport_module.MockTransport(handle))
    if name == "openai":
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider.client = openai.OpenAI(api_key="offline-test", http_client=client, max_retries=0)
    else:
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.client = anthropic.Anthropic(api_key="offline-test", http_client=client, max_retries=0)
    return provider


def sections(case, offset=0):
    context, primary, choice, facts = inputs(case, offset)
    return build_research_sections(context, primary, choice, facts=facts)


@pytest.mark.parametrize("case", (*CASES, "strategist"))
def test_structured_prompt_retains_exact_bytes_and_stable_boundary(case):
    first, second = (
        (build_strategist_sections(strategist_state()), build_strategist_sections(strategist_state(1000)))
        if case == "strategist" else (sections(case), sections(case, 1000))
    )
    content = first.as_prompt_content()
    assert content.render().encode() == first.render().encode()
    assert content.stable_prefix.encode() == second.as_prompt_content().stable_prefix.encode()
    assert content.dynamic_suffix != second.as_prompt_content().dynamic_suffix
    assert "CONTROLLER DECISION" not in content.stable_prefix
    assert "RESEARCH STATE\n" not in content.stable_prefix
    with pytest.raises(FrozenInstanceError):
        content.stable_prefix = "different"


@pytest.mark.parametrize("name,model", [("openai", "gpt-6-sol"), ("anthropic", "claude-sonnet-5")])
@pytest.mark.parametrize("case", (*CASES, "strategist"))
def test_sdk_request_shapes_and_cache_boundaries(name, model, case):
    requests = []
    provider = sdk_provider(name, requests)
    for offset in (0, 1000):
        selected = (build_strategist_sections(strategist_state(offset))
                    if case == "strategist" else sections(case, offset))
        content = selected.as_prompt_content()
        provider.complete(model=model, prompt=content, max_output_tokens=12000,
                          effort="medium", response_model=ResearchStepReport)
        body = requests[-1]
        assert not any(key.startswith("prompt_cache") or key == "cache_control" for key in body)
        if name == "openai":
            # Same request shape and single user-message role as legacy strings.
            assert body["input"] == content.render()
            assert body["reasoning"] == {"effort": "medium"}
            assert body["max_output_tokens"] == 12000
        else:
            assert "system" not in body
            assert len(body["messages"]) == 1
            assert body["messages"][0]["role"] == "user"
            blocks = body["messages"][0]["content"]
            assert blocks == [
                {"type": "text", "text": content.stable_prefix,
                 "cache_control": {"type": "ephemeral", "ttl": "5m"}},
                {"type": "text", "text": content.dynamic_suffix},
            ]
            assert "".join(block["text"] for block in blocks) == selected.render()
            assert body["output_config"]["effort"] == "medium"
            assert body["max_tokens"] == 12000
    assert len(requests) == 2  # No prewarming, token-counting, or follow-up calls.
    if name == "openai":
        assert requests[0]["text"] == requests[1]["text"]
    else:
        assert requests[0]["messages"][0]["content"][0] == requests[1]["messages"][0]["content"][0]
        assert requests[0]["output_config"] == requests[1]["output_config"]
    # Legacy string compatibility, including identical schemas.
    provider.complete(model=model, prompt="legacy", max_output_tokens=12000,
                      effort="medium", response_model=ResearchStepReport)
    key = "text" if name == "openai" else "output_config"
    assert requests[0][key] == requests[2][key]
    if name == "anthropic":
        assert requests[2]["messages"] == [{"role": "user", "content": "legacy"}]


@pytest.mark.parametrize("content", [PromptContent("", "dynamic"), PromptContent("stable", "")])
def test_anthropic_empty_sections_do_not_emit_empty_cache_blocks(content):
    requests = []
    provider = sdk_provider("anthropic", requests)
    provider.complete(model="claude-sonnet-5", prompt=content, max_output_tokens=100)
    blocks = requests[0]["messages"][0]["content"]
    if not content.stable_prefix:
        assert blocks == "dynamic"
    else:
        assert blocks == [{"type": "text", "text": "stable",
                           "cache_control": {"type": "ephemeral", "ttl": "5m"}}]


@pytest.mark.parametrize("name,model", [("openai", "gpt-6-sol"), ("anthropic", "claude-sonnet-5")])
@pytest.mark.parametrize("hit", [False, True])
def test_sdk_cache_usage_persists_with_actual_pricing(monkeypatch, tmp_path, name, model, hit):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    requests = []
    provider = sdk_provider(name, requests, hit=hit)
    content = sections("reframe").as_prompt_content()
    bound = budget_guard(Config(), model=model, prompt=content, max_output_tokens=1000,
                         purpose="research:reframe", response_model=ResearchStepReport)
    result = call_model(run_id=None, workstream_id=workstream, provider=provider,
                        provider_name=name, model=model, purpose="research:reframe",
                        prompt=content, max_output_tokens=1000, estimated_max_cost_usd=bound,
                        response_model=ResearchStepReport)
    expected = estimate_usage_cost(model, uncached_input_tokens=400, output_tokens=100,
                                  cache_read_input_tokens=2000 if hit else 0,
                                  cache_write_5m_input_tokens=0 if hit else 2000)
    assert result.input_tokens == 2400
    assert result.cache_read_input_tokens == (2000 if hit else 0)
    assert result.cache_write_input_tokens == (0 if hit else 2000)
    assert result.cache_write_1h_input_tokens == 0
    assert result.cost_usd == pytest.approx(expected.cost_usd)
    assert bound > result.cost_usd
    with connect() as con:
        rows = con.execute("SELECT * FROM api_calls").fetchall()
    assert len(rows) == len(requests) == 1
    row = rows[0]
    assert row["workstream_id"] == workstream
    assert row["purpose"] == "research:reframe"
    for key in CALL_TELEMETRY_COLUMNS:
        assert row[key] == (len(content.render().encode()) if key == "prompt_utf8_bytes"
                            else getattr(result, key))
    assert monthly_spend() == pytest.approx(expected.cost_usd)


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6-sol", "claude-sonnet-5", "claude-opus-5-5"])
def test_admission_covers_unknown_hits_writes_schema_and_output_cap(monkeypatch, tmp_path, model):
    init_workspace(monkeypatch, tmp_path)
    prompt = PromptContent("α" * 2000, "dynamic suffix")
    rate = get_model_spec(model)
    bound = conservative_call_cost(model, prompt, 1000, response_model=ResearchStepReport)
    text_only_cold_cost = estimate_usage_cost(
        model, uncached_input_tokens=0, cache_write_5m_input_tokens=len(prompt.render().encode()),
        output_tokens=1000,
    ).cost_usd
    assert bound > text_only_cold_cost
    assert bound > conservative_call_cost(model, prompt, 1000)
    assert conservative_call_cost(model, prompt, 1100, response_model=ResearchStepReport) - bound == pytest.approx(
        100 * rate.output_usd_per_million / 1e6)
    # A budget that covers ordinary input but not a cold cache must be rejected.
    with pytest.raises(BudgetExceededError):
        budget_guard(Config(monthly_budget_usd=text_only_cold_cost), model=model, prompt=prompt,
                     max_output_tokens=1000, purpose="research:develop", response_model=ResearchStepReport)
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 0


def test_research_uses_structured_prompts_without_extra_calls(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    Config(research_frontier_develop_model="claude-sonnet-5").save()
    add_linked_research_entity(workstream, "OpenQuestion", "Unresolved local route lemma",
                               proof_obligation=True)
    requests = {"openai": [], "anthropic": []}

    def strategy(body):
        state = json.loads(body["input"].split("RESEARCH STATE\n", 1)[1])
        move = next(move for move in state["legal_moves"]
                    if move["operation"] == "develop"
                    and move["target_entity_id"] == state["primary_target"]["id"])
        return json.dumps({"selected_move_id": move["move_id"],
                           "rationale": "Continue the legal development move."})

    def execution(body):
        prompt = "".join(block["text"] for block in body["messages"][0]["content"])
        decision = decision_from_prompt(prompt)
        number = len(requests["anthropic"])
        return json.dumps(step_report(decision, [artifact(
            "finding", f"Distinct result {number}", f"result_{number}", [decision["target_entity_id"]],
        )]))

    providers = {"openai": sdk_provider("openai", requests["openai"], strategy),
                 "anthropic": sdk_provider("anthropic", requests["anthropic"], execution)}
    monkeypatch.setattr("theory.research.get_provider", providers.__getitem__)
    outcome = research(workstream, max_calls=2, strategy="auto")
    assert outcome.calls_made == outcome.strategy_calls_made == 2
    assert outcome.total_api_calls_made == 4
    assert len(requests["openai"]) == len(requests["anthropic"]) == 2
    first, second = [body["messages"][0]["content"] for body in requests["anthropic"]]
    assert first[0] == second[0]
    assert first[1] != second[1]
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM api_calls WHERE status='completed'").fetchone()[0] == 4
        assert con.execute("SELECT COUNT(*) FROM research_iterations WHERE status='completed'").fetchone()[0] == 2
