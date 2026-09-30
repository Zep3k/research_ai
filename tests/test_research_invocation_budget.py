"""Every research request shares a fresh, local dollar allowance."""
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from theory.cli import app
from theory.config import Config
from theory.db import connect
from theory.errors import BudgetExceededError, ConfigurationError, ModelOutputError
from theory.model_calls import InvocationBudget, call_model
from theory.models import ModelResult
from theory.research import research
from test_research import init_workspace, make_research_workstream
from test_research_ideation import OfflineProvider, case02b, install
from test_research_reframe import wa
from test_research_strategy import StrategyProvider, historical, install_providers


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Real provider forbidden"))


def fixed_admission(monkeypatch, cost=0.25):
    monkeypatch.setattr("theory.model_calls.conservative_call_cost", lambda *args, **kwargs: cost)


def price_receipts(monkeypatch, provider_type, cost=0.125):
    complete = provider_type.complete

    def metered(self, **kwargs):
        result = complete(self, **kwargs)
        return result.model_copy(update={
            "cost_usd": cost, "output_cost_usd": cost, "uncached_input_cost_usd": 0.0,
            "cache_read_cost_usd": 0.0, "cache_write_cost_usd": 0.0,
        })

    monkeypatch.setattr(provider_type, "complete", metered)


def ledger():
    with connect() as con:
        return [dict(row) for row in con.execute("SELECT * FROM api_calls ORDER BY id")]


def assert_budget_stop(outcome, ws, *, executions=0, strategy=0, ideation=0, spend=0.0):
    assert outcome.stop_reason == "invocation_budget_exhausted"
    assert outcome.final_status == "completed"
    assert (outcome.calls_made, outcome.strategy_calls_made, outcome.ideation_calls_made) == (executions, strategy, ideation)
    assert outcome.invocation_spend_usd == pytest.approx(spend)
    assert outcome.total_api_calls_made == len(ledger())
    with connect() as con:
        row = con.execute("SELECT status,summary FROM workstreams WHERE id=?", (ws,)).fetchone()
        assert row["status"] == "completed"
        assert "invocation_budget_exhausted" in row["summary"]
        rows = [dict(row) for row in con.execute("SELECT * FROM research_iterations WHERE workstream_id=?", (ws,))]
    assert len(rows) == executions
    assert all(row["status"] == "completed" for row in rows)
    if rows:
        assert rows[-1]["stop_reason"] == "invocation_budget_exhausted"


def test_strategist_cost_blocks_anthropic_execution_before_any_receipt(historical, monkeypatch):
    ws, *_ = historical
    fixed_admission(monkeypatch)
    price_receipts(monkeypatch, StrategyProvider)
    requests, constructed = install_providers(monkeypatch)
    outcome = research(ws, max_calls=1, max_cost_usd=0.374)
    assert_budget_stop(outcome, ws, strategy=1, spend=0.125)
    assert outcome.invocation_budget_usd == 0.374
    assert [r["purpose"] for r in ledger()] == ["research:strategy"]
    assert len(requests) == 1 and constructed == ["openai"]


def test_anthropic_attack_cost_is_charged_to_same_allowance(historical, monkeypatch):
    ws, *_ = historical
    fixed_admission(monkeypatch)
    price_receipts(monkeypatch, StrategyProvider)
    requests, _ = install_providers(monkeypatch)
    outcome = research(ws, max_calls=1, max_cost_usd=0.375)
    assert outcome.stop_reason == "max_calls_exhausted"
    assert outcome.invocation_spend_usd == 0.25
    assert [(r["provider"], r["purpose"]) for r in ledger()] == [
        ("openai", "research:strategy"), ("anthropic", "research:attack"),
    ]
    assert len(requests) == 2


def test_strategist_is_refused_before_provider_setup(historical, monkeypatch):
    ws, *_ = historical
    fixed_admission(monkeypatch)
    outcome = research(ws, max_calls=1, max_cost_usd=0.249)
    assert_budget_stop(outcome, ws)
    assert ledger() == []


@pytest.mark.parametrize("cap,expected_purposes", [
    (0.249, []),
    (0.374, ["research:ideate"]),
    (0.499, ["research:ideate", "research:strategy"]),
    (0.624, ["research:ideate", "research:strategy", "research:develop"]),
])
def test_ideation_and_strategy_share_execution_budget(case02b, monkeypatch, cap, expected_purposes):
    wa, _, batch = case02b
    fixed_admission(monkeypatch)
    price_receipts(monkeypatch, OfflineProvider)
    provider = install(monkeypatch, batch)
    outcome = research(wa[0], max_calls=2, max_cost_usd=cap)
    assert [r["purpose"] for r in ledger()] == expected_purposes
    assert len(provider.requests) == len(expected_purposes)
    assert_budget_stop(
        outcome, wa[0], executions=expected_purposes.count("research:develop"),
        strategy=expected_purposes.count("research:strategy"), ideation=expected_purposes.count("research:ideate"),
        spend=len(expected_purposes) * 0.125,
    )
    assert outcome.invocation_budget_usd == cap


def test_max_calls_two_charges_all_four_requests_at_exact_admission_boundary(case02b, monkeypatch):
    wa, _, batch = case02b
    fixed_admission(monkeypatch)
    price_receipts(monkeypatch, OfflineProvider)
    provider = install(monkeypatch, batch)
    outcome = research(wa[0], max_calls=2, max_cost_usd=0.625)
    assert outcome.calls_made == 2 and outcome.total_api_calls_made == 4
    assert outcome.invocation_spend_usd == 0.5
    assert outcome.invocation_budget_usd == 0.625
    assert len(provider.requests) == 4
    assert [r["purpose"] for r in ledger()] == ["research:ideate", "research:strategy", "research:develop", "research:develop"]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_execution_spend_uses_actual_cost_not_accumulated_admission_reservations(monkeypatch, tmp_path, provider):
    init_workspace(monkeypatch, tmp_path)
    ws, _ = make_research_workstream()
    fixed_admission(monkeypatch)
    price_receipts(monkeypatch, StrategyProvider)
    requests, _ = install_providers(monkeypatch)
    outcome = research(ws, provider, max_calls=3, max_cost_usd=0.375)
    assert_budget_stop(outcome, ws, executions=2, spend=0.25)
    assert len(requests) == 2  # Reserving both estimates cumulatively would admit only one.
    assert all(r["estimated_max_cost_usd"] == 0.25 for r in ledger())


def test_monthly_budget_remains_an_independent_outer_guard(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, _ = make_research_workstream()
    Config(monthly_budget_usd=0.2, research_invocation_budget_usd=10).save()
    fixed_admission(monkeypatch)
    with pytest.raises(BudgetExceededError, match="monthly API budget"):
        research(ws, max_calls=1)
    assert ledger() == []
    with connect() as con:
        assert con.execute("SELECT status FROM workstreams WHERE id=?", (ws,)).fetchone()[0] == "active"


def test_resume_has_fresh_allowance_but_history_still_counts_monthly(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, _ = make_research_workstream()
    fixed_admission(monkeypatch)
    price_receipts(monkeypatch, StrategyProvider)
    requests, _ = install_providers(monkeypatch)
    for _ in range(2):
        with connect() as con:
            con.execute("UPDATE workstreams SET status='active' WHERE id=?", (ws,))
        outcome = research(ws, strategy="off", max_calls=1, max_cost_usd=0.25)
        assert outcome.invocation_spend_usd == 0.125
        assert outcome.stop_reason == "max_calls_exhausted"
    assert len(requests) == 2 and sum(r["cost_usd"] for r in ledger()) == 0.25
    Config(monthly_budget_usd=0.499).save()
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (ws,))
    with pytest.raises(BudgetExceededError, match="monthly API budget"):
        research(ws, strategy="off", max_calls=1, max_cost_usd=5)
    assert len(requests) == 2 and len(ledger()) == 2


@pytest.mark.parametrize("failure", ["incomplete", "validation"])
def test_returned_usage_on_failed_calls_is_charged(monkeypatch, tmp_path, failure):
    init_workspace(monkeypatch, tmp_path)
    budget = InvocationBudget(0.375)

    class FailedResponse:
        def complete(self, **kwargs):
            return ModelResult(text="malformed response", cost_usd=0.125, output_cost_usd=0.125,
                               response_status="incomplete" if failure == "incomplete" else "completed",
                               incomplete_reason="max_output_tokens" if failure == "incomplete" else None)

    def invalid(text):
        raise ModelOutputError("Invalid structured response")

    with pytest.raises(ModelOutputError):
        call_model(run_id=None, provider=FailedResponse(), provider_name="openai", model="gpt-6-sol",
                   purpose="research:strategy", prompt="Test", max_output_tokens=100,
                   estimated_max_cost_usd=0.25, validate_response=invalid, invocation_budget=budget)
    assert ledger()[0]["status"] == "failed" and ledger()[0]["cost_usd"] == 0.125
    assert budget.actual_spend_usd == 0.125
    assert budget.can_fit(0.25) and not budget.can_fit(0.250001)


def test_decimal_admission_never_exceeds_cap_and_allows_exact_boundary(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    budget = InvocationBudget(0.4)

    class Response:
        def __init__(self, cost):
            self.cost = cost

        def complete(self, **kwargs):
            return ModelResult(text="OK", cost_usd=self.cost, output_cost_usd=self.cost)

    for cost in (0.1, 0.2):
        call_model(run_id=None, provider=Response(cost), provider_name="openai", model="gpt-6-sol",
                   purpose="research:develop", prompt="Test", max_output_tokens=100,
                   estimated_max_cost_usd=cost, invocation_budget=budget)
    assert budget.can_fit(0.1)
    assert not budget.can_fit(0.1000001)


def test_cli_config_cap_and_explicit_override(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, _ = make_research_workstream()
    Config(research_invocation_budget_usd=0.249).save()
    fixed_admission(monkeypatch)
    price_receipts(monkeypatch, StrategyProvider)
    requests, _ = install_providers(monkeypatch)
    runner = CliRunner()
    refused = runner.invoke(app, ["research", str(ws), "--strategy", "off", "--max-calls", "1"])
    assert refused.exit_code == 0, refused.output
    assert "invocation_budget_exhausted" in refused.output
    assert "Invocation API spend: $0.0000 / $0.2490 cap." in refused.output
    assert not requests and not ledger()
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (ws,))
    accepted = runner.invoke(app, ["research", str(ws), "--strategy", "off", "--max-calls", "1", "--max-cost-usd", "0.25"])
    assert accepted.exit_code == 0, accepted.output
    assert "Invocation API spend: $0.1250 / $0.2500 cap." in accepted.output
    assert len(requests) == 1
    assert Config.load().research_invocation_budget_usd == 0.249


def test_legacy_config_receives_default_without_rewrite(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    path = tmp_path / ".theory" / "config.json"
    old = '{"monthly_budget_usd": 100}'
    path.write_text(old)
    assert Config.load().research_invocation_budget_usd == 1.50
    assert path.read_text() == old


@pytest.mark.parametrize("cap", [-1, float("nan"), float("inf")])
def test_invalid_caps_are_refused_before_calls(monkeypatch, tmp_path, cap):
    init_workspace(monkeypatch, tmp_path)
    ws, _ = make_research_workstream()
    with pytest.raises(ConfigurationError, match="finite and nonnegative"):
        research(ws, max_calls=1, max_cost_usd=cap)
    with pytest.raises(ValidationError):
        Config(research_invocation_budget_usd=cap)
    assert ledger() == []


def test_zero_cap_stops_cleanly(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, _ = make_research_workstream()
    outcome = research(ws, max_calls=1, max_cost_usd=0)
    assert_budget_stop(outcome, ws)
    assert outcome.invocation_budget_usd == 0
