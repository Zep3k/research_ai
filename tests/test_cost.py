import pytest

from theory.db import connect, initialize, monthly_spend, utcnow
from theory.errors import ConfigurationError
from theory.graph import create_workstream
from theory.providers import conservative_call_cost, estimate_cost


def test_openai_cost():
    assert abs(estimate_cost("gpt-5.6-sol", 1_000_000, 1_000_000) - 24.0) < 1e-9


def test_anthropic_cost():
    assert abs(estimate_cost("claude-opus-5-5", 500_000, 100_000) - 4.0) < 1e-9


def test_unknown_model_is_not_assumed_free():
    with pytest.raises(ConfigurationError, match="No trusted pricing"):
        estimate_cost("unpriced-model", 100, 100)


def test_budget_admission_estimate_is_conservative():
    actual_for_tiny_prompt = estimate_cost("gpt-5.6-sol", 2, 1_000)
    admitted_max = conservative_call_cost("gpt-5.6-sol", "hello", 1_000)
    assert admitted_max >= actual_for_tiny_prompt


def test_cost_ledger_can_attribute_calls_to_workstreams(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    initialize("Cost tests")
    with connect() as con:
        workstream = create_workstream(con, "proof", "Check the main lemma")
        con.execute(
            """
            INSERT INTO api_calls(
                workstream_id,provider,model,purpose,input_tokens,output_tokens,
                cost_usd,estimated_max_cost_usd,status,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                workstream,
                "openai",
                "gpt-5.6-sol",
                "proof_critique",
                100,
                50,
                0.002,
                0.05,
                "completed",
                utcnow(),
            ),
        )

    assert monthly_spend() == pytest.approx(0.002)
    with connect() as con:
        row = con.execute("SELECT * FROM api_calls").fetchone()
    assert row["workstream_id"] == workstream
    assert row["purpose"] == "proof_critique"
