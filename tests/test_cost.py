import pytest

from theory.errors import ConfigurationError
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
