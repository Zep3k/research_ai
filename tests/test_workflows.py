import json

import pytest

from theory.config import Config
from theory.db import connect, initialize, utcnow
from theory.errors import BudgetExceededError, ModelOutputError
from theory.models import LiteratureHit, ModelResult
from theory.workflows import investigate


FORMALIZATION = json.dumps(
    {
        "precise_question": "Is bound B necessary in model M?",
        "assumptions_to_pin_down": ["Synchrony"],
        "search_queries": ["distributed lower bound model M"],
        "possible_variants": [],
        "immediate_failure_modes": ["Known lower bound"],
    }
)

REPORT = json.dumps(
    {
        "precise_question": "Is bound B necessary in model M?",
        "nearest_results": [
            {
                "statement": "The retrieved abstract studies model M.",
                "epistemic_status": "sourced",
                "source_ids": ["S1"],
            }
        ],
        "reasons_to_continue": [],
        "reasons_to_stop_or_reframe": [],
        "hidden_assumptions": [],
        "counterexample_targets": [],
        "smallest_decisive_subproblems": [],
        "kill_conditions": [],
        "next_high_information_actions": [
            {
                "statement": "Read the primary source and extract the exact theorem.",
                "epistemic_status": "unresolved",
                "source_ids": ["S1"],
            }
        ],
        "epistemic_notes": [],
    }
)


class FakeProvider:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        text = next(self.texts)
        return ModelResult(text=text, input_tokens=100, output_tokens=50, cost_usd=0.0014)


def workspace(monkeypatch, tmp_path, budget=100.0):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    initialize("Test project")
    Config(monthly_budget_usd=budget).save()
    with connect() as con:
        con.execute(
            "INSERT INTO ideas(statement,notes,created_at) VALUES(?,?,?)",
            ("Can B be improved?", "", utcnow()),
        )


def test_investigation_persists_sources_report_and_each_call(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    provider = FakeProvider([FORMALIZATION, REPORT])
    monkeypatch.setattr("theory.workflows.get_provider", lambda _: provider)
    monkeypatch.setattr(
        "theory.workflows.search_works",
        lambda query, per_page: [
            LiteratureHit(
                openalex_id="https://openalex.org/W1",
                title="Relevant result",
                year=2025,
                doi="https://doi.org/10.1/example",
                abstract="We prove a bound in model M.",
            )
        ],
    )

    run_id = investigate(1, "openai")

    with connect() as con:
        run = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        calls = con.execute("SELECT * FROM api_calls WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
    literature = json.loads(run["literature_json"])
    report = json.loads(run["report_json"])
    assert run["status"] == "completed"
    assert literature["sources"][0]["source_id"] == "S1"
    assert literature["sources"][0]["retrieved_for_queries"] == [
        "distributed lower bound model M"
    ]
    assert report["nearest_results"][0]["source_ids"] == ["S1"]
    assert [call["purpose"] for call in calls] == ["formalize", "analyze"]
    assert all(call["status"] == "completed" for call in calls)
    assert calls[0]["response_text"] == FORMALIZATION
    assert calls[0]["estimated_max_cost_usd"] > calls[0]["cost_usd"]
    assert [call["max_output_tokens"] for call in provider.calls] == [8_000, 16_000]


def test_invalid_paid_output_is_logged_and_run_is_failed(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    provider = FakeProvider(["not json"])
    monkeypatch.setattr("theory.workflows.get_provider", lambda _: provider)

    with pytest.raises(ModelOutputError):
        investigate(1, "openai")

    with connect() as con:
        run = con.execute("SELECT * FROM runs").fetchone()
        calls = con.execute("SELECT * FROM api_calls").fetchall()
    assert run["status"] == "failed"
    assert "ModelOutputError" in run["error_message"]
    assert len(calls) == 1
    assert calls[0]["status"] == "completed"
    assert calls[0]["response_text"] == "not json"


def test_budget_blocks_call_before_provider_is_created(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path, budget=0.0001)

    def should_not_be_called(_):
        raise AssertionError("provider should not be constructed")

    monkeypatch.setattr("theory.workflows.get_provider", should_not_be_called)
    with pytest.raises(BudgetExceededError, match="cannot fit"):
        investigate(1, "openai")

    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_report_cannot_cite_unretrieved_source(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    bad_report = json.loads(REPORT)
    bad_report["nearest_results"][0]["source_ids"] = ["S999"]
    provider = FakeProvider([FORMALIZATION, json.dumps(bad_report)])
    monkeypatch.setattr("theory.workflows.get_provider", lambda _: provider)
    monkeypatch.setattr(
        "theory.workflows.search_works",
        lambda query, per_page: [
            LiteratureHit(openalex_id="https://openalex.org/W1", title="Relevant result")
        ],
    )

    with pytest.raises(ModelOutputError, match="not retrieved"):
        investigate(1, "openai")

    with connect() as con:
        run = con.execute("SELECT status FROM runs").fetchone()
        call_count = con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
    assert run["status"] == "failed"
    assert call_count == 2


def test_provider_failure_is_recorded_as_a_failed_call(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)

    class RaisingProvider:
        def complete(self, **kwargs):
            raise TimeoutError("provider timed out")

    monkeypatch.setattr("theory.workflows.get_provider", lambda _: RaisingProvider())
    with pytest.raises(Exception, match="timed out"):
        investigate(1, "openai")

    with connect() as con:
        run = con.execute("SELECT status FROM runs").fetchone()
        call = con.execute("SELECT * FROM api_calls").fetchone()
    assert run["status"] == "failed"
    assert call["status"] == "failed"
    assert call["input_tokens"] == 0
    assert "timed out" in call["error_message"]
