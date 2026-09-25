import json

import pytest
from typer.testing import CliRunner

from theory.attack import ATTACK_MAX_OUTPUT_TOKENS, attack
from theory.cli import app
from theory.config import Config
from theory.db import connect, initialize
from theory.errors import BudgetExceededError, ModelOutputError, TheoryError
from theory.graph import (
    add_entity,
    add_relation,
    add_source,
    create_workstream,
    link_workstream_entity,
)
from theory.models import ModelResult


class FakeProvider:
    def __init__(self, response: dict | str):
        self.response = response if isinstance(response, str) else json.dumps(response)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return ModelResult(
            text=self.response,
            input_tokens=400,
            output_tokens=200,
            cost_usd=0.0056,
        )


class RaisingProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        raise TimeoutError("mock provider timeout")


def init_workspace(monkeypatch, tmp_path, budget=100.0):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    initialize("Attack tests")
    Config(monthly_budget_usd=budget).save()


def make_attack_workstream(*, workstream_type="attack", target_count=1):
    with connect() as con:
        targets = [
            add_entity(con, "Conjecture", f"Target conjecture {index + 1}")
            for index in range(target_count)
        ]
        workstream = create_workstream(con, workstream_type, "Try to break the target")
        for target in targets:
            link_workstream_entity(con, workstream, target, "input")
    return workstream, targets


def empty_report(target_id):
    return {
        "target_entity_id": target_id,
        "summary": "No concrete flaw was identified in this bounded pass.",
        "candidates": [],
        "could_not_determine": [],
    }


def test_attack_uses_only_graph_context_and_persists_quarantined_artifacts(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    with connect() as con:
        target = add_entity(con, "Conjecture", "Quadratic authenticated agreement")
        known = add_entity(con, "Theorem", "Known cubic lower bound")
        finding = add_entity(con, "Finding", "Known theorem assumes static faults")
        paper = add_entity(con, "Paper", "Primary lower-bound paper")
        source = add_source(con, finding, paper_entity_id=paper, page=7)
        add_relation(con, target, "EXTENDS", known)
        add_relation(con, finding, "SUPPORTS", target)
        unrelated = add_entity(con, "ResearchIdea", "UNRELATED PROJECT HISTORY SENTINEL")
        workstream = create_workstream(con, "attack", "Try to refute quadratic agreement")
        link_workstream_entity(con, workstream, target, "input")

    report = {
        "target_entity_id": target,
        "summary": "An adaptive schedule may invalidate the claimed message bound.",
        "candidates": [
            {
                "candidate_type": "counterexample",
                "statement": "Delay one honest process until the fast path commits.",
                "epistemic_status": "inference",
                "related_entity_ids": [target, known],
                "source_ids": [source],
                "reasoning_summary": "The target appears to reuse a static-fault step.",
            }
        ],
        "could_not_determine": ["Whether authentication prevents the delayed schedule."],
    }
    provider = FakeProvider(report)
    monkeypatch.setattr("theory.attack.get_provider", lambda _: provider)

    def network_must_not_run(*args, **kwargs):
        raise AssertionError("attack must not perform literature or network retrieval")

    monkeypatch.setattr("theory.openalex.search_works", network_must_not_run)
    monkeypatch.setattr("theory.workflows.search_works", network_must_not_run)

    outcome = attack(workstream, "openai")

    assert len(provider.calls) == 1
    prompt = provider.calls[0]["prompt"]
    assert "Quadratic authenticated agreement" in prompt
    assert "Known cubic lower bound" in prompt
    assert "UNRELATED PROJECT HISTORY SENTINEL" not in prompt
    assert "SOURCED / SOURCE-BACKED EVIDENCE" in prompt
    assert "QUARANTINED — DO NOT ASSUME TRUE" in prompt
    assert provider.calls[0]["max_output_tokens"] == ATTACK_MAX_OUTPUT_TOKENS

    with connect() as con:
        workstream_row = con.execute(
            "SELECT * FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()
        artifact = con.execute(
            "SELECT * FROM entities WHERE id=?", (outcome.artifact_ids[0],)
        ).fetchone()
        link = con.execute(
            """
            SELECT role FROM workstream_entities
            WHERE workstream_id=? AND entity_id=?
            """,
            (workstream, artifact["id"]),
        ).fetchone()
        source_link = con.execute(
            "SELECT source_id FROM entity_sources WHERE entity_id=?", (artifact["id"],)
        ).fetchone()
        review = con.execute(
            "SELECT * FROM reviews WHERE id=?", (outcome.review_id,)
        ).fetchone()
        calls = con.execute(
            "SELECT * FROM api_calls WHERE workstream_id=?", (workstream,)
        ).fetchall()

    assert unrelated != artifact["id"]
    assert artifact["entity_type"] == "Counterexample"
    assert artifact["trust_state"] == "quarantined"
    assert link["role"] == "created"
    assert source_link["source_id"] == source
    assert workstream_row["status"] == "completed"
    assert review["result"] == "issue_found"
    assert review["provider"] == "openai"
    assert len(calls) == 1
    assert calls[0]["purpose"] == "attack"
    assert calls[0]["status"] == "completed"
    assert calls[0]["run_id"] is None
    assert calls[0]["cost_usd"] == pytest.approx(0.0056)
    assert calls[0]["estimated_max_cost_usd"] > calls[0]["cost_usd"]


def test_attack_rejects_non_attack_workstream_before_provider(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_attack_workstream(workstream_type="proof")
    monkeypatch.setattr(
        "theory.attack.get_provider",
        lambda _: (_ for _ in ()).throw(AssertionError("provider must not be created")),
    )

    with pytest.raises(TheoryError, match="not an attack workstream"):
        attack(workstream, "openai")


def test_attack_rejects_missing_and_ambiguous_targets(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    with connect() as con:
        empty_workstream = create_workstream(con, "attack", "No target")
        paper = add_entity(con, "Paper", "Supporting input but not a primary target")
        ineligible_workstream = create_workstream(con, "attack", "No eligible target")
        link_workstream_entity(con, ineligible_workstream, paper, "input")
    ambiguous_workstream, targets = make_attack_workstream(target_count=2)
    assert len(targets) == 2
    monkeypatch.setattr(
        "theory.attack.get_provider",
        lambda _: (_ for _ in ()).throw(AssertionError("provider must not be created")),
    )

    with pytest.raises(TheoryError, match="has no input entity"):
        attack(empty_workstream, "openai")
    with pytest.raises(TheoryError, match="no eligible primary target"):
        attack(ineligible_workstream, "openai")
    with pytest.raises(TheoryError, match="ambiguous primary targets"):
        attack(ambiguous_workstream, "openai")


def test_attack_strict_schema_forbids_model_sourced_status(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, targets = make_attack_workstream()
    report = {
        "target_entity_id": targets[0],
        "summary": "Bad trust claim.",
        "candidates": [
            {
                "candidate_type": "obstruction",
                "statement": "Claimed obstruction",
                "epistemic_status": "sourced",
                "related_entity_ids": [targets[0]],
                "source_ids": [],
                "reasoning_summary": "The model tried to self-certify this.",
            }
        ],
        "could_not_determine": [],
    }
    provider = FakeProvider(report)
    monkeypatch.setattr("theory.attack.get_provider", lambda _: provider)

    with pytest.raises(ModelOutputError, match="did not match"):
        attack(workstream, "openai")

    with connect() as con:
        status = con.execute(
            "SELECT status FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0]
        created = con.execute(
            "SELECT COUNT(*) FROM workstream_entities WHERE workstream_id=? AND role='created'",
            (workstream,),
        ).fetchone()[0]
        call = con.execute(
            "SELECT status FROM api_calls WHERE workstream_id=?", (workstream,)
        ).fetchone()
    assert len(provider.calls) == 1
    assert status == "error"
    assert created == 0
    assert call["status"] == "completed"


@pytest.mark.parametrize("invalid_kind", ["entity", "source"])
def test_attack_rejects_out_of_context_references(
    monkeypatch, tmp_path, invalid_kind
):
    init_workspace(monkeypatch, tmp_path)
    workstream, targets = make_attack_workstream()
    candidate = {
        "candidate_type": "obstruction",
        "statement": "Candidate with an invented reference",
        "epistemic_status": "speculation",
        "related_entity_ids": [targets[0]],
        "source_ids": [],
        "reasoning_summary": "Reference validation should reject this.",
    }
    if invalid_kind == "entity":
        candidate["related_entity_ids"].append(999)
    else:
        candidate["source_ids"].append(999)
    provider = FakeProvider(
        {
            "target_entity_id": targets[0],
            "summary": "Invalid reference test.",
            "candidates": [candidate],
            "could_not_determine": [],
        }
    )
    monkeypatch.setattr("theory.attack.get_provider", lambda _: provider)

    with pytest.raises(ModelOutputError, match="unknown/out-of-context"):
        attack(workstream, "openai")

    with connect() as con:
        assert con.execute(
            "SELECT status FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0] == "error"


def test_provider_failure_is_logged_and_marks_execution_error(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_attack_workstream()
    provider = RaisingProvider()
    monkeypatch.setattr("theory.attack.get_provider", lambda _: provider)

    with pytest.raises(TheoryError, match="mock provider timeout"):
        attack(workstream, "anthropic")

    with connect() as con:
        workstream_row = con.execute(
            "SELECT * FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()
        call = con.execute(
            "SELECT * FROM api_calls WHERE workstream_id=?", (workstream,)
        ).fetchone()
    assert provider.calls == 1
    assert workstream_row["status"] == "error"
    assert "mock provider timeout" in workstream_row["summary"]
    assert call["status"] == "failed"
    assert "mock provider timeout" in call["error_message"]


def test_budget_guard_blocks_before_provider_and_keeps_workstream_active(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path, budget=0.0001)
    workstream, _ = make_attack_workstream()
    monkeypatch.setattr(
        "theory.attack.get_provider",
        lambda _: (_ for _ in ()).throw(AssertionError("provider must not be created")),
    )

    with pytest.raises(BudgetExceededError, match="cannot fit"):
        attack(workstream, "openai")

    with connect() as con:
        assert con.execute(
            "SELECT status FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0] == "active"
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 0


def test_attack_cli_displays_persisted_review_and_cost_without_second_call(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, targets = make_attack_workstream()
    provider = FakeProvider(empty_report(targets[0]))
    monkeypatch.setattr("theory.attack.get_provider", lambda _: provider)
    runner = CliRunner()

    result = runner.invoke(app, ["attack", str(workstream), "--provider", "openai"])

    assert result.exit_code == 0, result.output
    assert "Attack completed" in result.output
    assert "no_flaw_found" in result.output
    assert "model call: openai" in result.output
    assert "estimated cost $0.0056" in result.output
    assert len(provider.calls) == 1

    shown = runner.invoke(app, ["workstream", "show", str(workstream)])
    assert shown.exit_code == 0, shown.output
    assert "no_flaw_found" in shown.output
    assert "estimated cost $0.0056" in shown.output
    assert len(provider.calls) == 1
