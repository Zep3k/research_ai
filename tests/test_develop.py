import json

import pytest
from typer.testing import CliRunner

from theory.cli import app
from theory.config import Config
from theory.db import connect, initialize
from theory.develop import DEVELOP_MAX_OUTPUT_TOKENS, develop
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
            input_tokens=600,
            output_tokens=400,
            cost_usd=0.0104,
        )


class RaisingProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        raise TimeoutError("mock develop timeout")


def init_workspace(monkeypatch, tmp_path, budget=100.0):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    initialize("Develop tests")
    Config(monthly_budget_usd=budget).save()


def make_develop_workstream(*, workstream_type="develop", target_count=1):
    with connect() as con:
        targets = [
            add_entity(con, "Conjecture", f"Promising target {index + 1}")
            for index in range(target_count)
        ]
        workstream = create_workstream(
            con, workstream_type, "Develop the promising construction"
        )
        for target in targets:
            link_workstream_entity(con, workstream, target, "input")
    return workstream, targets


def valid_report(target_id, *, related_id=None, source_id=None):
    related = [target_id] if related_id is None else [target_id, related_id]
    sources = [] if source_id is None else [source_id]
    return {
        "target_entity_id": target_id,
        "summary": "A threshold-counting route and a certificate route merit separate checks.",
        "developments": [
            {
                "item_type": "consequence",
                "statement": "Every successful execution must expose a quorum intersection.",
                "epistemic_status": "inference",
                "related_entity_ids": related,
                "source_ids": sources,
                "reasoning_summary": "Agreement forces two decision certificates to overlap.",
            },
            {
                "item_type": "intermediate_lemma",
                "statement": "Two certificates of size n-t share an honest process.",
                "epistemic_status": "speculation",
                "related_entity_ids": [target_id],
                "source_ids": [],
                "reasoning_summary": "Prove the intersection bound before using it in safety.",
            },
            {
                "item_type": "parameter_analysis",
                "statement": "The overlap lower bound is n-2t and must exceed t.",
                "epistemic_status": "inference",
                "related_entity_ids": [target_id],
                "source_ids": [],
                "reasoning_summary": "The condition n-2t>t recovers n>3t.",
            },
            {
                "item_type": "proof_obligation",
                "statement": "Show that every decision carries an n-t certificate.",
                "epistemic_status": "unresolved",
                "related_entity_ids": [target_id],
                "source_ids": [],
                "reasoning_summary": "The counting argument is unusable without this invariant.",
            },
        ],
        "branches": [
            {
                "name": "Threshold certificates",
                "status": "promising",
                "objective": "Build safety from explicit quorum certificates.",
                "technical_plan": "Define certificate formation, then prove honest overlap.",
                "proof_obligations": ["Prove certificate availability under t faults."],
                "epistemic_status": "speculation",
                "related_entity_ids": [target_id],
                "source_ids": [],
            },
            {
                "name": "Transcript compression",
                "status": "failed",
                "objective": "Compress all-to-all evidence into one relayed digest.",
                "technical_plan": "This branch fails because the digest omits sender identity.",
                "proof_obligations": ["Recover sender attribution without cubic messages."],
                "epistemic_status": "inference",
                "related_entity_ids": [target_id],
                "source_ids": [],
            },
        ],
        "could_not_determine": ["Whether authentication repairs sender attribution."],
    }


def test_develop_uses_graph_context_and_persists_quarantined_typed_artifacts(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    with connect() as con:
        target = add_entity(con, "Conjecture", "Quadratic authenticated agreement")
        known = add_entity(con, "Theorem", "Known quorum-intersection theorem")
        prior_failure = add_entity(
            con, "FailedApproach", "Failed compression branch: identities were lost"
        )
        paper = add_entity(con, "Paper", "Certificate lower-bound paper")
        source = add_source(con, known, paper_entity_id=paper, page=11)
        add_relation(con, target, "EXTENDS", known)
        add_relation(con, prior_failure, "FAILS_AT", target)
        unrelated = add_entity(con, "ResearchIdea", "UNRELATED HISTORY SENTINEL")
        workstream = create_workstream(
            con, "develop", "Advance the quadratic construction"
        )
        link_workstream_entity(con, workstream, target, "input")

    provider = FakeProvider(valid_report(target, related_id=known, source_id=source))
    monkeypatch.setattr("theory.develop.get_provider", lambda _: provider)

    def network_must_not_run(*args, **kwargs):
        raise AssertionError("develop must not perform literature or network retrieval")

    monkeypatch.setattr("theory.openalex.search_works", network_must_not_run)
    monkeypatch.setattr("theory.workflows.search_works", network_must_not_run)

    outcome = develop(workstream, "openai")

    assert len(provider.calls) == 1
    prompt = provider.calls[0]["prompt"]
    assert "Quadratic authenticated agreement" in prompt
    assert "Known quorum-intersection theorem" in prompt
    assert "Failed compression branch" in prompt
    assert "UNRELATED HISTORY SENTINEL" not in prompt
    assert "SOURCED / SOURCE-BACKED EVIDENCE" in prompt
    assert "QUARANTINED — DO NOT ASSUME TRUE" in prompt
    assert provider.calls[0]["max_output_tokens"] == DEVELOP_MAX_OUTPUT_TOKENS

    with connect() as con:
        workstream_row = con.execute(
            "SELECT * FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()
        development_types = [
            con.execute("SELECT entity_type FROM entities WHERE id=?", (entity_id,)).fetchone()[0]
            for entity_id in outcome.development_artifact_ids
        ]
        branch_rows = [
            con.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
            for entity_id in outcome.branch_artifact_ids
        ]
        generated_rows = con.execute(
            """
            SELECT e.* FROM workstream_entities we
            JOIN entities e ON e.id=we.entity_id
            WHERE we.workstream_id=? AND we.role='created' ORDER BY e.id
            """,
            (workstream,),
        ).fetchall()
        source_link = con.execute(
            "SELECT source_id FROM entity_sources WHERE entity_id=?",
            (outcome.development_artifact_ids[0],),
        ).fetchone()
        failed_status = con.execute(
            "SELECT value FROM entity_attributes WHERE entity_id=? AND key='develop_branch_status'",
            (outcome.branch_artifact_ids[1],),
        ).fetchone()[0]
        calls = con.execute(
            "SELECT * FROM api_calls WHERE workstream_id=?", (workstream,)
        ).fetchall()
        review_count = con.execute(
            "SELECT COUNT(*) FROM reviews WHERE workstream_id=?", (workstream,)
        ).fetchone()[0]

    assert unrelated not in outcome.artifact_ids
    assert development_types == ["Finding", "Lemma", "Finding", "OpenQuestion"]
    assert [row["entity_type"] for row in branch_rows] == ["Technique", "FailedApproach"]
    assert all(row["trust_state"] == "quarantined" for row in generated_rows)
    assert len(generated_rows) == 6
    assert source_link["source_id"] == source
    assert failed_status == "failed"
    assert workstream_row["status"] == "completed"
    assert "failed=1" in workstream_row["summary"]
    assert review_count == 0
    assert len(calls) == 1
    assert calls[0]["purpose"] == "develop"
    assert calls[0]["status"] == "completed"
    assert calls[0]["run_id"] is None
    assert calls[0]["cost_usd"] == pytest.approx(0.0104)
    assert calls[0]["estimated_max_cost_usd"] > calls[0]["cost_usd"]


def test_develop_rejects_wrong_workstream_and_missing_or_ambiguous_targets(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    wrong_workstream, _ = make_develop_workstream(workstream_type="attack")
    ambiguous_workstream, targets = make_develop_workstream(target_count=2)
    assert len(targets) == 2
    with connect() as con:
        empty_workstream = create_workstream(con, "develop", "No target")
        paper = add_entity(con, "Paper", "Supporting paper only")
        ineligible_workstream = create_workstream(con, "develop", "No eligible target")
        link_workstream_entity(con, ineligible_workstream, paper, "input")
    monkeypatch.setattr(
        "theory.develop.get_provider",
        lambda _: (_ for _ in ()).throw(AssertionError("provider must not be created")),
    )

    with pytest.raises(TheoryError, match="not a develop workstream"):
        develop(wrong_workstream, "openai")
    with pytest.raises(TheoryError, match="has no input entity"):
        develop(empty_workstream, "openai")
    with pytest.raises(TheoryError, match="no eligible primary target"):
        develop(ineligible_workstream, "openai")
    with pytest.raises(TheoryError, match="ambiguous primary targets"):
        develop(ambiguous_workstream, "openai")


def test_develop_strict_schema_forbids_sourced_output_and_requires_coverage(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, targets = make_develop_workstream()
    report = valid_report(targets[0])
    report["developments"][0]["epistemic_status"] = "sourced"
    provider = FakeProvider(report)
    monkeypatch.setattr("theory.develop.get_provider", lambda _: provider)

    with pytest.raises(ModelOutputError, match="did not match"):
        develop(workstream, "openai")

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
        ).fetchone()[0]
    assert len(provider.calls) == 1
    assert status == "error"
    assert created == 0
    assert call == "completed"


@pytest.mark.parametrize("invalid_kind", ["entity", "source"])
def test_develop_rejects_out_of_context_references(monkeypatch, tmp_path, invalid_kind):
    init_workspace(monkeypatch, tmp_path)
    workstream, targets = make_develop_workstream()
    report = valid_report(targets[0])
    if invalid_kind == "entity":
        report["developments"][0]["related_entity_ids"].append(999)
    else:
        report["branches"][0]["source_ids"].append(999)
    provider = FakeProvider(report)
    monkeypatch.setattr("theory.develop.get_provider", lambda _: provider)

    with pytest.raises(ModelOutputError, match="unknown/out-of-context"):
        develop(workstream, "openai")

    with connect() as con:
        assert con.execute(
            "SELECT status FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0] == "error"


def test_develop_provider_failure_is_logged_and_marks_execution_error(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_develop_workstream()
    provider = RaisingProvider()
    monkeypatch.setattr("theory.develop.get_provider", lambda _: provider)

    with pytest.raises(TheoryError, match="mock develop timeout"):
        develop(workstream, "anthropic")

    with connect() as con:
        workstream_row = con.execute(
            "SELECT * FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()
        call = con.execute(
            "SELECT * FROM api_calls WHERE workstream_id=?", (workstream,)
        ).fetchone()
    assert provider.calls == 1
    assert workstream_row["status"] == "error"
    assert "mock develop timeout" in workstream_row["summary"]
    assert call["status"] == "failed"
    assert "mock develop timeout" in call["error_message"]


def test_develop_budget_guard_blocks_before_provider_and_keeps_workstream_active(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path, budget=0.0001)
    workstream, _ = make_develop_workstream()
    monkeypatch.setattr(
        "theory.develop.get_provider",
        lambda _: (_ for _ in ()).throw(AssertionError("provider must not be created")),
    )

    with pytest.raises(BudgetExceededError, match="cannot fit"):
        develop(workstream, "openai")

    with connect() as con:
        assert con.execute(
            "SELECT status FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0] == "active"
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 0


def test_develop_cli_displays_persisted_artifacts_and_cost_without_second_call(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, targets = make_develop_workstream()
    provider = FakeProvider(valid_report(targets[0]))
    monkeypatch.setattr("theory.develop.get_provider", lambda _: provider)
    runner = CliRunner()

    result = runner.invoke(app, ["develop", str(workstream), "--provider", "openai"])

    assert result.exit_code == 0, result.output
    assert "Development completed" in result.output
    assert "4 technical artifact(s), 2 branch artifact(s)" in result.output
    assert "FailedApproach" in result.output
    assert "quarantined" in result.output
    assert "model call: openai" in result.output
    assert "$0.0104" in result.output
    assert len(provider.calls) == 1

    shown = runner.invoke(app, ["workstream", "show", str(workstream)])
    assert shown.exit_code == 0, shown.output
    assert "FailedApproach" in shown.output
    assert "$0.0104" in shown.output
    assert len(provider.calls) == 1
