import json

import pytest
from typer.testing import CliRunner

from theory.cli import app
from theory.config import Config
from theory.db import connect, initialize
from theory.errors import BudgetExceededError, ModelOutputError, TheoryError
from theory.graph import (
    add_entity,
    create_workstream,
    link_workstream_entity,
    set_attribute,
)
from theory.models import ModelResult
from theory.research import research


def decision_from_prompt(prompt: str) -> dict:
    payload = prompt.split("CONTROLLER DECISION\n", 1)[1].split(
        "\n\nGRAPH CONTEXT", 1
    )[0]
    return json.loads(payload)


def artifact(
    artifact_type: str,
    statement: str,
    material_key: str,
    related_entity_ids: list[int],
    *,
    branch_status=None,
    epistemic_status="speculation",
):
    return {
        "artifact_type": artifact_type,
        "statement": statement,
        "reasoning_summary": f"Technical reasoning for {material_key}.",
        "material_key": material_key,
        "epistemic_status": epistemic_status,
        "related_entity_ids": related_entity_ids,
        "source_ids": [],
        "branch_status": branch_status,
    }


def step_report(
    decision: dict,
    artifacts: list[dict],
    *,
    addressed=None,
    attack_outcome="not_applicable",
    unresolved=None,
    human=False,
    human_reason=None,
):
    return {
        "operation": decision["operation"],
        "target_entity_id": decision["target_entity_id"],
        "summary": f"Completed bounded {decision['operation']} operation.",
        "artifacts": artifacts,
        "consumed_entity_ids": decision["required_consumed_entity_ids"],
        "addressed_obligation_ids": addressed or [],
        "attack_outcome": attack_outcome,
        "could_not_determine": unresolved or [],
        "human_judgment_required": human,
        "human_judgment_reason": human_reason,
    }


class DynamicProvider:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responder(decision_from_prompt(kwargs["prompt"]), len(self.calls))
        return ModelResult(
            text=json.dumps(response),
            input_tokens=500,
            output_tokens=250,
            cost_usd=0.007,
        )


class RaisingProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        raise TimeoutError("mock controller timeout")


def init_workspace(monkeypatch, tmp_path, *, budget=100.0):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    initialize("Research controller tests")
    Config(monthly_budget_usd=budget).save()


def make_research_workstream(entity_type="ResearchIdea", title="Promising direction"):
    with connect() as con:
        target = add_entity(con, entity_type, title)
        workstream = create_workstream(con, "research", "Advance and test the direction")
        link_workstream_entity(con, workstream, target, "input")
    return workstream, target


def test_controller_adapts_develop_synthesize_attack_and_stops_successfully(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    with connect() as con:
        add_entity(con, "ResearchIdea", "UNRELATED PROJECT HISTORY SENTINEL")

    def responder(decision, call_number):
        selected = decision["target_entity_id"]
        if call_number == 1:
            assert decision["operation"] == "develop"
            return step_report(
                decision,
                [
                    artifact(
                        "lemma",
                        "Every decision certificate contains an honest signer.",
                        "honest_signer_lemma",
                        [selected],
                        epistemic_status="inference",
                    ),
                    artifact(
                        "protocol_component",
                        "Aggregate threshold certificates before the decision step.",
                        "threshold_certificate_branch",
                        [selected],
                        branch_status="promising",
                    ),
                    artifact(
                        "proof_obligation",
                        "Prove certificate availability despite t silent processes.",
                        "certificate_availability_obligation",
                        [selected],
                        epistemic_status="unresolved",
                    ),
                    artifact(
                        "failed_approach",
                        "Digest-only compression loses signer attribution.",
                        "digest_only_failed_branch",
                        [selected],
                        branch_status="failed",
                        epistemic_status="inference",
                    ),
                ],
            )
        if call_number == 2:
            assert decision["operation"] == "synthesize"
            consumed = decision["required_consumed_entity_ids"]
            assert len(consumed) >= 2
            related = [selected, *consumed]
            return step_report(
                decision,
                [
                    artifact(
                        "proof_attempt",
                        "Combine honest overlap with threshold aggregation to construct a certificate.",
                        "certificate_availability_proof_attempt",
                        related,
                        epistemic_status="inference",
                    )
                ],
                addressed=[selected],
            )
        assert call_number == 3
        assert decision["operation"] == "attack"
        return step_report(
            decision,
            [],
            attack_outcome="no_critical_issue",
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    def network_must_not_run(*args, **kwargs):
        raise AssertionError("research controller must not retrieve from the network")

    monkeypatch.setattr("theory.openalex.search_works", network_must_not_run)
    monkeypatch.setattr("theory.workflows.search_works", network_must_not_run)

    outcome = research(workstream, "openai", max_calls=6)

    assert outcome.calls_made == 3
    assert outcome.stop_reason == "candidate_survived_attack"
    assert outcome.final_status == "completed"
    assert len(provider.calls) == 3
    assert all(
        "UNRELATED PROJECT HISTORY SENTINEL" not in call["prompt"]
        for call in provider.calls
    )

    with connect() as con:
        iterations = con.execute(
            "SELECT * FROM research_iterations WHERE workstream_id=? ORDER BY iteration_number",
            (workstream,),
        ).fetchall()
        generated = con.execute(
            """
            SELECT e.* FROM workstream_entities we
            JOIN entities e ON e.id=we.entity_id
            WHERE we.workstream_id=? AND we.role='created' ORDER BY e.id
            """,
            (workstream,),
        ).fetchall()
        calls = con.execute(
            "SELECT * FROM api_calls WHERE workstream_id=? ORDER BY id", (workstream,)
        ).fetchall()
        attempts = con.execute(
            "SELECT * FROM relations WHERE relation_type='ATTEMPTS'"
        ).fetchall()
        review = con.execute(
            "SELECT * FROM reviews WHERE workstream_id=?", (workstream,)
        ).fetchone()
        status = con.execute(
            "SELECT status FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0]

    assert [row["operation"] for row in iterations] == [
        "develop",
        "synthesize",
        "attack",
    ]
    assert all(row["rationale"].strip() for row in iterations)
    assert all(row["status"] == "completed" for row in iterations)
    assert iterations[-1]["stop_reason"] == "candidate_survived_attack"
    assert [row["purpose"] for row in calls] == [
        "research:develop",
        "research:synthesize",
        "research:attack",
    ]
    assert all(row["status"] == "completed" for row in calls)
    assert all(row["trust_state"] == "quarantined" for row in generated)
    assert {row["entity_type"] for row in generated} >= {
        "Lemma",
        "Technique",
        "OpenQuestion",
        "FailedApproach",
        "ProofAttempt",
    }
    assert len(attempts) == 1
    assert attempts[0]["trust_state"] == "quarantined"
    assert review["result"] == "no_flaw_found"
    assert status == "completed"


def test_prove_is_selected_only_for_a_precise_candidate(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream("Conjecture", "Precise quorum claim")
    with connect() as con:
        set_attribute(con, target, "precise_candidate", "true")
        obligation = add_entity(con, "OpenQuestion", "Establish the quorum inequality")
        set_attribute(con, obligation, "is_proof_obligation", "true")
        link_workstream_entity(con, workstream, obligation, "created")

    def responder(decision, _):
        assert decision["operation"] == "prove"
        assert decision["target_entity_id"] == target
        return step_report(
            decision,
            [
                artifact(
                    "proof_attempt",
                    "For quorums Q1 and Q2, inclusion-exclusion gives the required overlap.",
                    "quorum_overlap_proof_attempt",
                    [target, obligation],
                    epistemic_status="inference",
                )
            ],
            addressed=[obligation],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=1)

    assert outcome.stop_reason == "max_calls_exhausted"
    assert len(provider.calls) == 1
    with connect() as con:
        iteration = con.execute(
            "SELECT operation,target_entity_id FROM research_iterations"
        ).fetchone()
    assert (iteration["operation"], iteration["target_entity_id"]) == ("prove", target)


def test_synthesize_targets_blocked_obligation_and_consumes_two_artifacts(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream("Conjecture", "Concrete candidate")
    with connect() as con:
        lemma = add_entity(con, "Lemma", "First supporting lemma")
        technique = add_entity(con, "Technique", "Second supporting construction")
        blocker = add_entity(con, "Obstruction", "Missing composition argument")
        link_workstream_entity(con, workstream, lemma, "evidence")
        link_workstream_entity(con, workstream, technique, "evidence")
        link_workstream_entity(con, workstream, blocker, "blocked_by")

    def responder(decision, _):
        assert decision["operation"] == "synthesize"
        assert decision["target_entity_id"] == blocker
        assert {lemma, technique} <= set(decision["required_consumed_entity_ids"])
        related = [blocker, *decision["required_consumed_entity_ids"]]
        return step_report(
            decision,
            [
                artifact(
                    "failed_approach",
                    "The two ingredients use incompatible round boundaries.",
                    "composition_round_boundary_failure",
                    related,
                    branch_status="failed",
                    epistemic_status="inference",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=1)

    assert outcome.calls_made == 1
    assert len(provider.calls) == 1
    with connect() as con:
        iteration = con.execute(
            "SELECT operation,target_entity_id FROM research_iterations"
        ).fetchone()
    assert (iteration["operation"], iteration["target_entity_id"]) == (
        "synthesize",
        blocker,
    )


@pytest.mark.parametrize(
    ("entity_type", "precise", "expected_operation"),
    [
        ("ResearchIdea", False, "develop"),
        ("Theorem", False, "attack"),
    ],
)
def test_attack_is_reserved_for_concrete_candidates(
    monkeypatch, tmp_path, entity_type, precise, expected_operation
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream(entity_type, "Candidate statement")
    if precise:
        with connect() as con:
            set_attribute(con, target, "precise_candidate", "true")

    def responder(decision, _):
        assert decision["operation"] == expected_operation
        if expected_operation == "attack":
            return step_report(decision, [], attack_outcome="inconclusive")
        return step_report(
            decision,
            [
                artifact(
                    "finding",
                    "The model must first specify a concrete invariant.",
                    "missing_concrete_invariant",
                    [target],
                    epistemic_status="unresolved",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    research(workstream, "openai", max_calls=1)
    assert len(provider.calls) == 1


def test_all_terminal_branches_stop_before_a_model_call(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    with connect() as con:
        blocker = add_entity(con, "Obstruction", "Every branch hits the lower bound")
        set_attribute(con, blocker, "research_branch_status", "blocked")
        link_workstream_entity(con, workstream, blocker, "created")
    provider = DynamicProvider(lambda *_: pytest.fail("provider must not be called"))
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=5)

    assert outcome.calls_made == 0
    assert outcome.stop_reason == "all_branches_blocked_or_refuted"
    assert outcome.final_status == "blocked"
    assert provider.calls == []


def test_duplicate_outputs_are_rejected_and_two_stagnant_iterations_stop(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    with connect() as con:
        existing = add_entity(
            con,
            "Finding",
            "Existing count",
            body="The quorum overlap has size at least n minus two t.",
        )
        set_attribute(con, existing, "research_material_key", "quorum_overlap_count")
        link_workstream_entity(con, workstream, existing, "created")

    def responder(decision, _):
        assert decision["operation"] == "develop"
        return step_report(
            decision,
            [
                artifact(
                    "parameter_analysis",
                    "The quorum overlap has size at least n minus two t.",
                    "quorum_overlap_count",
                    [target],
                    epistemic_status="inference",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=6)

    assert outcome.calls_made == 2
    assert outcome.stop_reason == "stagnation"
    assert outcome.artifact_ids == ()
    with connect() as con:
        iterations = con.execute(
            "SELECT material_progress,duplicate_count FROM research_iterations ORDER BY id"
        ).fetchall()
        count = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    assert [(row[0], row[1]) for row in iterations] == [(0, 1), (0, 1)]
    assert count == 2


def test_max_calls_is_a_hard_bound_and_new_material_is_persisted(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()

    def responder(decision, call_number):
        return step_report(
            decision,
            [
                artifact(
                    "finding",
                    f"Distinct technical consequence number {call_number}.",
                    f"distinct_consequence_{call_number}",
                    [target],
                    epistemic_status="inference",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=2)

    assert outcome.calls_made == 2
    assert len(provider.calls) == 2
    assert len(outcome.artifact_ids) == 2
    assert outcome.stop_reason == "max_calls_exhausted"


def test_human_judgment_request_stops_controller_as_blocked(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()

    def responder(decision, _):
        return step_report(
            decision,
            [
                artifact(
                    "open_question",
                    "Choose whether adaptive corruptions belong in the intended model.",
                    "adaptive_corruption_model_choice",
                    [target],
                    epistemic_status="unresolved",
                )
            ],
            human=True,
            human_reason="The graph does not specify the intended adversary model.",
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    outcome = research(workstream, "openai", max_calls=5)

    assert outcome.calls_made == 1
    assert outcome.stop_reason == "human_judgment_required"
    assert outcome.final_status == "blocked"


def test_budget_refusal_makes_no_iteration_or_state_change(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path, budget=0.0001)
    workstream, _ = make_research_workstream()
    monkeypatch.setattr(
        "theory.research.get_provider",
        lambda _: (_ for _ in ()).throw(AssertionError("provider must not be created")),
    )

    with pytest.raises(BudgetExceededError, match="cannot fit"):
        research(workstream, "openai", max_calls=3)

    with connect() as con:
        status = con.execute(
            "SELECT status FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0]
        iterations = con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0]
        calls = con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
    assert status == "active"
    assert iterations == 0
    assert calls == 0


def test_provider_and_validation_failures_are_recorded_consistently(
    monkeypatch, tmp_path
):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    provider = RaisingProvider()
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)

    with pytest.raises(TheoryError, match="mock controller timeout"):
        research(workstream, "anthropic", max_calls=2)

    with connect() as con:
        iteration = con.execute("SELECT * FROM research_iterations").fetchone()
        call = con.execute("SELECT * FROM api_calls").fetchone()
        status = con.execute(
            "SELECT status FROM workstreams WHERE id=?", (workstream,)
        ).fetchone()[0]
    assert provider.calls == 1
    assert iteration["status"] == "error"
    assert "mock controller timeout" in iteration["error_message"]
    assert call["status"] == "failed"
    assert status == "error"

    with connect() as con:
        con.execute(
            "UPDATE workstreams SET status='active',summary='' WHERE id=?", (workstream,)
        )

    invalid_provider = DynamicProvider(lambda *_: {"unexpected": True})
    monkeypatch.setattr("theory.research.get_provider", lambda _: invalid_provider)
    with pytest.raises(ModelOutputError, match="did not match"):
        research(workstream, "openai", max_calls=1)

    with connect() as con:
        latest_iteration = con.execute(
            "SELECT * FROM research_iterations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        latest_call = con.execute(
            "SELECT * FROM api_calls ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert latest_iteration["status"] == "error"
    assert latest_call["status"] == "completed"


def test_research_cli_and_workstream_show_make_no_extra_call(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()

    def responder(decision, call_number):
        return step_report(
            decision,
            [
                artifact(
                    "finding",
                    "A new invariant candidate relates delivery and certificate formation.",
                    f"delivery_certificate_invariant_{call_number}",
                    [target],
                    epistemic_status="inference",
                )
            ],
        )

    provider = DynamicProvider(responder)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["research", str(workstream), "--provider", "openai", "--max-calls", "1"],
    )

    assert result.exit_code == 0, result.output
    assert "Research controller stopped" in result.output
    assert "max_calls_exhausted" in result.output
    assert "iteration 1: develop" in result.output
    assert "why:" in result.output
    assert "research:develop" in result.output
    assert len(provider.calls) == 1

    shown = runner.invoke(app, ["workstream", "show", str(workstream)])
    assert shown.exit_code == 0, shown.output
    assert "iteration 1: develop" in shown.output
    assert "max_calls_exhausted" in shown.output
    assert len(provider.calls) == 1
