"""Scripted executor quality and actual controller orchestration are separate checks."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest
from pydantic import ValidationError

from theory.config import Config
from theory.db import connect, SCHEMA_VERSION
from theory.errors import BudgetExceededError, ModelOutputError
from theory.graph import add_entity, link_workstream_entity, set_attribute
from theory.models import ModelResult
from theory.prompts import render_prompt
from theory.research import research, _research_prompt_sections, OperationChoice
from theory.research_context import for_workstream
from theory.research_ideation import (
    IdeaBatch, IdeationTrigger, build_ideation_prompt,
    choose_ideation_trigger, ideation_telemetry, validate_ideas,
)
from test_research import artifact, decision_from_prompt, step_report
from test_research_reframe import wa, planning

CASE = json.loads((Path(__file__).parents[1] / "eval/ideation/case02b.json").read_text())


@pytest.fixture
def case02b(wa):
    ws, primary, contract, obligation = wa
    with connect() as con:
        con.execute("UPDATE entities SET body=? WHERE id=?", (CASE["contract"], contract))
        finding = add_entity(con, "Counterexample", "Certificate uniqueness fails", body=CASE["finding"], trust_state="quarantined")
        primitive = add_entity(con, "Assumption", "Existing authenticated super-send", body=CASE["primitive"])
        link_workstream_entity(con, ws, finding, "created")
        link_workstream_entity(con, ws, primitive, "input")
        set_attribute(con, finding, "related_entity_ids", json.dumps([obligation]))
    ids = {"contract": contract, "finding": finding, "primitive": primitive}
    batch = copy.deepcopy({"ideas": CASE["ideas"]})
    for idea in batch["ideas"]:
        for use in idea["exploits"]:
            use["entity_id"] = ids[use["entity_id"]]
    return wa, ids, batch


class OfflineProvider:
    def __init__(self, batch, *, select_idea=True):
        self.batch, self.select_idea = batch, select_idea
        self.requests = []
        self.offered = []
        self.executions = 0

    def complete(self, **kwargs):
        prompt = render_prompt(kwargs["prompt"])
        self.requests.append({**kwargs, "prompt": prompt})
        if kwargs["response_model"] is IdeaBatch:
            result = self.batch
        elif "RESEARCH STATE\n" in prompt:
            state = json.loads(prompt.split("RESEARCH STATE\n", 1)[1])
            self.offered.append(state["legal_moves"])
            move = next((m for m in state["legal_moves"] if m.get("idea") and self.select_idea), state["legal_moves"][0])
            result = {"selected_move_id": move["move_id"], "rationale": "Test the simplifying candidate if useful; otherwise retain the ordinary move."}
        else:
            self.executions += 1
            decision = decision_from_prompt(prompt)
            selected = decision.get("selected_idea")
            refs = [decision["target_entity_id"]]
            if selected:
                refs += [use["entity_id"] for use in selected["exploits"]]
                statement = "Candidate guard propagates authenticated conflict evidence using super-send and permits bottom; timely delivery remains unproved."
            else:
                statement = f"Provisional local construction extension number {self.executions} needs a missing timing argument."
            result = step_report(decision, [artifact("protocol_component", statement,
                f"execution_candidate_{self.executions}", sorted(set(refs)), branch_status="unresolved")])
        return ModelResult(text=json.dumps(result), input_tokens=10, uncached_input_tokens=10,
                           output_tokens=5, cost_usd=0.002, output_cost_usd=0.002)


def install(monkeypatch, batch, **kwargs):
    provider = OfflineProvider(batch, **kwargs)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    return provider


def test_case02b_exposes_selects_and_executes_simplification_with_provenance(case02b, monkeypatch):
    wa, ids, batch = case02b
    provider = install(monkeypatch, batch)
    result = research(wa[0], max_calls=2)
    assert result.ideation_calls_made == result.strategy_calls_made == 1
    assert result.calls_made == 2
    assert result.total_api_calls_made == len(provider.requests) == 4
    offered_ideas = [m for m in provider.offered[0] if m["idea"]]
    assert len(offered_ideas) == 3
    assert all(m["operation"] == "develop" and m["target_entity_id"] == wa[1] for m in offered_ideas)
    selected = offered_ideas[0]["idea"]
    assert "conflict evidence" in selected["mechanism"]
    assert "bottom" in selected["mechanism"] and "super-send" in selected["mechanism"]
    assert "Remove the requirement" in selected["route_change"]
    assert {u["entity_id"] for u in selected["exploits"]} == set(ids.values())
    context = for_workstream(wa[0])
    attrs = context.attributes[result.artifact_ids[0]]
    trace, = ideation_telemetry(wa[0])
    assert json.loads(attrs["research_selected_idea"]) == selected
    assert int(attrs["research_ideation_call_id"]) == trace["id"]
    assert trace["planning"]["selected_idea_id"] == "propagate_conflict"
    assert trace["planning"]["trigger"] == "concrete_refutation"
    assert trace["model"] == Config.load().research_ideation_model
    assert trace["input_tokens"] == 10 and trace["cost_usd"] == 0.002
    assert len(trace["generated_ideas"]) == 3
    assert all(e["trust_state"] == "quarantined" for e in context.entities if e["id"] in result.artifact_ids)
    serialized_graph = json.dumps(context.as_model_payload())
    assert "receiver_filter" not in serialized_graph and "reuse_relay_evidence" not in serialized_graph
    with connect() as con:
        iterations = [dict(r) for r in con.execute("SELECT * FROM research_iterations ORDER BY id")]
    assert iterations[0]["selected_move_id"].endswith(":propagate_conflict")
    assert iterations[1]["selection_mode"] == "deterministic_baseline"
    assert all(row["operation"] != "ideate" for row in iterations)
    assert result.stop_reason == "max_calls_exhausted"


def test_strategist_can_decline_all_ideas(case02b, monkeypatch):
    wa, _, batch = case02b
    provider = install(monkeypatch, batch, select_idea=False)
    research(wa[0], max_calls=2)
    assert len(provider.offered[0]) > len(batch["ideas"])
    assert ideation_telemetry(wa[0])[0]["planning"]["selected_idea_id"] is None
    assert all("research_selected_idea" not in a for a in for_workstream(wa[0]).attributes.values())


@pytest.mark.parametrize("option", ("off", "forced", "one_call"))
def test_ideation_respects_ablation_provider_override_and_call_cap(case02b, monkeypatch, option):
    wa, _, batch = case02b
    provider = install(monkeypatch, batch)
    outcome = research(wa[0], provider_name="openai" if option == "forced" else "auto",
                       strategy="off" if option == "off" else "auto",
                       max_calls=1 if option == "one_call" else 2)
    assert outcome.ideation_calls_made == 0
    assert all(r["response_model"] is not IdeaBatch for r in provider.requests)


def test_failed_ideation_is_logged_without_retry_or_scientific_write(case02b, monkeypatch):
    wa, _, batch = case02b
    batch["ideas"][0]["exploits"][0]["entity_id"] = 999999
    provider = install(monkeypatch, batch)
    before = len(for_workstream(wa[0]).entities)
    with pytest.raises(ModelOutputError, match="supplied graph"):
        research(wa[0], max_calls=2)
    assert len(provider.requests) == 1
    assert len(for_workstream(wa[0]).entities) == before
    trace, = ideation_telemetry(wa[0])
    assert trace["status"] == "failed" and trace["cost_usd"] == 0.002
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0] == 0


def test_ideation_budget_admission_precedes_provider_call(case02b, monkeypatch):
    wa, _, batch = case02b
    Config(monthly_budget_usd=0.000001).save()
    provider = install(monkeypatch, batch)
    with pytest.raises(BudgetExceededError):
        research(wa[0], max_calls=2)
    assert not provider.requests
    assert not ideation_telemetry(wa[0])


def test_cooldown_and_trigger_identity_survive_resume(case02b, monkeypatch):
    wa, _, batch = case02b
    provider = install(monkeypatch, batch)
    research(wa[0], max_calls=2)
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (wa[0],))
    result = research(wa[0], max_calls=2)
    assert result.ideation_calls_made == 0
    assert len(ideation_telemetry(wa[0])) == 1


@pytest.mark.parametrize("fault", ("too_few", "too_many", "duplicate_id", "duplicate_mechanism", "proof", "blank"))
def test_strict_ideation_schema(case02b, fault):
    _, _, batch = case02b
    if fault == "too_few": batch["ideas"].pop()
    if fault == "too_many": batch["ideas"] *= 2
    if fault == "duplicate_id": batch["ideas"][1]["idea_id"] = batch["ideas"][0]["idea_id"]
    if fault == "duplicate_mechanism": batch["ideas"][1]["mechanism"] = batch["ideas"][0]["mechanism"]
    if fault == "proof": batch["ideas"][0]["proof"] = "Supposed proof."
    if fault == "blank": batch["ideas"][0]["main_risk"] = "  "
    with pytest.raises(ValidationError):
        IdeaBatch.model_validate(batch)


def test_trigger_is_deterministic_and_uses_graph_evidence(case02b):
    wa, ids, _ = case02b
    context = for_workstream(wa[0])
    first = choose_ideation_trigger(context, (), ())
    assert first == choose_ideation_trigger(context, (), ())
    assert first.entity_ids == (ids["finding"],)
    assert choose_ideation_trigger(context, (), (first.metadata(()),)) is None
    fresh = replace(context, entities=tuple(e for e in context.entities if e["id"] != ids["finding"]))
    assert choose_ideation_trigger(fresh, (), ()) is None


@pytest.mark.parametrize(("operation", "extra", "reason"), (
    ("attack", {"attack_outcome": "critical_issue"}, "concrete_attack_defect"),
    ("reframe", {"necessity_outcome": "alternative_route_found"}, "contract_route_alternative"),
    ("synthesize", {"resolution_progress": 0, "consumed_entity_ids_json": "[2,3]"}, "unresolved_combination"),
))
def test_iteration_triggers(wa, operation, extra, reason):
    row = {"id": 1, "status": "completed", "operation": operation, "target_entity_id": wa[3], **extra}
    assert choose_ideation_trigger(for_workstream(wa[0]), (row,), ()).reason == reason
    assert choose_ideation_trigger(for_workstream(wa[0]), ({**row, "status": "error"},), ()) is None


def test_repeated_expansion_requires_three_unresolved_steps(wa):
    context = for_workstream(wa[0])
    rows = tuple({"id": i, "status": "completed", "operation": "develop", "resolution_progress": 0} for i in (1,2,3))
    assert choose_ideation_trigger(context, rows[:2], ()) is None
    assert choose_ideation_trigger(context, rows, ()).reason == "repeated_expansion_without_resolution"
    assert choose_ideation_trigger(context, (*rows[:2], {**rows[-1], "resolution_progress": 1}), ()) is None


def test_prompts_keep_ideas_dynamic_and_provisional(case02b):
    wa, _, raw = case02b
    context, primary, _, _, state = planning(wa)
    trigger = choose_ideation_trigger(context, (), ())
    sections = build_ideation_prompt(context, state.problem_contract, trigger)
    other = build_ideation_prompt(context, state.problem_contract, IdeationTrigger("other_reason", (999,)))
    assert sections.stable_prefix == other.stable_prefix
    assert "Do not prove, rank, select" in sections.stable_prefix
    assert "two certificates" not in sections.stable_prefix
    idea = IdeaBatch.model_validate(raw).ideas[0]
    choice = OperationChoice("develop", wa[1], "Test selected idea", idea=idea, ideation_call_id=9)
    prompt = _research_prompt_sections(context, primary, choice)
    changed = _research_prompt_sections(context, primary, replace(choice, idea=idea.model_copy(update={"mechanism": "Another provisional idea"}), ideation_call_id=10))
    assert prompt.stable_prefix == changed.stable_prefix
    assert idea.mechanism not in prompt.stable_prefix
    assert idea.mechanism in prompt.render()


def test_case02b_quality_control_is_separate_from_structural_validity(case02b):
    wa, ids, raw = case02b
    batch = IdeaBatch.model_validate(raw)
    validate_ideas(batch, for_workstream(wa[0]), {wa[1], ids["contract"], ids["primitive"]})
    def useful(ideas):
        return any(all(term in i.mechanism.casefold() for term in ("propagate", "conflict", "bottom", "super-send"))
                   and "remove" in i.route_change.casefold() for i in ideas)
    assert useful(batch.ideas)
    bad = batch.model_copy(update={"ideas": [i.model_copy(update={"mechanism": f"Add a new machinery layer {n}"}) for n, i in enumerate(batch.ideas)]})
    validate_ideas(bad, for_workstream(wa[0]), {ids["contract"], ids["primitive"]})
    assert not useful(bad.ideas)  # Valid orchestration inputs do not imply useful science.


def test_v11_call_log_migration_preserves_old_receipts(monkeypatch, tmp_path):
    from theory.db import SCHEMA
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    with sqlite3.connect(tmp_path / ".theory/research.db") as con:
        con.executescript(SCHEMA.replace("    planning_metadata_json TEXT,\n", ""))
        con.execute("INSERT INTO projects VALUES(1,'Legacy','','t')")
        con.execute("INSERT INTO api_calls(provider,model,purpose,cost_usd,created_at) VALUES('openai','gpt-6-sol','research:develop',0.1,'t')")
        before = con.execute("SELECT * FROM api_calls").fetchone()
        columns = [r[1] for r in con.execute("PRAGMA table_info(api_calls)")]
        con.execute("PRAGMA user_version=11")
    for _ in range(2):
        with connect() as con:
            assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            row = con.execute("SELECT * FROM api_calls").fetchone()
            assert tuple(row[k] for k in columns) == before
            assert row["planning_metadata_json"] is None


def test_unoffered_idea_selection_fails_without_execution(case02b, monkeypatch):
    from theory.research import StrategistDecision
    wa, _, batch = case02b
    provider = install(monkeypatch, batch)
    complete = provider.complete
    def wrong_selection(**kwargs):
        result = complete(**kwargs)
        if kwargs["response_model"] is StrategistDecision:
            return result.model_copy(update={"text": json.dumps({"selected_move_id": "invented_idea", "rationale": "Not an offered move."})})
        return result
    monkeypatch.setattr(provider, "complete", wrong_selection)
    with pytest.raises(ModelOutputError):
        research(wa[0], max_calls=2)
    assert provider.executions == 0 and len(provider.requests) == 2
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0] == 0
        assert con.execute("SELECT status FROM api_calls ORDER BY id DESC LIMIT 1").fetchone()[0] == "failed"


def test_ideation_failure_retains_trigger_and_budget_telemetry(case02b, monkeypatch):
    wa, _, batch = case02b
    provider = install(monkeypatch, batch)
    def failing(**kwargs):
        with connect() as con:
            row = con.execute("SELECT * FROM api_calls ORDER BY id DESC LIMIT 1").fetchone()
            assert row["status"] == "started"
            assert json.loads(row["planning_metadata_json"])["trigger"] == "concrete_refutation"
        raise TimeoutError("Offline timeout")
    monkeypatch.setattr(provider, "complete", failing)
    from theory.errors import TheoryError
    with pytest.raises(TheoryError, match="Offline timeout"):
        research(wa[0], max_calls=2)
    trace, = ideation_telemetry(wa[0])
    assert trace["status"] == "failed" and trace["estimated_max_cost_usd"] > 0
    assert trace["planning"]["trigger"] == "concrete_refutation"


def test_debug_telemetry_shows_ideas_without_extra_calls(case02b, monkeypatch):
    from typer.testing import CliRunner
    from theory.cli import app
    wa, _, batch = case02b
    provider = install(monkeypatch, batch)
    research(wa[0], max_calls=2)
    call_count = len(provider.requests)
    result = CliRunner().invoke(app, ["workstream", "show", str(wa[0])])
    assert result.exit_code == 0, result.output
    assert "ideation trigger: concrete_refutation" in result.output
    assert "selected idea: propagate_conflict" in result.output
    assert "transient idea receiver_filter" in result.output
    assert len(provider.requests) == call_count


def test_synthesis_candidate_gets_ordinary_testing_before_more_ideation(wa):
    row = {"id": 1, "status": "completed", "operation": "synthesize",
           "resolution_progress": 0, "candidate_created_count": 1,
           "consumed_entity_ids_json": "[2,3]", "target_entity_id": wa[3]}
    assert choose_ideation_trigger(for_workstream(wa[0]), (row,), ()) is None
