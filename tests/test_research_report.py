"""Read-only graph projections and CLI snapshots; no paid calls."""
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import sqlite3

import pytest
from pydantic import ValidationError
from rich.console import Console
from typer.testing import CliRunner

from theory.cli import app
from theory.db import connect
from theory.errors import TheoryError
from theory.graph import add_entity, add_relation, add_review, add_source, create_workstream, link_workstream_entity, set_attribute
from theory.research_report import ResearchReport, _read_snapshot, build_research_report, report_from_snapshot
from theory.research_report_ui import render_research_report
from test_research import init_workspace

STAMP = "2026-09-28T12:00:00+00:00"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    monkeypatch.setattr("theory.graph.utcnow", lambda: STAMP)
    return tmp_path


def seed_report():
    with connect() as con:
        workstream = create_workstream(con, "research", "Find a weak-agreement construction")
        def entity(kind, title, role="created", **attrs):
            eid = add_entity(con, kind, title, body=title, trust_state="quarantined" if role == "created" else "unverified")
            link_workstream_entity(con, workstream, eid, role)
            for key, value in attrs.items():
                set_attribute(con, eid, key, value)
            return eid
        goal = entity("ResearchIdea", "Weak agreement under partial synchrony", "input")
        obligation = entity("OpenQuestion", "Rule out conflicting certificates", is_proof_obligation="true",
                            research_artifact_type="proof_obligation", research_obligation_state="bypassed",
                            research_necessity_audit_state="bypassed")
        replacement = entity("OpenQuestion", "Bound the fallback disagreement probability", is_proof_obligation="true",
                             research_obligation_state="candidate_pending_attack")
        candidate = entity("Lemma", "A conditional fallback bound", precise_candidate="true", research_branch_status="promising",
                           research_artifact_type="lemma", model_epistemic_status="inference")
        add_relation(con, candidate, "ATTEMPTS", replacement, trust_state="quarantined")
        route = entity("Finding", "A fallback route bypasses certificate uniqueness", research_artifact_type="finding",
                       research_reframe_target_obligation_id=str(obligation), research_bypass_activated_iteration_id="2",
                       research_bypass_replacement_obligation_ids=json.dumps([replacement]), research_attack_state="survived_attack")
        set_attribute(con, obligation, "research_reframe_candidate_id", str(route))
        set_attribute(con, obligation, "research_bypass_candidate_ids", json.dumps([route]))
        failed = entity("FailedApproach", "Identical certificates fail at the boundary", research_branch_status="refuted",
                        research_artifact_type="failed_approach")
        paper = entity("Paper", "A source on weak agreement", "evidence")
        source = add_source(con, paper, paper_entity_id=paper, page=7, section="3.2", theorem="Lemma 4", excerpt="Conditional agreement bound.")
        sourced = add_entity(con, "Lemma", "Source-backed conditional lemma", trust_state="sourced", source_ids=[source])
        link_workstream_entity(con, workstream, sourced, "evidence")
        # Incomplete locators remain notes attached to quarantined material.
        add_source(con, candidate, section="Unidentified appendix")
        add_review(con, "counterexample_attempt", "no_flaw_found", target_entity_id=route,
                   issues="A bounded pass found no critical issue; replacement premise remains open.", provider="openai", model="gpt-6-sol")
        for number, operation, target, progress, artifacts, attack in (
            (1, "reframe", obligation, "obligation_audited", [route, replacement], "not_applicable"),
            (2, "attack", route, "obligation_bypassed", [], "no_critical_issue"),
            (3, "prove", candidate, "candidate_created", [candidate], "not_applicable"),
        ):
            con.execute("""INSERT INTO research_iterations(
                project_id,workstream_id,iteration_number,operation,target_entity_id,rationale,status,
                material_progress,artifact_ids_json,attack_outcome,created_at,completed_at,
                progress_class,progress_level,progress_events_json,focus_obligation_id)
                VALUES(1,?,?,?,?,?,'completed',1,?,?,?,?,?,?,?,?)""",
                (workstream, number, operation, target, "Test the recorded route", json.dumps(artifacts), attack, STAMP, STAMP,
                 progress, "construction" if number == 3 else "validation", json.dumps([
                     {"kind": progress, "entity_ids": [target], "obligation_ids": [obligation]}]), obligation if number < 3 else replacement))
        for purpose, provider, model, response in (
            ("research:strategy", "openai", "gpt-6-luna", {}),
            ("research:prove", "openai", "gpt-6-sol", {"could_not_determine": ["The boundary case remains undecided."]}),
        ):
            con.execute("""INSERT INTO api_calls(workstream_id,provider,model,purpose,input_tokens,output_tokens,cost_usd,status,
                cache_read_input_tokens,cache_write_input_tokens,cache_write_5m_input_tokens,cache_write_1h_input_tokens,
                response_text,created_at) VALUES(?,?,?,?,1000,100,0.001,'completed',600,300,300,0,?,?)""",
                (workstream, provider, model, purpose, json.dumps(response), STAMP))
        con.execute("""INSERT INTO api_calls(workstream_id,provider,model,purpose,status,created_at)
                       VALUES(?,'anthropic','claude-sonnet-5','research:attack','failed',?)""", (workstream, STAMP))
        # No attribution or receipt leaks from a different workstream.
        other = create_workstream(con, "research", "UNRELATED GOAL")
        unlinked = add_entity(con, "Finding", "UNRELATED ARTIFACT")
        link_workstream_entity(con, other, unlinked, "created")
    return workstream, obligation, replacement, candidate, route, failed


def forbid_writes_and_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Report attempted a write path or model call")
    monkeypatch.setattr("theory.db._migrate_existing", forbidden)
    monkeypatch.setattr("theory.research.reactivate_bypassed_obligations", forbidden)
    monkeypatch.setattr("theory.research.call_model", forbidden)
    monkeypatch.setattr("theory.model_calls.call_model", forbidden)
    monkeypatch.setattr("theory.providers.OpenAIProvider.complete", forbidden)
    monkeypatch.setattr("theory.providers.AnthropicProvider.complete", forbidden)
    real_connect = sqlite3.connect
    traces = []
    def readonly(*args, **kwargs):
        assert kwargs.get("uri") is True and str(args[0]).endswith("?mode=ro")
        con = real_connect(*args, **kwargs)
        def authorize(action, arg1, arg2, database, source):
            assert action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                              sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_PRAGMA}, (action, arg1)
            return sqlite3.SQLITE_OK
        con.set_authorizer(authorize)
        con.set_trace_callback(traces.append)
        return con
    monkeypatch.setattr(sqlite3, "connect", readonly)
    return traces


def test_report_and_cli_are_deterministic_immutable_and_readonly(workspace, monkeypatch):
    workstream, obligation, replacement, candidate, route, failed = seed_report()
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (workspace / ".theory").iterdir() if p.is_file()}
    traces = forbid_writes_and_calls(monkeypatch)
    report = build_research_report(workstream)
    assert report.model_dump_json() == build_research_report(workstream).model_dump_json()
    assert report.frontier.open_obligation_ids == (replacement,)
    assert report.frontier.pending_attack_ids == (candidate,)
    assert report.frontier.candidate_ids == (candidate,)
    assert report.obligations[0].routes[0].survives_under_recorded_graph
    assert failed in report.frontier.blocked_branch_ids
    assert report.latest_material_progress.number == 3
    assert [i.number for i in report.recent_iterations] == [3, 2, 1]
    assert "UNRELATED" not in report.model_dump_json()
    assert report.execution.strategy_calls == 1
    assert report.execution.execution_calls == 2
    assert report.execution.total.cache_read.known_tokens == 1200
    assert report.execution.total.cache_read.unknown_calls == 1
    assert report.execution.total.cost_usd == pytest.approx(0.002)
    assert report.execution.total.unmetered_calls == 1
    with pytest.raises(ValidationError):
        report.goal = "changed"
    with pytest.raises(ValidationError):
        report.entities[0].title = "changed"
    with pytest.raises(ValidationError):
        report.execution.total.cache_read.known_tokens = 0
    output = CliRunner().invoke(app, ["research-report", str(workstream), "--json"])
    assert output.exit_code == 0, output.output
    assert json.loads(output.output) == report.model_dump(mode="json")
    assert ResearchReport.model_validate_json(output.output) == report
    assert output.output == report.model_dump_json(indent=2) + "\n"
    assert json.loads(output.output) == json.loads((FIXTURES / "research_report.json").read_text())
    shown = CliRunner().invoke(app, ["research-report", str(workstream)])
    assert shown.exit_code == 0, shown.output
    assert "Research #1" in shown.output and "Execution accounting" in shown.output
    assert "legal moves:" not in shown.output and "model call:" not in shown.output
    assert all(not command.startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER")) for command in traces)
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (workspace / ".theory").iterdir() if p.is_file()}
    assert after == before


def test_snapshot_aggregation_order_and_rich_snapshot(workspace):
    workstream, *_ = seed_report()
    snapshot = _read_snapshot(workstream)
    first = report_from_snapshot(snapshot)
    second = report_from_snapshot(replace(snapshot, iterations=snapshot.iterations[::-1], reviews=snapshot.reviews[::-1], calls=snapshot.calls[::-1],
                                          context=replace(snapshot.context, entities=snapshot.context.entities[::-1], sources=snapshot.context.sources[::-1])))
    assert first.model_dump_json() == second.model_dump_json()
    output = StringIO()
    render_research_report(first, Console(file=output, width=110, color_system=None, force_terminal=False))
    assert output.getvalue() == (FIXTURES / "research_report.txt").read_text()
    assert [group.role for group in first.artifact_groups] == [
        "problem_contract", "proof_candidates", "findings", "proof_obligations", "negative_results", "source_material"]


@pytest.mark.parametrize("status,reason", [("active", None), ("completed", "max_calls_exhausted"),
    ("blocked", "human_judgment_required"), ("blocked", "stagnation"),
    ("completed", "candidate_survived_attack"), ("abandoned", None), ("error", None)])
def test_empty_and_terminal_lifecycles(workspace, status, reason):
    with connect() as con:
        workstream = create_workstream(con, "research", "A new goal", status=status,
            summary=f"Research controller stopped: {reason}. Recorded detail." if reason else "")
    report = build_research_report(workstream)
    assert report.lifecycle_status == status and report.stop_reason == reason
    assert report.entities == report.obligations == report.recent_iterations == ()
    assert report.execution.total.calls == 0 and report.latest_material_progress is None
    assert bool(report.unresolved_items) == (reason == "human_judgment_required")
    assert CliRunner().invoke(app, ["research-report", str(workstream)]).exit_code == 0


@pytest.mark.parametrize("reactivated", [False, True])
def test_invalidated_bypass_is_reported_without_reconciliation(workspace, monkeypatch, reactivated):
    workstream, obligation, replacement, candidate, route, _ = seed_report()
    with connect() as con:
        set_attribute(con, route, "research_attack_state", "challenged")
        if reactivated:
            set_attribute(con, obligation, "research_obligation_state", "open")
            set_attribute(con, obligation, "research_necessity_audit_state", "reactivated")
            set_attribute(con, obligation, "research_bypass_reactivation_events", json.dumps([
                {"timestamp": STAMP, "previous_state": "bypassed", "candidate_ids": [route]}]))
    forbid_writes_and_calls(monkeypatch)
    report = build_research_report(workstream)
    item = report.obligations[0]
    assert not item.routes[0].survives_under_recorded_graph
    assert item.open_under_controller_rules is reactivated
    assert bool(item.reactivations) is reactivated
    assert item.recorded_state == ("open" if reactivated else "bypassed")
    if not reactivated:
        assert any("does not reactivate" in note for note in report.notes)


def test_active_restart_does_not_reuse_old_stop_or_invent_progress(workspace):
    workstream, *_ = seed_report()
    with connect() as con:
        con.execute("UPDATE research_iterations SET stop_reason='stagnation',progress_class=NULL,progress_level=NULL,progress_events_json=NULL")
    report = build_research_report(workstream)
    assert report.stop_reason is None and report.last_recorded_stop_reason == "stagnation"
    assert report.latest_material_progress.progress_class is None
    assert report.latest_material_progress.events == ()


def test_precise_sources_and_incomplete_notes_stay_distinct(workspace):
    workstream, _, _, candidate, *_ = seed_report()
    report = build_research_report(workstream)
    by_id = {entity.id: entity for entity in report.entities}
    assert not by_id[candidate].sources[0].identifiable_origin
    assert by_id[candidate].trust_state == "quarantined"
    sourced = next(entity for entity in report.entities if entity.trust_state == "sourced")
    assert sourced.sources[0].paper_title == "A source on weak agreement"
    assert (sourced.sources[0].page, sourced.sources[0].section, sourced.sources[0].theorem) == (7, "3.2", "Lemma 4")
    assert sourced.sources[0].excerpt == "Conditional agreement bound."
    assert any("never theorem verification" in note for note in report.notes)


def test_history_window_and_failed_receipt_uncertainty(workspace):
    workstream, *_ = seed_report()
    with connect() as con:
        con.execute("UPDATE api_calls SET response_text=? WHERE status='failed'", (json.dumps({
            "could_not_determine": ["Unaccepted model speculation"], "human_judgment_required": True,
            "human_judgment_reason": "Historical request, not a current stop",
        }),))
        for number in range(4, 14):
            con.execute("""INSERT INTO research_iterations(project_id,workstream_id,iteration_number,operation,
                target_entity_id,rationale,status,created_at) VALUES(1,?,?,'develop',1,'Continue','error',?)""", (workstream, number, STAMP))
    report = build_research_report(workstream)
    assert [i.number for i in report.recent_iterations] == list(range(13, 5, -1))
    assert report.iteration_count == 13 and report.latest_material_progress.number == 3
    assert report.stop_reason is None
    requests = [item for item in report.unresolved_items if item.kind == "model_human_judgment_request"]
    assert len(requests) == 1 and requests[0].scope == "recorded_unreconciled"
    assert "Unaccepted model speculation" not in " ".join(e.statement for e in report.entities)


def test_errors_never_create_or_migrate_a_database(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TheoryError, match="No theory workspace"):
        build_research_report(1)
    assert not (tmp_path / ".theory").exists()
    init_workspace(monkeypatch, tmp_path)
    with pytest.raises(TheoryError, match="does not exist"):
        build_research_report(999)
    with connect() as con:
        wrong = create_workstream(con, "proof", "A proof effort")
    with pytest.raises(TheoryError, match="not a research"):
        build_research_report(wrong)
    with sqlite3.connect(tmp_path / ".theory" / "research.db") as con:
        con.execute("PRAGMA user_version=10")
    before = (tmp_path / ".theory" / "research.db").read_bytes()
    with pytest.raises(TheoryError, match="never migrates"):
        build_research_report(wrong)
    assert (tmp_path / ".theory" / "research.db").read_bytes() == before


def test_empty_json_snapshot(workspace):
    seed_report()
    with connect() as con:
        workstream = create_workstream(con, "research", "New unscheduled research goal")
    assert build_research_report(workstream).model_dump(mode="json") == json.loads(
        (FIXTURES / "research_report_empty.json").read_text())


@pytest.mark.parametrize("audit_state,obligation_state", [
    ("bypass_candidate", "reframe_pending_attack"),
    ("required_on_current_routes", "open"), ("inconclusive", "open"),
])
def test_recorded_necessity_audits_remain_provisional(workspace, audit_state, obligation_state):
    workstream, obligation, _, _, route, _ = seed_report()
    with connect() as con:
        set_attribute(con, obligation, "research_necessity_audit_state", audit_state)
        set_attribute(con, obligation, "research_obligation_state", obligation_state)
        con.execute("DELETE FROM entity_attributes WHERE entity_id=? AND key='research_bypass_activated_iteration_id'", (route,))
    report = build_research_report(workstream)
    item = report.obligations[0]
    assert item.necessity_audit_state == audit_state and item.open_under_controller_rules
    assert not item.routes[0].survives_under_recorded_graph
    assert (route in report.frontier.pending_attack_ids) == (obligation_state == "reframe_pending_attack")


def test_survived_attack_completion_is_not_verification(workspace):
    workstream, _, replacement, candidate, *_ = seed_report()
    with connect() as con:
        set_attribute(con, replacement, "research_obligation_state", "resolved_candidate")
        set_attribute(con, replacement, "research_surviving_candidate_id", str(candidate))
        set_attribute(con, candidate, "research_attack_state", "survived_attack")
        con.execute("UPDATE workstreams SET status='completed' WHERE id=?", (workstream,))
        con.execute("""UPDATE research_iterations SET operation='attack',attack_outcome='no_critical_issue',
                       progress_class='candidate_survived_attack',progress_level='validation',
                       stop_reason='candidate_survived_attack' WHERE iteration_number=3""")
    report = build_research_report(workstream)
    assert report.stop_reason == "candidate_survived_attack"
    assert report.frontier.open_obligation_ids == ()
    assert report.obligations[1].surviving_candidate_id == candidate
    item = next(e for e in report.entities if e.id == candidate)
    assert item.trust_state == "quarantined" and item.bounded_attack_state == "survived_attack"


def test_stale_running_receipts_are_not_reconciled_by_report(workspace, monkeypatch):
    workstream, *_ = seed_report()
    with connect() as con:
        con.execute("UPDATE research_iterations SET status='running',completed_at=NULL WHERE iteration_number=3")
        con.execute("UPDATE api_calls SET status='started',estimated_max_cost_usd=0.25 WHERE status='failed'")
    forbid_writes_and_calls(monkeypatch)
    report = build_research_report(workstream)
    assert report.recent_iterations[0].status == "running"
    assert report.latest_material_progress.number == 2
    assert report.execution.total.pending_calls == 1
    assert report.execution.total.pending_admission_usd == 0.25
    assert report.lifecycle_status == "active"


def test_rich_does_not_interpret_persisted_markup(workspace):
    with connect() as con:
        workstream = create_workstream(con, "research", "[red]Literal goal[/red]")
    shown = CliRunner().invoke(app, ["research-report", str(workstream)])
    assert shown.exit_code == 0 and "[red]Literal goal[/red]" in shown.output
