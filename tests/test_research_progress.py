"""Progress classification uses real local graph writes and offline fake providers."""
import copy
import json
import sqlite3
from dataclasses import replace

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from theory.cli import app
from theory.db import SCHEMA_VERSION, RESEARCH_PROGRESS_COLUMNS, SCHEMA, connect, utcnow
from theory.errors import TheoryError
from theory.graph import set_attribute, set_workstream_status
from theory.research import (
    LegalResearchMove, OperationChoice, PersistedStep, ResearchSelection,
    ResearchStepReport, _complete_iteration, _persist_step, _research_prompt,
    _start_iteration, _strategist_prompt, _validate_step_report,
    build_progress_record, build_research_state, generate_legal_research_moves, research,
)
from theory.research_context import for_workstream
from theory.research_progress import (
    PROGRESS_PRECEDENCE, ProgressEvent, ProgressRecord, progress_event_kinds,
)
from test_research import (
    DynamicProvider, add_linked_research_entity, artifact, decision_from_prompt,
    init_workspace, make_research_workstream, mark_candidate_attempt, step_report,
)
from test_research_strategy import install_providers


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Real provider forbidden"))


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, primary = make_research_workstream()
    obligation = add_linked_research_entity(workstream, "OpenQuestion", "Certificate completeness", proof_obligation=True)
    return workstream, primary, obligation


def persist_operation(workspace, choice, artifacts=(), *, addressed=(), attack_outcome="not_applicable"):
    workstream, primary_id, _ = workspace
    before = for_workstream(workstream)
    primary = next(entity for entity in before.entities if entity["id"] == primary_id)
    decision = decision_from_prompt(_research_prompt(before, primary, choice))
    report = ResearchStepReport.model_validate(step_report(
        decision, list(artifacts), addressed=list(addressed), attack_outcome=attack_outcome,
        unresolved=["Boundary assumption remains uncertain."] if attack_outcome == "inconclusive" else [],
    ))
    _validate_step_report(report, before, choice)
    move = LegalResearchMove.from_choice(choice)
    iteration = _start_iteration(workstream, choice, ResearchSelection(move, "single_legal_move", (move.move_id,), "test"))
    persisted = _persist_step(
        iteration_id=iteration, workstream_id=workstream, provider_name="openai", model="gpt-6-luna",
        choice=choice, context=before, report=report,
    )
    after = for_workstream(workstream)
    params = dict(context_before=before, context_after=after, workstream_id=workstream,
                  primary_id=primary_id, choice=choice, report=report, persisted=persisted)
    record = build_progress_record(**params)
    _complete_iteration(iteration, workstream, choice, report, record)
    with connect() as con:
        row = dict(con.execute("SELECT * FROM research_iterations WHERE id=?", (iteration,)).fetchone())
    assert row["status"] == "completed"
    for key, value in record.persistence_fields().items():
        assert row[key] == value
    return record, params, row


def attack_fixture(workspace, *, closure=True):
    workstream, _, obligation = workspace
    candidate = add_linked_research_entity(
        workstream, "ProofAttempt", "Pigeonhole argument constructs the required certificate.",
        related_entity_ids=(obligation,) if closure else (),
    )
    mark_candidate_attempt(candidate, obligation)
    return candidate, OperationChoice("attack", candidate, "test", open_obligation_ids=(obligation,), focus_obligation_id=obligation)


@pytest.mark.parametrize("closure", [True, False])
def test_survived_attack_distinguishes_actual_obligation_closure(workspace, closure):
    candidate, choice = attack_fixture(workspace, closure=closure)
    record, _, _ = persist_operation(workspace, choice, attack_outcome="no_critical_issue")
    assert record.progress_class == ("obligation_resolved" if closure else "candidate_survived_attack")
    assert record.progress_level == ("closure" if closure else "validation")
    assert record.material_progress and record.resolution_progress == closure
    assert (record.open_obligations_before, record.open_obligations_after) == (1, 0 if closure else 1)
    assert record.resolved_obligation_count == int(closure)
    assert record.candidate_tested_count == 1
    assert "candidate_survived_attack" in {event.kind for event in record.events}
    if closure:
        assert record.events[0] == ProgressEvent(kind="obligation_resolved", entity_ids=(candidate,), obligation_ids=(workspace[2],))


def test_zero_artifact_inconclusive_attack_is_validation_and_not_stagnation(workspace, monkeypatch):
    workstream, _, _ = workspace
    attack_fixture(workspace)
    requests, _ = install_providers(monkeypatch)
    outcome = research(workstream, max_calls=1, strategy="off")
    with connect() as con:
        row = dict(con.execute("SELECT * FROM research_iterations").fetchone())
    assert (row["progress_class"], row["progress_level"]) == ("candidate_tested_inconclusive", "validation")
    assert (row["material_progress"], row["resolution_progress"], row["candidate_tested_count"]) == (1, 0, 1)
    assert row["accepted_artifact_count"] == 0
    assert row["open_obligations_before"] == row["open_obligations_after"] == 1
    assert outcome.calls_made == len(requests) == 1 and outcome.strategy_calls_made == 0
    assert outcome.stop_reason == "max_calls_exhausted"


def test_critical_attack_records_challenge_and_each_terminal_branch(workspace):
    candidate, choice = attack_fixture(workspace)
    record, _, _ = persist_operation(workspace, choice, [
        artifact("obstruction", "Unbounded timeout invalidates the certificate construction.", "timeout_gap", [candidate], branch_status="blocked"),
        artifact("failed_approach", "Adversarial scheduling destroys the induction invariant.", "induction_failure", [candidate], branch_status="refuted"),
    ], attack_outcome="critical_issue")
    assert [event.kind for event in record.events] == ["candidate_challenged", "branch_closed", "branch_closed"]
    assert record.progress_class == "candidate_challenged" and record.progress_level == "closure"
    assert record.closed_branch_count == 2 and record.candidate_tested_count == 1
    assert not record.resolution_progress and record.resolved_obligation_count == 0


@pytest.mark.parametrize("kind,addressed,expected", [
    ("proof_attempt", False, "candidate_created"),
    ("lemma", True, "candidate_created"),
    ("lemma", False, "frontier_expanded"),
])
def test_only_accepted_obligation_candidates_count_as_construction(workspace, kind, addressed, expected):
    workstream, _, obligation = workspace
    choice = OperationChoice("develop", obligation, "test", open_obligation_ids=(obligation,), focus_obligation_id=obligation)
    record, params, _ = persist_operation(workspace, choice, [artifact(
        kind, "A pigeonhole estimate bounds the overlap of certificates.", "overlap_construction", [obligation],
    )], addressed=(obligation,) if addressed else ())
    assert record.progress_class == expected
    assert record.candidate_created_count == int(expected == "candidate_created")
    assert record.progress_level == ("construction" if expected == "candidate_created" else "exploration")
    assert not record.resolution_progress
    accepted_id = params["persisted"].artifact_ids[0]
    assert record.events[0].entity_ids == (accepted_id,)
    if expected == "candidate_created":
        assert record.events[0].obligation_ids == (obligation,)
        assert params["persisted"].accepted_artifacts[0].attempted_obligation_ids == (obligation,)


def test_proof_candidate_for_prove(workspace):
    workstream, _, obligation = workspace
    lemma = add_linked_research_entity(workstream, "Lemma", "Precise target lemma", related_entity_ids=(obligation,))
    choice = OperationChoice("prove", lemma, "test", open_obligation_ids=(obligation,), focus_obligation_id=obligation)
    record, _, _ = persist_operation(workspace, choice, [artifact(
        "proof_attempt", "Induction over certificate length proves the overlap bound.", "induction_proof", [lemma, obligation],
    )], addressed=(obligation,))
    assert record.progress_class == "candidate_created" and record.candidate_created_count == 1
    assert record.open_obligations_after == record.open_obligations_before == 1


def test_new_obligation_is_decomposition_not_resolution(workspace):
    _, _, obligation = workspace
    choice = OperationChoice("develop", obligation, "test", open_obligation_ids=(obligation,), focus_obligation_id=obligation)
    record, _, _ = persist_operation(workspace, choice, [artifact(
        "proof_obligation", "Bound the number of distinct certificates needed for termination.", "certificate_number_obligation", [obligation],
    )])
    assert record.progress_class == "obligation_created" and record.progress_level == "exploration"
    assert record.new_obligation_count == 1 and not record.resolution_progress
    assert (record.open_obligations_before, record.open_obligations_after) == (1, 2)


@pytest.mark.parametrize("kind,status", [("obstruction", "blocked"), ("failed_approach", "failed"), ("failed_approach", "refuted")])
def test_terminal_artifact_closes_only_its_branch(workspace, kind, status):
    _, _, obligation = workspace
    choice = OperationChoice("develop", obligation, "test", open_obligation_ids=(obligation,), focus_obligation_id=obligation)
    record, _, _ = persist_operation(workspace, choice, [artifact(
        kind, "Unbounded adversarial scheduling defeats this construction.", "adversarial_failure", [obligation], branch_status=status,
    )])
    assert record.progress_class == "branch_closed" and record.progress_level == "closure"
    assert record.closed_branch_count == 1 and record.resolved_obligation_count == 0
    assert not record.resolution_progress
    assert record.open_obligations_before == record.open_obligations_after == 1


@pytest.mark.parametrize("kind", ["finding", "parameter_analysis", "protocol_component", "consequence", "open_question", "synthesis"])
def test_other_substantive_artifacts_expand_frontier(workspace, kind):
    _, _, obligation = workspace
    choice = OperationChoice("develop", obligation, "test", open_obligation_ids=(obligation,), focus_obligation_id=obligation)
    record, _, _ = persist_operation(workspace, choice, [artifact(
        kind, "Counting certificates yields a polynomial communication estimate.", "communication_estimate", [obligation],
    )])
    assert record.progress_class == "frontier_expanded" and record.progress_level == "exploration"
    assert record.accepted_artifact_count == 1 and record.candidate_created_count == 0
    assert not record.resolution_progress


def test_duplicate_candidate_is_not_new_construction(workspace):
    workstream, _, obligation = workspace
    prior = add_linked_research_entity(workstream, "ProofAttempt", "Existing candidate", related_entity_ids=(obligation,))
    with connect() as con:
        set_attribute(con, prior, "research_material_key", "existing_proof")
    choice = OperationChoice("develop", obligation, "test", open_obligation_ids=(obligation,), focus_obligation_id=obligation)
    record, _, _ = persist_operation(workspace, choice, [artifact(
        "proof_attempt", "A polished restatement of the previous proof.", "existing_proof", [obligation],
    )], addressed=(obligation,))
    assert record.progress_class == "duplicate_only" and record.progress_level == "none"
    assert record.duplicate_count == 1 and record.accepted_artifact_count == record.candidate_created_count == 0
    assert not record.material_progress and not record.resolution_progress
    assert record.events == (ProgressEvent(kind="duplicate_only"),)


def test_attack_with_only_duplicate_artifacts_still_records_testing(workspace):
    candidate, choice = attack_fixture(workspace)
    with connect() as con:
        set_attribute(con, candidate, "research_material_key", "existing_material")
    record, _, _ = persist_operation(workspace, choice, [artifact(
        "finding", "A restatement of an already known estimate.", "existing_material", [candidate],
    )], attack_outcome="inconclusive")
    assert record.progress_class == "candidate_tested_inconclusive"
    assert record.material_progress and record.accepted_artifact_count == 0 and record.duplicate_count == 1


def test_historical_three_step_trace_and_unchanged_call_bounds(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    obligations = [add_linked_research_entity(workstream, "OpenQuestion", f"Obligation {letter}", proof_obligation=True)
                   for letter in ("A", "B", "C")]
    a, b, _ = obligations
    for obligation in (a, b):
        for n in range(2):
            add_linked_research_entity(workstream, "Finding", f"Estimate {obligation} component {n}", related_entity_ids=(obligation,))

    def select(state):
        focus = b if len(state["recent_iterations"]) < 2 else a
        move = next(move for move in state["legal_moves"] if move["focus_obligation_id"] == focus)
        return {"selected_move_id": move["move_id"], "rationale": "Follow the synthetic trace."}

    def execute(decision, _):
        target = decision["target_entity_id"]
        if decision["operation"] == "attack":
            return step_report(decision, [], attack_outcome="inconclusive", unresolved=["A remaining overlap assumption is not established."])
        refs = [target, *decision["required_consumed_entity_ids"]]
        output = [artifact("proof_attempt",
                           "Pigeonhole counting bounds certificate overlap." if target == b else "Induction constructs a finite message schedule.",
                           f"proof_candidate_{target}", refs)]
        if target == a:
            output.append(artifact("obstruction", "An unbounded timeout prevents termination of this subbranch.", "timeout_obstruction", refs, branch_status="blocked"))
        return step_report(decision, output, addressed=[target])

    requests, _ = install_providers(monkeypatch, select=select, execute=execute)
    outcome = research(workstream, max_calls=3)
    assert (outcome.calls_made, outcome.strategy_calls_made, len(requests)) == (3, 3, 6)
    with connect() as con:
        rows = [dict(row) for row in con.execute("SELECT * FROM research_iterations ORDER BY iteration_number")]
    assert [(row["operation"], row["progress_level"], row["progress_class"]) for row in rows] == [
        ("synthesize", "construction", "candidate_created"),
        ("attack", "validation", "candidate_tested_inconclusive"),
        ("synthesize", "closure", "branch_closed"),
    ]
    assert progress_event_kinds(rows[2]["progress_events_json"]) == ("branch_closed", "candidate_created")
    assert all(row["material_progress"] == 1 and row["resolution_progress"] == 0 for row in rows)
    assert all(row["open_obligations_before"] == row["open_obligations_after"] == 3 for row in rows)
    assert sum(row["resolved_obligation_count"] for row in rows) == 0
    assert [row["accepted_artifact_count"] for row in rows] == [1, 0, 2]
    assert rows[2]["candidate_created_count"] == rows[2]["closed_branch_count"] == 1
    # The next strategist sees prior persisted telemetry, without event subject dumps.
    third_strategy = [request for _, request in requests if "RESEARCH STATE\n" in request["prompt"]][2]
    state = json.loads(third_strategy["prompt"].split("RESEARCH STATE\n")[1])
    assert [row["progress_level"] for row in state["recent_iterations"]] == ["construction", "validation"]
    assert all("progress_events_json" not in row for row in state["recent_iterations"])


def test_progress_is_pure_and_ignores_model_and_strategy_prose(workspace, monkeypatch):
    _, _, obligation = workspace
    choice = OperationChoice("develop", obligation, "test", open_obligation_ids=(obligation,), focus_obligation_id=obligation)
    record, params, _ = persist_operation(workspace, choice, [artifact(
        "finding", "This branch is closed and the theorem is fully verified.", "untrusted_prose", [obligation],
    )])
    before = copy.deepcopy(params["context_before"].as_dict())
    after = copy.deepcopy(params["context_after"].as_dict())
    def forbidden(*args, **kwargs):
        pytest.fail("Progress cannot query providers or storage")
    for name in ("connect", "get_provider", "call_model", "choose_model_route", "for_workstream"):
        monkeypatch.setattr(f"theory.research.{name}", forbidden)
    again = build_progress_record(**{**params, "report": params["report"].model_copy(update={"summary": "All obligations solved!"}),
                                     "choice": replace(choice, rationale="Claim closure")})
    assert again.model_dump_json() == record.model_dump_json()
    assert again.progress_class == "frontier_expanded"
    assert params["context_before"].as_dict() == before and params["context_after"].as_dict() == after


def test_internal_classification_failure_retains_artifacts_and_retry_deduplicates(workspace, monkeypatch):
    import theory.research as controller
    workstream, _, obligation = workspace
    original = controller.build_progress_record
    provider = DynamicProvider(lambda decision, _: step_report(decision, [artifact(
        "finding", "Counting certificates bounds communication complexity.", "retained_finding", [obligation],
    )]))
    monkeypatch.setattr(controller, "get_provider", lambda _: provider)
    def fail(**kwargs):
        raise TheoryError("Synthetic progress derivation failure")
    monkeypatch.setattr(controller, "build_progress_record", fail)
    with pytest.raises(TheoryError, match="Synthetic progress"):
        research(workstream, max_calls=1, strategy="off")
    with connect() as con:
        row = dict(con.execute("SELECT * FROM research_iterations").fetchone())
        assert row["status"] == "error" and row["progress_class"] is None
        retained = json.loads(row["artifact_ids_json"])
        assert len(retained) == 1
        assert con.execute("SELECT status FROM workstreams").fetchone()[0] == "error"
        original_count = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        set_workstream_status(con, workstream, "active")
    monkeypatch.setattr(controller, "build_progress_record", original)
    research(workstream, max_calls=1, strategy="off")
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == original_count
        assert con.execute("SELECT progress_class FROM research_iterations ORDER BY id DESC LIMIT 1").fetchone()[0] == "duplicate_only"
    assert len(provider.calls) == 2  # One per explicit run, no automatic retry.


def test_progress_record_invariants_and_empty_result():
    empty = ProgressRecord.from_events((), open_obligations_before=3, open_obligations_after=3,
                                       accepted_artifact_count=0, duplicate_count=0)
    assert empty.progress_class == "no_progress" and empty.progress_level == "none"
    assert not empty.material_progress
    for field, value in (("material_progress", True), ("resolution_progress", True),
                         ("candidate_created_count", 1), ("closed_branch_count", -1),
                         ("progress_level", "construction"), ("progress_class", "frontier_expanded")):
        with pytest.raises(ValidationError):
            ProgressRecord.model_validate({**empty.model_dump(), field: value})
    with pytest.raises(ValidationError):
        ProgressEvent(kind="candidate_created", entity_ids=(1,))
    with pytest.raises(ValidationError):
        ProgressEvent(kind="frontier_expanded", entity_ids=(2, 1))
    with pytest.raises(ValidationError):
        empty.material_progress = True
    assert PROGRESS_PRECEDENCE == (
        "obligation_resolved", "obligation_bypassed", "obligation_retracted", "candidate_challenged", "branch_closed", "candidate_survived_attack",
        "candidate_tested_inconclusive", "obligation_audited", "candidate_created", "obligation_created", "obligation_reactivated", "frontier_expanded",
        "duplicate_only", "no_progress",
    )


def test_execution_schema_has_no_progress_self_grading():
    fields = ResearchStepReport.model_json_schema()["properties"]
    assert not any("progress" in key or "information_gain" in key for key in fields)


def test_resolution_progress_uses_actual_net_open_count(workspace):
    candidate, choice = attack_fixture(workspace)
    record, _, _ = persist_operation(workspace, choice, [artifact(
        "proof_obligation", "Prove a separate boundary estimate for the next extension.", "extension_obligation", [candidate],
    )], attack_outcome="no_critical_issue")
    assert record.progress_class == "obligation_resolved" and record.progress_level == "closure"
    assert record.resolved_obligation_count == record.new_obligation_count == 1
    assert record.open_obligations_before == record.open_obligations_after == 1
    assert not record.resolution_progress  # Closure of one obligation is not a net reduction here.
    assert {event.kind for event in record.events} == {"obligation_resolved", "candidate_survived_attack", "obligation_created"}


def test_recent_state_preserves_six_mixed_records_and_legacy_nulls(workspace):
    workstream, primary_id, obligation = workspace
    kinds = ("frontier_expanded", "candidate_created", "candidate_tested_inconclusive", "obligation_created", "duplicate_only", "branch_closed")
    records = []
    for kind in kinds:
        event = ProgressEvent(
            kind=kind, entity_ids=() if kind == "duplicate_only" else (primary_id,),
            obligation_ids=(obligation,) if kind in {"candidate_created", "obligation_created"} else (),
        )
        record = ProgressRecord.from_events(
            (event,), open_obligations_before=3, open_obligations_after=4 if kind == "obligation_created" else 3,
            accepted_artifact_count=int(kind not in {"duplicate_only", "candidate_tested_inconclusive"}),
            duplicate_count=int(kind == "duplicate_only"),
        )
        records.append(record)
        with connect() as con:
            cur = con.execute("""INSERT INTO research_iterations(
                project_id,workstream_id,iteration_number,operation,target_entity_id,rationale,status,created_at)
                VALUES(1,?,?,'develop',?,'Synthetic historical decision','completed',?)""",
                (workstream, len(records), primary_id, utcnow()))
            fields = record.persistence_fields()
            con.execute(f"UPDATE research_iterations SET {','.join(f'{key}=?' for key in fields)} WHERE id=?", (*fields.values(), cur.lastrowid))

    def state():
        context = for_workstream(workstream)
        primary = next(entity for entity in context.entities if entity["id"] == primary_id)
        with connect() as con:
            history = tuple(dict(row) for row in con.execute("SELECT * FROM research_iterations ORDER BY iteration_number"))
        moves = generate_legal_research_moves(context, workstream, primary, history)
        return build_research_state(context, workstream_id=workstream, primary=primary, history=history, legal_moves=moves)

    view = state()
    assert [row.progress_level for row in view.recent_iterations] == ["exploration", "construction", "validation", "exploration", "none", "closure"]
    for brief, record in zip(view.recent_iterations, records):
        fields = record.persistence_fields()
        for key in RESEARCH_PROGRESS_COLUMNS:
            if key != "progress_events_json":
                assert getattr(brief, key) == fields[key]
        assert brief.progress_event_kinds == tuple(event.kind for event in record.events)
    assert "progress_events_json" not in view.model_dump_json()
    with connect() as con:
        con.execute("""INSERT INTO research_iterations(
            project_id,workstream_id,iteration_number,operation,target_entity_id,rationale,status,material_progress,created_at)
            VALUES(1,?,7,'develop',?,'Legacy unknown progress','completed',1,?)""", (workstream, primary_id, utcnow()))
    legacy = state().recent_iterations[-1]
    assert legacy.material_progress is True
    assert legacy.progress_event_kinds is None
    assert all(getattr(legacy, key) is None for key in RESEARCH_PROGRESS_COLUMNS if key != "progress_events_json")
    serialized = json.loads(state().model_dump_json())["recent_iterations"][-1]
    assert serialized["progress_class"] is serialized["progress_level"] is serialized["resolution_progress"] is None


def test_strategist_prompt_defines_progress_without_mechanical_ranking(workspace):
    workstream, primary_id, _ = workspace
    context = for_workstream(workstream)
    primary = next(entity for entity in context.entities if entity["id"] == primary_id)
    moves = generate_legal_research_moves(context, workstream, primary, ())
    prompt = _strategist_prompt(build_research_state(context, workstream_id=workstream, primary=primary, history=(), legal_moves=moves))
    for text in ("closure: a branch/candidate/obligation", "validation: a concrete candidate was tested",
                 "construction: a concrete obligation candidate", "exploration: the frontier expanded",
                 "none: no accepted progress", "resolution_progress specifically means",
                 "Do not treat artifact count or material_progress alone", "inconclusive\nattack may still be useful validation",
                 "increasing unresolved work"):
        assert text in prompt
    assert "always choose the highest progress level" not in prompt
    assert "always prefer attacks" not in prompt
    assert "always minimize open obligations" not in prompt


def test_v9_migration_is_idempotent_and_preserves_every_existing_value(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    old_schema = "\n".join(line for line in SCHEMA.splitlines()
                           if not any(line.strip().startswith(key + " ") for key in RESEARCH_PROGRESS_COLUMNS))
    with sqlite3.connect(tmp_path / ".theory" / "research.db") as con:
        con.executescript(old_schema)
        con.execute("INSERT INTO projects VALUES(1,'Legacy','','t')")
        con.execute("INSERT INTO entities(project_id,entity_type,title,body,created_at,updated_at) VALUES(1,'ResearchIdea','Goal','',?,?)", (utcnow(), utcnow()))
        con.execute("INSERT INTO workstreams(project_id,workstream_type,status,goal,created_at,updated_at) VALUES(1,'research','active','Goal',?,?)", (utcnow(), utcnow()))
        con.execute("""INSERT INTO research_iterations(project_id,workstream_id,iteration_number,operation,target_entity_id,rationale,
                       status,material_progress,selection_mode,selected_move_id,selection_rationale,created_at)
                       VALUES(1,1,1,'develop',1,'Historical rationale','completed',1,'single_legal_move','develop:1:none:none','One legal move',?)""", (utcnow(),))
        original = con.execute("SELECT * FROM research_iterations").fetchone()
        old_columns = [row[1] for row in con.execute("PRAGMA table_info(research_iterations)")]
        con.execute("PRAGMA user_version=9")
    for _ in range(2):
        with connect() as con:
            assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert con.execute("SELECT name FROM schema_migrations WHERE version=10").fetchone()[0] == "research_progress_metrics"
            row = con.execute("SELECT * FROM research_iterations").fetchone()
            assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        assert tuple(row[key] for key in old_columns) == original
        assert all(row[key] is None for key in RESEARCH_PROGRESS_COLUMNS)
    with connect() as con:
        for name in RESEARCH_PROGRESS_COLUMNS:
            if name not in {"progress_class", "progress_level", "progress_events_json"}:
                with pytest.raises(sqlite3.IntegrityError):
                    con.execute(f"UPDATE research_iterations SET {name}=-1")
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("UPDATE research_iterations SET resolution_progress=2")


def test_cli_shows_typed_progress_and_retains_legacy_display(workspace, monkeypatch):
    workstream, primary, _ = workspace
    attack_fixture(workspace)
    requests, _ = install_providers(monkeypatch)
    research(workstream, max_calls=1, strategy="off")
    with connect() as con:
        con.execute("""INSERT INTO research_iterations(
            project_id,workstream_id,iteration_number,operation,target_entity_id,rationale,status,material_progress,created_at)
            VALUES(1,?,2,'develop',?,'Legacy','completed',1,?)""", (workstream, primary, utcnow()))
    result = CliRunner().invoke(app, ["workstream", "show", str(workstream)])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "progress: validation / candidate_tested_inconclusive" in output
    assert "obligations: 1 -> 1 | resolution progress: no" in output
    assert "events: candidate_tested_inconclusive" in output
    assert "artifacts accepted: 0 | duplicates: 0" in output
    assert f"iteration 2: develop -> entity #{primary} | completed | material progress" in output
    assert len(requests) == 1  # Display and classification add zero requests.
