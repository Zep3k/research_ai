"""Offline necessity audits and independent reversible-bypass handshake."""
import copy
import json
import sqlite3

import pytest
from typer.testing import CliRunner

from theory.cli import app
from theory.config import Config
from theory.db import RESEARCH_NECESSITY_COLUMNS, SCHEMA, connect, utcnow
from theory.errors import ModelOutputError
from theory.graph import add_entity, link_workstream_entity, set_attribute
from theory.research import (
    LegalResearchMove, OperationChoice, ProblemContractBrief, ResearchStepReport,
    StrategistDecision, _open_obligation_ids, _research_prompt, _validate_step_report,
    build_research_state, choose_model_route, choose_next_operation,
    generate_legal_research_moves, research,
)
from theory.research_context import for_workstream
from test_research import (
    add_linked_research_entity, artifact, completed_history, decision_from_prompt, init_workspace,
    make_research_workstream, step_report,
)
from test_research_strategy import install_providers


CONTRACT = "Weak consistency: if an honest party decides y, every honest party decides y or bottom."
OBLIGATION = "Conflicting certificates must be impossible."


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Real provider forbidden"))


@pytest.fixture
def wa(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, primary = make_research_workstream()
    with connect() as con:
        con.execute("UPDATE entities SET title='Weak Agreement' WHERE id=?", (primary,))
        contract = add_entity(con, "Definition", "Weak consistency", body=CONTRACT)
        link_workstream_entity(con, workstream, contract, "input")
    obligation = add_linked_research_entity(workstream, "OpenQuestion", OBLIGATION, proof_obligation=True)
    return workstream, primary, contract, obligation


def planning(wa):
    workstream, primary_id, _, _ = wa
    context = for_workstream(workstream)
    primary = next(entity for entity in context.entities if entity["id"] == primary_id)
    with connect() as con:
        history = tuple(dict(row) for row in con.execute("SELECT * FROM research_iterations ORDER BY iteration_number"))
    moves = generate_legal_research_moves(context, workstream, primary, history)
    state = build_research_state(context, workstream_id=workstream, primary=primary, history=history, legal_moves=moves)
    return context, primary, history, moves, state


def test_wa_primary_develop_escape_preserves_old_route(wa, monkeypatch):
    workstream, primary_id, contract, obligation = wa
    broadcast = add_linked_research_entity(
        workstream, "OpenQuestion", "Broadcast must yield identical per-slot outputs.",
        proof_obligation=True,
    )
    context, primary, history, moves, state = planning(wa)
    baseline = choose_next_operation(context, workstream, primary, history)
    ordinary = generate_legal_research_moves(
        context, workstream, primary, history, strategy_enabled=False,
    )
    escape_id = f"develop:{primary_id}:none:none"
    assert [move.move_id for move in moves].count(escape_id) == 1
    assert moves[:-1] == ordinary
    assert LegalResearchMove.from_choice(baseline) in ordinary
    assert baseline.operation == "develop" and baseline.target_entity_id == obligation
    assert {(move.operation, move.focus_obligation_id) for move in ordinary} == {
        (operation, focus) for operation in ("develop", "reframe")
        for focus in (obligation, broadcast)
    }
    escape = moves[-1].to_operation_choice()
    assert escape.open_obligation_ids == (obligation, broadcast)
    assert escape.focus_obligation_id is None
    before = {oid: copy.deepcopy(context.attributes[oid]) for oid in (obligation, broadcast)}
    instruction = "Develop a genuinely different top-level route from the exact problem contract."
    assert instruction not in _research_prompt(context, primary, baseline)

    def select(payload):
        assert any(item["body"] == CONTRACT for item in payload["problem_contract"])
        return {"selected_move_id": escape_id, "rationale": "Identical outputs are stronger than weak consistency; explore conflict visibility."}

    def execute(decision, _):
        assert decision["target_entity_id"] == primary_id
        assert decision["focus_obligation_id"] is None
        return step_report(decision, [
            artifact("protocol_component", "Use conflict visibility to output bottom on conflicting certificates, allowing honest outputs y or bottom.",
                     "conflict_visibility_protocol", [primary_id, contract]),
            artifact("proof_obligation", "Show conflict visibility prevents incompatible non-bottom honest decisions before termination.",
                     "conflict_visibility_safety", [primary_id, contract], epistemic_status="unresolved"),
        ])

    requests, _ = install_providers(monkeypatch, select=select, execute=execute)
    outcome = research(workstream, max_calls=1)
    assert (outcome.calls_made, outcome.strategy_calls_made, outcome.total_api_calls_made) == (1, 1, 2)
    assert outcome.stop_reason == "max_calls_exhausted"
    assert len(outcome.artifact_ids) == 2
    assert instruction in requests[1][1]["prompt"]
    assert CONTRACT in requests[1][1]["prompt"]
    assert requests[1][1]["max_output_tokens"] == 12_000
    context, primary, history, moves, _ = planning(wa)
    for oid in (obligation, broadcast):
        assert context.attributes[oid] == before[oid]
    new_obligation = next(oid for oid in outcome.artifact_ids
                          if context.attributes[oid].get("research_artifact_type") == "proof_obligation")
    assert set(_open_obligation_ids(context, workstream, primary_id)) == {obligation, broadcast, new_obligation}
    assert any(move.operation == "develop" and move.focus_obligation_id == new_obligation for move in moves)
    assert any(move.operation == "reframe" and move.focus_obligation_id == new_obligation for move in moves)
    assert [move.move_id for move in moves].count(escape_id) == 1
    candidate = add_linked_research_entity(
        workstream, "Lemma", "Conditional conflict visibility safety bound",
        related_entity_ids=(new_obligation,),
    )
    context, primary, history, moves, _ = planning(wa)
    synthesis = next(move for move in moves if move.focus_obligation_id == new_obligation
                     and move.operation == "synthesize")
    history += (completed_history("synthesize", new_obligation,
                                  consumed_entity_ids=synthesis.consumed_entity_ids),)
    assert any(move.operation == "prove" and move.target_entity_id == candidate
               and move.focus_obligation_id == new_obligation
               for move in generate_legal_research_moves(context, workstream, primary, history))
    proof = add_linked_research_entity(
        workstream, "ProofAttempt", "Candidate conflict visibility safety proof",
        related_entity_ids=(new_obligation,),
    )
    assert any(move.operation == "attack" and move.target_entity_id == proof
               and move.focus_obligation_id == new_obligation for move in planning(wa)[3])


def responder(wa, *, necessity="alternative_route_found", attack="no_critical_issue", replacement=False):
    _, _, contract, obligation = wa
    def execute(decision, _):
        if decision["operation"] == "attack":
            artifacts = []
            if attack == "critical_issue":
                artifacts = [artifact("obstruction", "Conflict propagation may arrive after an honest decision.",
                                      "late_conflict_obstruction", [decision["target_entity_id"]], branch_status="blocked")]
            return step_report(decision, artifacts, attack_outcome=attack,
                               unresolved=["Propagation timing remains uncertain."] if attack == "inconclusive" else [])
        assert decision["operation"] == "reframe"
        artifacts = []
        if necessity == "alternative_route_found":
            candidate = artifact("finding", "Conflicts may be propagated so honest parties output bottom; forbidding all conflicts is only a sufficient route.",
                                 "propagated_conflict_route", [obligation, contract])
            candidate["reasoning_summary"] = "The stated weak-consistency condition permits bottom. An alternative protocol propagates conflict evidence and makes recipients abstain; it must still meet all other contract conditions."
            artifacts.append(candidate)
        if replacement:
            artifacts.append(artifact("proof_obligation", "Show timely dissemination of conflict evidence before incompatible honest decisions.",
                                      "conflict_dissemination_obligation", [obligation, contract]))
        return {**step_report(decision, artifacts), "necessity_outcome": necessity,
                "necessity_contract_entity_ids": [contract],
                "necessity_audit": {
                    "parent_requirement": {"entity_id": contract, "quote": CONTRACT},
                    "contract_clauses": [{"entity_id": contract, "quote": CONTRACT}],
                    "argument": "Weak consistency permits bottom. Propagate conflicting evidence before incompatible non-bottom decisions; this route does not require excluding conflicting certificates. The timing premise is explicit below when unresolved; no proof of unrelated protocol properties is claimed.",
                    "replacement_obligation_keys": ["conflict_dissemination_obligation"] if replacement else [],
                }}
    return execute


def select_audit_or_attack(state):
    move = next((move for move in state["legal_moves"] if move["operation"] == "attack"), None)
    if move is None:
        move = next(move for move in state["legal_moves"] if move["operation"] == "reframe")
    return {"selected_move_id": move["move_id"], "rationale": "Audit necessity against the exact contract, then independently test the proposed route."}


def test_exact_contract_and_obligation_statement_are_pure_untruncated(wa, monkeypatch):
    workstream, primary_id, contract, obligation = wa
    with connect() as con:
        for kind in ("Model", "Assumption", "Paper"):
            entity = add_entity(con, kind, kind, body=f"Exact {kind} body\n" * 200)
            link_workstream_entity(con, workstream, entity, "input")
        # Simulate legacy controller title truncation with its full stored statement.
        statement = OBLIGATION + " Precise quantifier condition." * 40
        con.execute("UPDATE entities SET title=?,body=? WHERE id=?", (statement[:240], statement + "\n\nReasoning: old derivation", obligation))
        set_attribute(con, obligation, "research_artifact_type", "proof_obligation")
    context, primary, history, moves, state = planning(wa)
    before = copy.deepcopy(context.as_dict())
    def forbidden(*args, **kwargs):
        pytest.fail("Planning must remain pure")
    for name in ("connect", "get_provider", "call_model"):
        monkeypatch.setattr(f"theory.research.{name}", forbidden)
    again = build_research_state(context, workstream_id=workstream, primary=primary, history=history, legal_moves=moves)
    assert state.model_dump_json() == again.model_dump_json()
    assert context.as_dict() == before
    assert {entity.entity_type for entity in state.problem_contract} == {"ResearchIdea", "Definition", "Model", "Assumption"}
    assert next(entity.body for entity in state.problem_contract if entity.id == contract) == CONTRACT
    assert next(entity.body for entity in state.problem_contract if entity.entity_type == "Model") == "Exact Model body\n" * 200
    assert state.open_obligations[0].statement == statement
    assert ProblemContractBrief.model_config["frozen"]
    assert primary_id in {entity.id for entity in state.problem_contract}


@pytest.mark.parametrize("replacement", [False, True])
def test_wa_audit_requires_independent_attack_before_bypass(wa, monkeypatch, replacement):
    workstream, primary_id, contract, obligation = wa
    context, primary, history, moves, state = planning(wa)
    assert [move.move_id for move in moves] == [
        f"develop:{obligation}:{obligation}:none", f"reframe:{obligation}:{obligation}:none",
        f"develop:{primary_id}:none:none",
    ]
    assert LegalResearchMove.from_choice(choose_next_operation(context, workstream, primary, history)) == moves[0]
    assert any(entity.body == CONTRACT for entity in state.problem_contract)
    stage = []
    execute = responder(wa, replacement=replacement)
    def inspect(decision, n):
        if decision["operation"] == "attack":
            after = for_workstream(workstream)
            attrs = after.attributes[obligation]
            assert attrs["research_obligation_state"] == "reframe_pending_attack"
            assert attrs["research_necessity_audit_state"] == "bypass_candidate"
            candidate = int(attrs["research_reframe_candidate_id"])
            assert candidate == decision["target_entity_id"]
            assert after.attributes[candidate]["research_reframe_target_obligation_id"] == str(obligation)
            assert obligation in _open_obligation_ids(after, workstream, primary_id)
            assert next(entity for entity in after.entities if entity["id"] == candidate)["trust_state"] == "quarantined"
            _, _, _, pending_moves, _ = planning(wa)
            assert next(move for move in pending_moves if move.focus_obligation_id == obligation).operation == "attack"
            assert not any(move.operation == "reframe" and move.target_entity_id == obligation for move in pending_moves)
            stage.append(candidate)
        return execute(decision, n)
    requests, constructed = install_providers(monkeypatch, execute=inspect, select=select_audit_or_attack)
    outcome = research(workstream, max_calls=2)
    expected = [("openai", "gpt-6-luna"), ("openai", "gpt-6-sol")]
    expected.append(("openai", "gpt-6-luna"))
    expected.append(("anthropic", "claude-opus-5-5"))
    assert [(name, request["model"]) for name, request in requests] == expected
    assert requests[1][1]["effort"] == "high" and requests[-1][1]["effort"] == "medium"
    assert CONTRACT in requests[0][1]["prompt"] and CONTRACT in requests[1][1]["prompt"]
    assert "independent attack on a necessity-audit finding" in requests[-1][1]["prompt"]
    assert constructed == ["openai", "anthropic"]
    assert outcome.calls_made == 2 and outcome.strategy_calls_made == 2
    assert outcome.total_api_calls_made == len(requests) <= 4
    assert stage
    after, _, _, remaining_moves, _ = planning(wa)
    assert after.attributes[obligation]["research_obligation_state"] == "bypassed"
    assert obligation not in _open_obligation_ids(after, workstream, primary_id)
    assert not any(move.target_entity_id == obligation or move.focus_obligation_id == obligation for move in remaining_moves)
    if replacement:
        remaining = _open_obligation_ids(after, workstream, primary_id)
        assert len(remaining) == 1
        assert any(move.target_entity_id == remaining[0] for move in remaining_moves)
    with connect() as con:
        iterations = [dict(row) for row in con.execute("SELECT * FROM research_iterations ORDER BY id")]
        receipts = [tuple(row) for row in con.execute("SELECT purpose,provider,model FROM api_calls ORDER BY id")]
    assert [row["operation"] for row in iterations] == ["reframe", "attack"]
    assert iterations[0]["necessity_outcome"] == "alternative_route_found"
    assert json.loads(iterations[0]["necessity_contract_entity_ids_json"]) == [contract]
    assert iterations[0]["necessity_audit_summary"]
    assert iterations[0]["progress_class"] == "obligation_audited"
    assert iterations[0]["resolution_progress"] == 0
    assert iterations[1]["progress_class"] == "obligation_bypassed"
    assert iterations[1]["resolution_progress"] == 1
    assert iterations[1]["resolved_obligation_count"] == 0  # Retraction is not a proof.
    assert receipts[0] == ("research:strategy", "openai", "gpt-6-luna")
    assert receipts[1] == ("research:reframe", "openai", "gpt-6-sol")
    assert receipts[-1] == ("research:attack", "anthropic", "claude-opus-5-5")


@pytest.mark.parametrize("necessity", ["required_on_current_routes", "inconclusive"])
def test_completed_audit_stays_open_and_is_not_repeated(wa, monkeypatch, necessity):
    workstream, primary_id, _, obligation = wa
    requests, _ = install_providers(monkeypatch, execute=responder(wa, necessity=necessity), select=select_audit_or_attack)
    outcome = research(workstream, max_calls=1)
    context, _, _, moves, _ = planning(wa)
    assert obligation in _open_obligation_ids(context, workstream, primary_id)
    assert context.attributes[obligation]["research_necessity_audit_state"] == necessity
    assert all(move.operation != "reframe" for move in moves)
    assert len(requests) == 2 and outcome.calls_made == outcome.strategy_calls_made == 1
    with connect() as con:
        row = con.execute("SELECT progress_class,progress_level,resolution_progress FROM research_iterations").fetchone()
    assert tuple(row) == ("obligation_audited", "validation", 0)


@pytest.mark.parametrize("attack", ["critical_issue", "inconclusive"])
def test_failed_independent_attack_keeps_original_open_without_repeat_audit(wa, monkeypatch, attack):
    workstream, primary_id, _, obligation = wa
    requests, _ = install_providers(monkeypatch, execute=responder(wa, attack=attack), select=select_audit_or_attack)
    research(workstream, max_calls=2)
    context, _, _, moves, _ = planning(wa)
    assert context.attributes[obligation]["research_obligation_state"] == "open"
    assert context.attributes[obligation]["research_necessity_audit_state"] == ("challenged" if attack == "critical_issue" else "inconclusive")
    assert obligation in _open_obligation_ids(context, workstream, primary_id)
    assert all(move.operation != "reframe" for move in moves)
    assert len(requests) == 4


def test_input_obligations_never_offer_reframe(wa):
    workstream, _, _, obligation = wa
    with connect() as con:
        link_workstream_entity(con, workstream, obligation, "input")
    _, _, _, moves, _ = planning(wa)
    assert len(moves) == 2 and all(move.operation == "develop" for move in moves)


@pytest.mark.parametrize("provider,strategy", [("auto", "off"), ("openai", "auto"), ("anthropic", "auto")])
def test_ablations_never_select_reframe(wa, monkeypatch, provider, strategy):
    workstream, _, _, obligation = wa
    requests, _ = install_providers(monkeypatch)
    outcome = research(workstream, provider, strategy=strategy, max_calls=1)
    assert outcome.strategy_calls_made == 0 and len(requests) == 1
    assert decision_from_prompt(requests[0][1]["prompt"])["operation"] == "develop"
    assert for_workstream(workstream).attributes[obligation].get("research_necessity_audit_state") is None


@pytest.mark.parametrize("mutate", [
    lambda report, ids: report.update(necessity_outcome="not_applicable"),
    lambda report, ids: report.update(necessity_contract_entity_ids=[]),
    lambda report, ids: report.update(necessity_contract_entity_ids=[ids[3]]),
    lambda report, ids: report.update(necessity_contract_entity_ids=[99999]),
    lambda report, ids: report.update(attack_outcome="no_critical_issue"),
    lambda report, ids: report.update(consumed_entity_ids=[ids[2]]),
    lambda report, ids: report.update(artifacts=[]),
    lambda report, ids: report["artifacts"][0].update(related_entity_ids=[ids[2]]),
    lambda report, ids: report["artifacts"].append(artifact("proof_obligation", "New missing estimate", "missing_estimate", [ids[3]])),
])
def test_reframe_validation_rejects_invalid_contract_or_output(wa, mutate):
    context, primary, _, moves, _ = planning(wa)
    choice = next(move for move in moves if move.operation == "reframe").to_operation_choice()
    decision = decision_from_prompt(_research_prompt(context, primary, choice))
    output = responder(wa)(decision, 1)
    mutate(output, wa)
    report = ResearchStepReport.model_validate(output)
    with pytest.raises(ModelOutputError):
        _validate_step_report(report, context, choice)


def test_non_reframe_cannot_claim_necessity(wa):
    context, primary, _, moves, _ = planning(wa)
    choice = moves[0].to_operation_choice()
    decision = decision_from_prompt(_research_prompt(context, primary, choice))
    report = ResearchStepReport.model_validate({**step_report(decision, []), "necessity_outcome": "required_on_current_routes",
                                               "necessity_contract_entity_ids": [wa[2]]})
    with pytest.raises(ModelOutputError, match="Only reframe"):
        _validate_step_report(report, context, choice)


def test_duplicate_alternative_does_not_start_bypass_handshake(wa, monkeypatch):
    workstream, _, _, obligation = wa
    prior = add_linked_research_entity(workstream, "Finding", "Previously considered route")
    with connect() as con:
        set_attribute(con, prior, "research_material_key", "propagated_conflict_route")
        initial_count = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    requests, _ = install_providers(monkeypatch, execute=responder(wa, replacement=True), select=select_audit_or_attack)
    with pytest.raises(ModelOutputError, match="No non-duplicate"):
        research(workstream, max_calls=1)
    assert len(requests) == 2
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == initial_count
        assert con.execute("SELECT status FROM research_iterations").fetchone()[0] == "error"
    attrs = for_workstream(workstream).attributes[obligation]
    assert attrs.get("research_obligation_state") != "reframe_pending_attack"
    assert attrs.get("research_necessity_audit_state") is None


def test_old_config_gets_reframe_default_and_forced_routes_still_work(wa, tmp_path):
    path = tmp_path / ".theory" / "config.json"
    original = '{"monthly_budget_usd": 100}'
    path.write_text(original)
    cfg = Config.load()
    assert cfg.research_reframe_model == "gpt-6-sol" and path.read_text() == original
    choice = OperationChoice("reframe", wa[3], "test", focus_obligation_id=wa[3])
    route = choose_model_route(choice, cfg)
    assert (route.provider, route.model, route.effort) == ("openai", "gpt-6-sol", "high")
    for provider in ("openai", "anthropic"):
        assert choose_model_route(choice, cfg, provider_override=provider).model == getattr(cfg, f"{provider}_model")
    assert set(StrategistDecision.model_fields) == {"selected_move_id", "rationale"}


def test_v10_migration_preserves_rows_constraints_indexes_foreign_keys_and_sequence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    old_schema = "\n".join(line for line in SCHEMA.replace(", 'reframe'", "").splitlines()
                           if not any(line.strip().startswith(key + " ") for key in RESEARCH_NECESSITY_COLUMNS))
    with sqlite3.connect(tmp_path / ".theory" / "research.db") as con:
        con.executescript(old_schema)
        con.execute("INSERT INTO projects VALUES(1,'Legacy','','t')")
        con.execute("INSERT INTO entities(project_id,entity_type,title,body,created_at,updated_at) VALUES(1,'ResearchIdea','Goal','',?,?)", (utcnow(), utcnow()))
        con.execute("INSERT INTO workstreams(project_id,workstream_type,status,goal,created_at,updated_at) VALUES(1,'research','active','Goal',?,?)", (utcnow(), utcnow()))
        con.execute("""INSERT INTO research_iterations(id,project_id,workstream_id,iteration_number,operation,target_entity_id,rationale,
                       status,material_progress,progress_class,progress_level,progress_events_json,selection_mode,selected_move_id,created_at)
                       VALUES(7,1,1,1,'develop',1,'Historical rationale','completed',1,'frontier_expanded','exploration','[]',
                              'single_legal_move','develop:1:none:none',?)""", (utcnow(),))
        # Preserve extensions/indexes and incoming references, not only the current built-in layout.
        con.execute("ALTER TABLE research_iterations ADD COLUMN external_note TEXT CHECK (external_note != 'invalid')")
        con.execute("UPDATE research_iterations SET external_note='retain exactly'")
        con.execute("CREATE INDEX iteration_custom_index ON research_iterations(operation,status)")
        con.execute("CREATE TABLE iteration_reference(iteration_id INTEGER REFERENCES research_iterations(id))")
        con.execute("INSERT INTO iteration_reference VALUES(7)")
        con.execute("CREATE TRIGGER iteration_note_guard BEFORE UPDATE OF external_note ON research_iterations WHEN NEW.external_note='forbidden' BEGIN SELECT RAISE(ABORT,'forbidden note'); END")
        con.execute("UPDATE sqlite_sequence SET seq=77 WHERE name='research_iterations'")
        original = con.execute("SELECT * FROM research_iterations").fetchone()
        original_columns = [row[1] for row in con.execute("PRAGMA table_info(research_iterations)")]
        original_indexes = con.execute("SELECT name,sql FROM sqlite_master WHERE tbl_name='research_iterations' AND type IN ('index','trigger') ORDER BY name").fetchall()
        original_fks = con.execute("PRAGMA foreign_key_list(research_iterations)").fetchall()
        con.execute("PRAGMA user_version=10")
    for _ in range(2):
        with connect() as con:
            assert con.execute("PRAGMA user_version").fetchone()[0] == 11
            assert con.execute("SELECT name FROM schema_migrations WHERE version=11").fetchone()[0] == "obligation_reframing"
            assert con.execute("PRAGMA foreign_key_check").fetchall() == []
            assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert [tuple(row) for row in con.execute("PRAGMA foreign_key_list(research_iterations)")] == original_fks
            assert [tuple(row) for row in con.execute("SELECT name,sql FROM sqlite_master WHERE tbl_name='research_iterations' AND type IN ('index','trigger') ORDER BY name")] == original_indexes
            row = con.execute("SELECT * FROM research_iterations").fetchone()
            assert tuple(row[key] for key in original_columns) == original
            assert all(row[key] is None for key in RESEARCH_NECESSITY_COLUMNS)
            assert con.execute("SELECT iteration_id FROM iteration_reference").fetchone()[0] == 7
            assert con.execute("SELECT seq FROM sqlite_sequence WHERE name='research_iterations'").fetchone()[0] == 77
    with connect() as con:
        for statement in ("UPDATE research_iterations SET operation='invented'", "UPDATE research_iterations SET external_note='invalid'",
                          "UPDATE research_iterations SET external_note='forbidden'", "UPDATE research_iterations SET resolution_progress=2",
                          "UPDATE research_iterations SET target_entity_id=9999"):
            with pytest.raises(sqlite3.IntegrityError):
                con.execute(statement)
        cur = con.execute("""INSERT INTO research_iterations(project_id,workstream_id,iteration_number,operation,target_entity_id,rationale,status,created_at)
                             VALUES(1,1,2,'reframe',1,'New legal operation','running',?)""", (utcnow(),))
        assert cur.lastrowid == 78


def test_preexisting_critical_issue_blocks_bypass_despite_no_issue_response(wa, monkeypatch):
    workstream, primary_id, _, obligation = wa
    execute = responder(wa)
    def inspect(decision, n):
        if decision["operation"] == "attack":
            # The full snapshot must already contain the issue; inject after audit,
            # before the next controller context load, through the persistence hook below.
            assert decision["target_entity_id"]
        return execute(decision, n)
    import theory.research as controller
    complete = controller._complete_iteration
    def add_issue(*args, **kwargs):
        complete(*args, **kwargs)
        choice = args[2]
        if choice.operation == "reframe":
            context = for_workstream(workstream)
            candidate = int(context.attributes[obligation]["research_reframe_candidate_id"])
            add_linked_research_entity(workstream, "Obstruction", "Previously found circular alternative route", related_entity_ids=(candidate,))
    monkeypatch.setattr(controller, "_complete_iteration", add_issue)
    install_providers(monkeypatch, execute=inspect, select=select_audit_or_attack)
    research(workstream, max_calls=2)
    context = for_workstream(workstream)
    assert obligation in _open_obligation_ids(context, workstream, primary_id)
    assert context.attributes[obligation]["research_obligation_state"] == "open"


def test_attack_can_create_replacement_without_net_resolution_progress(wa, monkeypatch):
    workstream, _, contract, obligation = wa
    execute = responder(wa)
    def with_replacement(decision, n):
        output = execute(decision, n)
        if decision["operation"] == "attack":
            output["artifacts"] = [artifact("proof_obligation", "Check timely delivery of conflict messages before committing.",
                                            "delivery_obligation", [decision["target_entity_id"], contract])]
        return output
    install_providers(monkeypatch, execute=with_replacement, select=select_audit_or_attack)
    research(workstream, max_calls=2)
    with connect() as con:
        row = con.execute("SELECT progress_class,resolution_progress,open_obligations_before,open_obligations_after FROM research_iterations ORDER BY id DESC LIMIT 1").fetchone()
    assert tuple(row) == ("obligation_bypassed", 0, 1, 1)


def test_cli_displays_reframe_audit_without_extra_calls(wa, monkeypatch):
    workstream, _, _, _ = wa
    requests, _ = install_providers(monkeypatch, execute=responder(wa, necessity="required_on_current_routes"), select=select_audit_or_attack)
    result = CliRunner().invoke(app, ["research", str(workstream), "--max-calls", "1"])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "necessity audit: required_on_current_routes" in output and "research:reframe" in output
    assert "progress: validation / obligation_audited" in output
    assert len(requests) == 2


def test_completed_history_alone_suppresses_repeat_audit(wa):
    workstream, _, _, obligation = wa
    context, primary, _, _, _ = planning(wa)
    history = ({"status": "completed", "operation": "reframe", "target_entity_id": obligation},)
    moves = generate_legal_research_moves(context, workstream, primary, history)
    assert all(move.operation != "reframe" for move in moves)


@pytest.mark.parametrize("state", ["blocked", "resolved_candidate", "bypassed", "unnecessary"])
def test_inactive_obligations_are_not_offered_for_reframe(wa, state):
    workstream, _, _, obligation = wa
    with connect() as con:
        set_attribute(con, obligation, "research_obligation_state", state)
    _, _, _, moves, _ = planning(wa)
    assert all(move.operation != "reframe" for move in moves)


def activate_bypass(wa, monkeypatch, *, replacement=True):
    requests, _ = install_providers(monkeypatch, execute=responder(wa, replacement=replacement), select=select_audit_or_attack)
    outcome = research(wa[0], provider_name="auto", max_calls=2)
    context = for_workstream(wa[0])
    candidate = int(context.attributes[wa[3]]["research_reframe_candidate_id"])
    replacements = json.loads(context.attributes[candidate]["research_bypass_replacement_obligation_ids"])
    assert context.attributes[wa[3]]["research_obligation_state"] == "bypassed"
    return requests, outcome, candidate, replacements


@pytest.mark.parametrize("terminal", ["blocked", "failed", "refuted", "contradicted"])
def test_wa_bypass_reopens_without_calls_and_keeps_history(wa, monkeypatch, terminal):
    from theory.research import bypass_route_is_live, reactivate_bypassed_obligations
    workstream, primary, contract, obligation = wa
    requests, outcome, candidate, replacements = activate_bypass(wa, monkeypatch)
    assert len(replacements) == 1 and outcome.total_api_calls_made <= 4
    context = for_workstream(workstream)
    assert bypass_route_is_live(context, obligation, candidate)
    assert contract in json.loads(context.attributes[candidate]["research_necessity_contract_entity_ids"])
    audit = json.loads(context.attributes[candidate]["research_necessity_audit"])
    assert audit["parent_requirement"] == {"entity_id": contract, "quote": CONTRACT}
    assert replacements[0] in _open_obligation_ids(context, workstream, primary)
    with connect() as con:
        history = [tuple(row) for row in con.execute("SELECT * FROM research_iterations ORDER BY id")]
        entities_before = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        receipts = con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0]
        if terminal == "contradicted":
            con.execute("UPDATE entities SET trust_state='contradicted' WHERE id=?", (replacements[0],))
        else:
            set_attribute(con, replacements[0], "research_branch_status", terminal)
    monkeypatch.setattr("theory.research.call_model", lambda **_: pytest.fail("Reactivation must be free"))
    events = reactivate_bypassed_obligations(workstream)
    assert [event.kind for event in events] == ["obligation_reactivated"]
    assert reactivate_bypassed_obligations(workstream) == ()
    context, _, _, moves, _ = planning(wa)
    assert not bypass_route_is_live(context, obligation, candidate)
    assert context.attributes[obligation]["research_obligation_state"] == "open"
    assert obligation in _open_obligation_ids(context, workstream, primary)
    assert any(move.focus_obligation_id == obligation and move.operation != "reframe" for move in moves)
    assert not any(move.operation == "reframe" and move.target_entity_id == obligation for move in moves)
    log = json.loads(context.attributes[obligation]["research_bypass_reactivation_events"])
    assert len(log) == 1 and log[0]["candidate_ids"] == [candidate]
    assert context.attributes[candidate]["research_attack_state"] == "survived_attack"
    with connect() as con:
        assert [tuple(row) for row in con.execute("SELECT * FROM research_iterations ORDER BY id")] == history
        assert con.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == entities_before
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == receipts == len(requests)


def test_one_surviving_replacement_keeps_route_live(wa, monkeypatch):
    from theory.research import bypass_route_is_live, reactivate_bypassed_obligations
    _, _, candidate, replacements = activate_bypass(wa, monkeypatch)
    other = add_linked_research_entity(wa[0], "OpenQuestion", "Another viable replacement premise", proof_obligation=True)
    with connect() as con:
        set_attribute(con, candidate, "research_bypass_replacement_obligation_ids", json.dumps([*replacements, other]))
        set_attribute(con, replacements[0], "research_branch_status", "failed")
        # An inconclusive test is not terminalization of the surviving premise.
        set_attribute(con, other, "research_attack_state", "inconclusive")
    assert bypass_route_is_live(for_workstream(wa[0]), wa[3], candidate)
    assert reactivate_bypassed_obligations(wa[0]) == ()
    assert for_workstream(wa[0]).attributes[wa[3]]["research_obligation_state"] == "bypassed"


def test_another_surviving_bypass_candidate_prevents_reactivation(wa, monkeypatch):
    from theory.research import reactivate_bypassed_obligations
    _, _, candidate, replacements = activate_bypass(wa, monkeypatch)
    other = add_linked_research_entity(wa[0], "Finding", "A separately audited direct parent route")
    with connect() as con:
        set_attribute(con, replacements[0], "research_branch_status", "refuted")
        for key, value in {
            "research_reframe_target_obligation_id": str(wa[3]),
            "research_bypass_replacement_obligation_ids": "[]",
            "research_bypass_activated_iteration_id": "1",
            "research_attack_state": "survived_attack",
        }.items():
            set_attribute(con, other, key, value)
        set_attribute(con, wa[3], "research_bypass_candidate_ids", json.dumps([candidate, other]))
    assert reactivate_bypassed_obligations(wa[0]) == ()
    with connect() as con:
        con.execute("UPDATE entities SET trust_state='contradicted' WHERE id=?", (other,))
    assert len(reactivate_bypassed_obligations(wa[0])) == 1


@pytest.mark.parametrize("terminal", ["failed", "contradicted"])
def test_direct_discharge_persists_until_audit_candidate_invalidated(wa, monkeypatch, terminal):
    from theory.research import reactivate_bypassed_obligations
    _, _, candidate, replacements = activate_bypass(wa, monkeypatch, replacement=False)
    assert replacements == []
    assert reactivate_bypassed_obligations(wa[0]) == ()
    with connect() as con:
        if terminal == "contradicted":
            con.execute("UPDATE entities SET trust_state='contradicted' WHERE id=?", (candidate,))
        else:
            set_attribute(con, candidate, "research_branch_status", terminal)
    assert len(reactivate_bypassed_obligations(wa[0])) == 1


@pytest.mark.parametrize("omission", ["citations", "misattributed_quote", "artifact_dependency"])
def test_wa_cannot_ground_weak_consistency_in_goal_title_only(wa, omission):
    context, _, _, moves, _ = planning(wa)
    choice = next(move for move in moves if move.operation == "reframe").to_operation_choice()
    decision = {"operation": "reframe", "target_entity_id": wa[3], "required_consumed_entity_ids": []}
    raw = responder(wa, replacement=True)(decision, None)
    raw["necessity_contract_entity_ids"] = [wa[1]]  # Omits the decisive Definition #2.
    if omission == "misattributed_quote":
        raw["necessity_audit"]["parent_requirement"]["entity_id"] = wa[1]
        raw["necessity_audit"]["contract_clauses"][0]["entity_id"] = wa[1]
    if omission == "artifact_dependency":
        # Even valid goal-body quotations cannot hide explicit artifact dependencies.
        with connect() as con:
            con.execute("UPDATE entities SET body='Weak Agreement' WHERE id=?", (wa[1],))
        context = for_workstream(wa[0])
        clause = {"entity_id": wa[1], "quote": "Weak Agreement"}
        raw["necessity_audit"]["parent_requirement"] = clause
        raw["necessity_audit"]["contract_clauses"] = [clause]
    with pytest.raises(ModelOutputError, match="contract|Contract"):
        _validate_step_report(ResearchStepReport.model_validate(raw), context, choice)


def test_missing_replacement_premise_is_rejected_without_retry(wa, monkeypatch):
    execute = responder(wa, replacement=True)
    def incomplete(decision, prompt):
        raw = execute(decision, prompt)
        raw["necessity_audit"]["replacement_obligation_keys"].append("unrecorded_premise")
        return raw
    requests, _ = install_providers(monkeypatch, execute=incomplete, select=select_audit_or_attack)
    with pytest.raises(ModelOutputError, match="replacement premise"):
        research(wa[0], provider_name="auto", max_calls=1)
    assert len(requests) == 2
    context = for_workstream(wa[0])
    assert context.attributes[wa[3]].get("research_necessity_audit_state") is None


def test_duplicate_replacement_cannot_be_silently_dropped(wa, monkeypatch):
    execute = responder(wa, replacement=True)
    replacement = execute({"operation": "reframe", "target_entity_id": wa[3], "required_consumed_entity_ids": []}, None)["artifacts"][1]
    prior = add_linked_research_entity(wa[0], "OpenQuestion", replacement["statement"], proof_obligation=True)
    with connect() as con:
        set_attribute(con, prior, "research_material_key", replacement["material_key"])
    requests, _ = install_providers(monkeypatch, execute=execute, select=select_audit_or_attack)
    with pytest.raises(ModelOutputError, match="replacement premise"):
        research(wa[0], provider_name="auto", max_calls=1)
    assert len(requests) == 2
    assert for_workstream(wa[0]).attributes[wa[3]].get("research_reframe_candidate_id") is None


def test_legacy_unnecessary_without_route_provenance_reopens_conservatively(wa):
    from theory.research import reactivate_bypassed_obligations
    with connect() as con:
        set_attribute(con, wa[3], "research_obligation_state", "unnecessary")
        set_attribute(con, wa[3], "research_necessity_audit_state", "unnecessary")
    assert len(reactivate_bypassed_obligations(wa[0])) == 1
    _, _, _, moves, _ = planning(wa)
    assert all(move.operation != "reframe" for move in moves)


def test_reactivation_is_automatic_before_scheduling(wa, monkeypatch):
    _, _, _, replacements = activate_bypass(wa, monkeypatch)
    with connect() as con:
        set_attribute(con, replacements[0], "research_branch_status", "failed")
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (wa[0],))
    def select(state):
        original = next(o for o in state["open_obligations"] if o["id"] == wa[3])
        assert original["obligation_state"] == "open"
        moves = [m for m in state["legal_moves"] if m["focus_obligation_id"] == wa[3]]
        assert moves and all(m["operation"] != "reframe" for m in moves)
        return {"selected_move_id": moves[0]["move_id"], "rationale": "Resume the original route."}
    def execute(decision, _):
        return step_report(decision, [artifact("obstruction", "The original certificate route still needs a coherent locking invariant.",
                                               "resumed_original_route", [decision["target_entity_id"],
                                                                          *decision["required_consumed_entity_ids"]])])
    requests, _ = install_providers(monkeypatch, execute=execute, select=select)
    result = research(wa[0], provider_name="auto", max_calls=1)
    assert result.calls_made == 1 and result.strategy_calls_made <= 1
    assert len(requests) == result.total_api_calls_made <= 2
    assert for_workstream(wa[0]).attributes[wa[3]]["research_obligation_state"] == "open"


def test_route_failure_during_execution_records_reactivation_progress(wa, monkeypatch):
    _, _, _, replacements = activate_bypass(wa, monkeypatch)
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (wa[0],))
    def select(state):
        move = next(m for m in state["legal_moves"] if m["operation"] == "develop" and m["target_entity_id"] == replacements[0])
        return {"selected_move_id": move["move_id"], "rationale": "Check conflict visibility."}
    def execute(decision, _):
        return step_report(decision, [artifact("failed_approach", "This visibility route is refuted by an admissible delayed-evidence schedule.",
                                               "visibility_route_refuted", [replacements[0]], branch_status="refuted")])
    requests, _ = install_providers(monkeypatch, execute=execute, select=select)
    result = research(wa[0], provider_name="auto", max_calls=1)
    assert len(requests) == result.total_api_calls_made == 2
    with connect() as con:
        row = dict(con.execute("SELECT * FROM research_iterations ORDER BY id DESC LIMIT 1").fetchone())
    assert row["open_obligations_before"] == 1 and row["open_obligations_after"] == 2
    assert row["resolution_progress"] == 0
    events = json.loads(row["progress_events_json"])
    assert {e["kind"] for e in events} == {"branch_closed", "obligation_reactivated"}
    assert for_workstream(wa[0]).attributes[wa[3]]["research_obligation_state"] == "open"


def test_reframe_prompt_uses_parent_scope_and_explicit_premises(wa):
    context, primary, _, moves, _ = planning(wa)
    choice = next(move for move in moves if move.operation == "reframe").to_operation_choice()
    prompt = _research_prompt(context, primary, choice)
    assert "A complete solution to unrelated workstream properties is NOT required" in prompt
    assert "EVERY unresolved premise" in prompt
    assert "not absence of evidence" in prompt
    assert "Definition, Assumption, Model, or Technique" in prompt
    assert "parent_requirement" in prompt and "contract_clauses" in prompt
    assert "replacement_obligation_keys" in prompt


def test_later_inconclusive_test_does_not_terminalize_direct_bypass(wa, monkeypatch):
    from theory.research import bypass_route_is_live, reactivate_bypassed_obligations
    _, _, candidate, _ = activate_bypass(wa, monkeypatch, replacement=False)
    with connect() as con:
        set_attribute(con, candidate, "research_attack_state", "inconclusive")
    assert bypass_route_is_live(for_workstream(wa[0]), wa[3], candidate)
    assert reactivate_bypassed_obligations(wa[0]) == ()
