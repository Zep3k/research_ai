"""Execution scope depends on graph ownership, not historical prompt volume."""
import json
import sqlite3
from dataclasses import replace

import pytest

from theory.db import SCHEMA_VERSION, connect, utcnow
from theory.errors import ModelOutputError
from theory.graph import add_entity, add_relation, add_source, link_workstream_entity, set_attribute
from theory.research import OperationChoice, _research_prompt, research
from theory.research_context import focus_research_context, for_workstream
from theory.research_routes import ROUTE_IDS, SUPERSEDED_BY
from test_research import DynamicProvider, artifact, init_workspace, make_research_workstream, step_report
from test_research_consolidation import append_step


CONTRACT = "For every admissible schedule, deliver exactly once.\nPreserve ∑ α and the full timing model.\n" * 80
MODEL = "Messages may reorder and duplicate.\nNo FIFO assumption is supplied.\n" * 60


def generated(con, ws, name, *, route=None, related=(), kind="Technique", status="unresolved"):
    entity = add_entity(con, kind, name, body=f"EXACT_{name}\nFull derivation for {name}.\n",
                        trust_state="quarantined")
    link_workstream_entity(con, ws, entity, "created")
    for key, value in {
        "related_entity_ids": json.dumps(list(related)),
        "research_branch_status": status,
        ROUTE_IDS: json.dumps([route] if route else []),
    }.items():
        set_attribute(con, entity, key, value)
    return entity


@pytest.fixture
def routed(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, primary = make_research_workstream()
    old, = append_step(ws, primary)
    root, = append_step(ws, primary)
    with connect() as con:
        con.execute("UPDATE entities SET body=? WHERE id=?", (CONTRACT, primary))
        model = add_entity(con, "Definition", "Full supplied model", body=MODEL)
        link_workstream_entity(con, ws, model, "input")
        set_attribute(con, old, SUPERSEDED_BY, json.dumps([root]))
        current = tuple(generated(con, ws, f"component_{n}", route=root, related=(primary, model))
                        for n in range(6))
        terminal = generated(con, ws, "terminal_piece", route=root, related=(primary,), status="failed")
        challenged = generated(con, ws, "challenged_piece", route=root, related=(primary,))
        set_attribute(con, challenged, "research_attack_state", "challenged")
    return ws, primary, model, root, old, current, terminal, challenged


def continuation_scope(routed, context=None):
    ws, primary, _, root, *_ = routed
    return focus_research_context(context or for_workstream(ws), workstream_id=ws,
        primary_entity_id=primary, target_entity_id=primary, operation="develop",
        continuation_route_ids=(root,))


def append_history(routed, count=100):
    ws, primary, model, _, old, *_ = routed
    with connect() as con:
        ids = []
        for n in range(count):
            # Even a route-neutral record must not enter via the contract input.
            entity = generated(con, ws, f"unrelated_{n}", route=old if n % 2 else None,
                               related=(primary, model))
            con.execute("UPDATE entities SET body=? WHERE id=?",
                        ((f"UNRELATED_HISTORY_{n} with complete historical reasoning.\n" * 100), entity))
            add_relation(con, entity, "SUPPORTS", primary)
            add_relation(con, entity, "USES", model)
            ids.append(entity)
    return set(ids)


def test_primary_continuation_selects_latest_four_live_route_artifacts_only(routed, monkeypatch):
    ws, primary, model, root, old, current, terminal, challenged = routed
    unrelated = append_history(routed)
    full = for_workstream(ws)
    before = full.as_dict()
    monkeypatch.setattr("theory.research_context.connect", lambda: pytest.fail("Focusing must stay pure"))
    focused = continuation_scope(routed, full)
    expected = {primary, model, *current[-4:]}
    assert {e["id"] for e in focused.entities} == expected
    assert not unrelated & expected and {old, root, terminal, challenged}.isdisjoint(expected)
    assert focused.context_scope["continuation_artifact_ids"] == list(reversed(current[-4:]))
    assert focused.context_scope["expansion_anchor_ids"] == list(current[-4:])
    assert focused.context_scope["full_workstream_entity_count"] == len(full.entities)
    assert focused.context_scope["focused_entity_count"] == len(expected)
    assert full.as_dict() == before
    reordered = continuation_scope(routed, replace(full, entities=full.entities[::-1]))
    assert reordered.context_scope["continuation_artifact_ids"] == focused.context_scope["continuation_artifact_ids"]
    assert {e["id"] for e in reordered.entities} == expected


def test_continuation_forward_dependency_chains_and_cycles_are_exact(routed):
    ws, primary, model, root, _, current, *_ = routed
    with connect() as con:
        dependency = generated(con, ws, "forward_premise", kind="Lemma", related=(model,))
        source = add_source(con, dependency, external_url="https://example.test/explicit-premise")
        set_attribute(con, current[-1], "related_entity_ids", json.dumps([primary, current[0]]))
        add_relation(con, current[0], "DEPENDS_ON", dependency)
        add_relation(con, dependency, "USES", current[0])  # Finite closure despite a cycle.
        reverse = generated(con, ws, "reverse_dependent", route=root, related=(current[0],))
        # A terminal reverse-dependent artifact cannot be a latest-live seed.
        set_attribute(con, reverse, "research_branch_status", "blocked")
    full = for_workstream(ws)
    focused = continuation_scope(routed, full)
    expected = {primary, model, *current[-4:], current[0], dependency}
    assert {e["id"] for e in focused.entities} == expected
    assert reverse not in expected
    assert any(s["id"] == source for s in focused.sources)
    assert focused.attributes[dependency] == full.attributes[dependency]
    assert {r["relation_type"] for r in focused.relations} == {"DEPENDS_ON", "USES"}
    assert focused == continuation_scope(routed, full)


@pytest.mark.parametrize("operation", ["synthesize", "develop"])
def test_primary_and_input_targets_never_expand_contract_neighbors(routed, operation):
    ws, primary, model, _, _, current, *_ = routed
    unrelated = append_history(routed)
    for target in (primary, model):
        focused = focus_research_context(for_workstream(ws), workstream_id=ws,
            primary_entity_id=primary, target_entity_id=target, operation=operation,
            consumed_entity_ids=current[-2:] if operation == "synthesize" else ())
        expected = {primary, model, *(current[-2:] if operation == "synthesize" else ())}
        assert {e["id"] for e in focused.entities} == expected
        assert unrelated.isdisjoint(focused.context_scope["included_entity_ids"])
        assert {primary, model}.isdisjoint(focused.context_scope["expansion_anchor_ids"])


@pytest.mark.parametrize("selection", ["consumed", "target", "idea"])
def test_superseded_branch_is_excluded_unless_explicitly_selected(routed, selection):
    ws, primary, _, root, old, current, *_ = routed
    with connect() as con:
        stale_dep = generated(con, ws, "superseded_dependency", route=old)
        stale = generated(con, ws, "superseded_selected", route=old, related=(current[-1], stale_dep))
        add_relation(con, stale, "SUPPORTS", current[-1])
    base = dict(workstream_id=ws, primary_entity_id=primary, target_entity_id=current[-1], operation="prove")
    full = for_workstream(ws)
    focused = focus_research_context(full, **base)
    assert {stale, stale_dep}.isdisjoint(focused.context_scope["included_entity_ids"])
    if selection == "consumed":
        base.update(operation="synthesize", target_entity_id=primary, consumed_entity_ids=(stale, current[-1]))
    elif selection == "target":
        base["target_entity_id"] = stale
    else:
        base.update(operation="develop", target_entity_id=primary, additional_entity_ids=(stale,))
    selected = focus_research_context(full, **base)
    assert {stale, stale_dep}.issubset(selected.context_scope["included_entity_ids"])
    assert stale in selected.context_scope["expansion_anchor_ids"]
    # Selection changes only the view, never ownership or liveness.
    assert for_workstream(ws).attributes[old][SUPERSEDED_BY] == json.dumps([root])


@pytest.mark.parametrize("operation", ["prove", "attack"])
def test_local_proof_work_retains_focus_evidence_dependencies_and_negative_relations(routed, operation):
    ws, primary, model, root, old, *_ = routed
    with connect() as con:
        obligation = generated(con, ws, "timing_obligation", route=root, kind="OpenQuestion", related=(primary,))
        proof = generated(con, ws, "timing_argument", route=root, kind="ProofAttempt", related=(obligation,))
        lemma = generated(con, ws, "direct_evidence", route=root, kind="Lemma", related=(obligation,))
        premise = generated(con, ws, "evidence_dependency", kind="Finding", related=(model,))
        add_relation(con, proof, "DEPENDS_ON", lemma)
        add_relation(con, lemma, "DEPENDS_ON", premise)
        obstruction = generated(con, ws, "known_obstruction", kind="Obstruction", status="blocked")
        counterexample = generated(con, ws, "known_counterexample", kind="Counterexample")
        add_relation(con, obstruction, "BLOCKS", proof)
        add_relation(con, counterexample, "REFUTES", proof)
        omitted = generated(con, ws, "old_route_argument", route=old, kind="ProofAttempt", related=(obligation,))
    full = for_workstream(ws)
    focused = focus_research_context(full, workstream_id=ws, primary_entity_id=primary,
        target_entity_id=proof, focus_obligation_id=obligation, operation=operation)
    expected = {primary, model, obligation, proof, lemma, premise, obstruction, counterexample}
    if operation == "attack":
        expected.add(root)
    assert {e["id"] for e in focused.entities} == expected
    assert omitted not in focused.context_scope["included_entity_ids"]
    assert focused.context_scope["expansion_anchor_ids"] == [obligation, proof]
    assert {r["relation_type"] for r in focused.relations} == {"BLOCKS", "REFUTES", "DEPENDS_ON"}


def test_large_primary_linked_history_does_not_grow_execution_prompt(routed):
    ws, primary, model, *_ = routed
    choice = OperationChoice("develop", primary, "Extend this unfinished route.", continue_construction=True)
    before = continuation_scope(routed)
    goal = next(e for e in before.entities if e["id"] == primary)
    prompt_before = _research_prompt(before, goal, choice)
    unrelated = append_history(routed)
    full = for_workstream(ws)
    after = continuation_scope(routed, full)
    prompt_after = _research_prompt(after, goal, choice)
    assert len(json.dumps(full.as_model_payload()).encode("utf-8")) > 500_000
    assert before.entities == after.entities and before.attributes == after.attributes
    assert {e["id"]: e["body"] for e in after.entities}[primary] == CONTRACT
    assert {e["id"]: e["body"] for e in after.entities}[model] == MODEL
    assert CONTRACT in json.loads(json.dumps(after.as_model_payload(), ensure_ascii=False))["target_entity"]["body"]
    assert "UNRELATED_HISTORY_" not in prompt_after
    assert unrelated.isdisjoint(after.context_scope["included_entity_ids"])
    # Only the digit count in the full-workstream count may grow.
    assert abs(len(prompt_after.encode("utf-8")) - len(prompt_before.encode("utf-8"))) <= 8
    assert len(prompt_after.encode("utf-8")) < 50_000


def test_controller_passes_current_route_and_persists_scope_on_execution_receipt(routed, monkeypatch):
    ws, primary, model, _, _, current, *_ = routed
    unrelated = append_history(routed)
    before = for_workstream(ws)
    provider = DynamicProvider(lambda decision, _: step_report(decision, [artifact(
        "protocol_component", "Tag each pending echo with its release deadline and bound its buffer occupancy.",
        "bounded_echo_deadline", [primary], branch_status="unresolved",
    )]))
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    outcome = research(ws, max_calls=1, strategy="off")
    assert outcome.calls_made == 1
    with connect() as con:
        receipt = dict(con.execute("SELECT * FROM api_calls").fetchone())
        move_id = con.execute("SELECT selected_move_id FROM research_iterations ORDER BY id DESC LIMIT 1").fetchone()[0]
    scope = json.loads(receipt["context_scope_json"])
    assert move_id.endswith(":continue")
    assert scope["full_workstream_entity_count"] == len(before.entities)
    assert scope["focused_entity_count"] == 6
    assert set(scope["included_entity_ids"]) == {primary, model, *current[-4:]}
    assert scope["expansion_anchor_ids"] == list(current[-4:])
    assert unrelated.isdisjoint(scope["included_entity_ids"])
    assert receipt["prompt_utf8_bytes"] == len(provider.calls[0]["prompt"].encode("utf-8"))
    assert "UNRELATED_HISTORY_" not in provider.calls[0]["prompt"]


def test_execution_validation_rejects_omitted_history_and_keeps_scope_receipt(routed, monkeypatch):
    ws, primary, *_ = routed
    omitted = next(iter(append_history(routed, count=1)))
    provider = DynamicProvider(lambda decision, _: step_report(decision, [artifact(
        "protocol_component", "Record the relay scheduling rule with this in-context reference.",
        "invalid_history_reference", [primary, omitted], branch_status="unresolved",
    )]))
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    with pytest.raises(ModelOutputError, match="unknown/out-of-context"):
        research(ws, max_calls=1, strategy="off")
    with connect() as con:
        receipt = con.execute("SELECT context_scope_json,prompt_utf8_bytes FROM api_calls").fetchone()
        assert con.execute("SELECT status FROM workstreams WHERE id=?", (ws,)).fetchone()[0] == "error"
    assert omitted not in json.loads(receipt["context_scope_json"])["included_entity_ids"]
    assert receipt["prompt_utf8_bytes"] == len(provider.calls[0]["prompt"].encode("utf-8"))


def test_v15_scope_migration_retains_historical_receipts_and_is_idempotent(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    path = tmp_path / ".theory" / "research.db"
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        con.execute("ALTER TABLE api_calls DROP COLUMN context_scope_json")
        con.execute("DELETE FROM schema_migrations WHERE version=16")
        con.execute("PRAGMA user_version=15")
        con.execute("INSERT INTO api_calls(provider,model,purpose,cost_usd,created_at) VALUES('openai','gpt-6-sol','legacy',.25,?)", (utcnow(),))
        original = dict(con.execute("SELECT * FROM api_calls").fetchone())
    for _ in range(2):
        with connect() as con:
            assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert con.execute("SELECT name FROM schema_migrations WHERE version=16").fetchone()[0] == "research_execution_context_scope"
            receipt = dict(con.execute("SELECT * FROM api_calls").fetchone())
            assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        assert receipt.pop("context_scope_json") is None
        assert receipt == original
