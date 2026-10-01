"""Local alternatives use persisted obligation attempts, route state and cooldown."""
import copy
import json

import pytest

from theory.db import connect, utcnow
from theory.errors import ModelOutputError
from theory.graph import add_relation, set_attribute
from theory.research import (
    _history, _iteration_focus_obligation_id, _obligation_ids, _open_obligation_ids,
    _obligation_ideation_context, research,
)
from theory.research_context import for_workstream
from theory.research_ideation import (
    IdeaBatch, build_ideation_prompt, choose_ideation_trigger, ideation_telemetry, validate_ideas,
)
from theory.research_routes import ROUTE_IDS, SUPERSEDED_BY, STARTED_AT, live_construction_route_ids
from test_research import init_workspace, make_research_workstream, decision_from_prompt
from test_research_consolidation import append_step
from test_research_ideation import install
from test_route_exhaustion import add_work, mark


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Real providers are forbidden"))


@pytest.fixture
def local(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    ws, primary = make_research_workstream()
    root, = append_step(ws, primary)
    obligation = add_work(ws, "proof_obligation", route=root, related=(primary, root))
    with connect() as con:
        con.execute("UPDATE entities SET body=? WHERE id=?", ("FULL_CONTRACT π\n" * 80, primary))
        con.execute("UPDATE entities SET body='Establish uniform publication.' WHERE id=?", (obligation,))
    return ws, primary, root, obligation


def attempt(local, *, obligation=None, operation="develop", status="completed", substantive=True, events=()):
    ws, primary, root, original = local
    obligation = obligation or original
    ids = append_step(ws, primary, route=root, operation=operation, status=status)
    with connect() as con:
        row = con.execute("SELECT id FROM research_iterations WHERE workstream_id=? ORDER BY iteration_number DESC LIMIT 1", (ws,)).fetchone()[0]
        con.execute("UPDATE research_iterations SET target_entity_id=?,focus_obligation_id=?,selected_move_id=?,"
                    "progress_events_json=?,artifact_ids_json=?,material_progress=? WHERE id=?",
                    (obligation, obligation, f"{operation}:{obligation}:{obligation}:none", json.dumps(list(events)),
                     json.dumps(list(ids) if substantive else []), int(substantive and bool(ids)), row))
        for entity in ids:
            set_attribute(con, entity, "research_focus_obligation_id", str(obligation))
            set_attribute(con, entity, "related_entity_ids", json.dumps([primary, root, obligation]))
    return ids


def trigger(local, previous=()):
    ws, primary, _, _ = local
    context = for_workstream(ws)
    obligations = set(_obligation_ids(context, ws, primary))
    history = tuple({**r, "focus_obligation_id": _iteration_focus_obligation_id(context, r, obligations)}
                    for r in _history(ws))
    return choose_ideation_trigger(context, history, previous,
        open_obligation_ids=_open_obligation_ids(context, ws, primary))


def batch(local):
    _, primary, root, obligation = local
    return {"ideas": [
        {"idea_id": name, "mechanism": mechanism, "route_change": "Replace the tested local rule while preserving its parent construction.",
         "main_risk": "The mechanism still needs a sufficient-condition argument.",
         "exploits": [{"entity_id": i, "exploitation": description} for i, description in (
             (primary, "Preserve the exact contract."), (obligation, "Solve this uniform-publication premise."),
             (root, "Use the existing parent construction."))]}
        for name, mechanism in (
            ("monotone_witness", "Retain monotone witness sets through a common acceptance threshold."),
            ("canonical_projection", "Apply canonical projection onto an independently fixed admissible domain."),
            ("deferred_commitment", "Defer commitment until authenticated completion evidence becomes available."))
    ]}


def test_two_substantive_local_attempts_trigger_with_resume_provenance(local):
    attempt(local)
    assert trigger(local) is None
    attempt(local, operation="synthesize")
    result = trigger(local)
    assert result.reason == "repeated_obligation_failure"
    assert result.focus_obligation_id == local[3]
    assert result.entity_ids == (local[3],)
    assert result.iteration_ids == tuple(r["id"] for r in _history(local[0])[-2:])
    assert trigger(local) == result


def test_no_semantic_classification_or_other_obligation_resolution_gate(local):
    first, = attempt(local)
    another = add_work(local[0], "proof_obligation", route=local[2])
    attempt(local, obligation=another, operation="attack", events=(
        {"kind": "obligation_resolved", "obligation_ids": [another], "entity_ids": [local[2]]},
    ))
    second, = attempt(local)
    with connect() as con:
        con.execute("UPDATE entities SET body='Same unclassified mechanism description.' WHERE id IN (?,?)", (first, second))
    assert trigger(local).focus_obligation_id == local[3]


@pytest.mark.parametrize("other", ["different_obligation", "error", "duplicate_only"])
def test_unrelated_failed_and_non_substantive_attempts_do_not_count(local, other):
    attempt(local)
    if other == "different_obligation":
        another = add_work(local[0], "proof_obligation", route=local[2])
        attempt(local, obligation=another)
    elif other == "error":
        attempt(local, status="error")
    else:
        attempt(local, substantive=False)
    assert trigger(local) is None


@pytest.mark.parametrize("event", ["obligation_resolved", "obligation_reactivated", "obligation_bypassed"])
def test_resolution_or_reopening_resets_local_attempts(local, event):
    attempt(local)
    attempt(local)
    assert trigger(local).focus_obligation_id == local[3]
    attempt(local, operation="attack", events=({"kind": event, "obligation_ids": [local[3]], "entity_ids": [local[2]]},))
    attempt(local)
    current = trigger(local)
    assert current is None or current.focus_obligation_id is None
    attempt(local)
    assert trigger(local).iteration_ids == tuple(r["id"] for r in _history(local[0])[-2:])


def test_out_of_iteration_reactivation_timestamp_resets_attempts(local):
    attempt(local)
    attempt(local)
    mark(local[3], research_bypass_reactivation_events=json.dumps([{"timestamp": utcnow(), "previous_state": "bypassed"}]))
    assert trigger(local) is None
    attempt(local)
    current = trigger(local)
    assert current is None or current.focus_obligation_id is None
    attempt(local)
    assert trigger(local).focus_obligation_id == local[3]


@pytest.mark.parametrize("inactive", ["resolved_candidate", "superseded", "blocked", "contradicted"])
def test_inactive_obligations_cannot_trigger_local_ideation(local, inactive):
    attempt(local)
    attempt(local)
    if inactive == "superseded":
        mark(local[2], **{SUPERSEDED_BY: json.dumps([9999])})
    elif inactive == "contradicted":
        with connect() as con:
            con.execute("UPDATE entities SET trust_state='contradicted' WHERE id=?", (local[3],))
    else:
        mark(local[3], research_obligation_state=inactive)
    result = trigger(local)
    assert result is None or result.focus_obligation_id is None


def test_shared_cooldown_and_identity_prevent_local_ideation_spam(local):
    attempt(local)
    attempt(local)
    first = trigger(local)
    previous = (first.metadata(_history(local[0])),)
    assert trigger(local, previous) is None
    for _ in range(2):
        attempt(local)
        assert trigger(local, previous) is None
    attempt(local)
    second = trigger(local, previous)
    assert second.focus_obligation_id == local[3] and second.key != first.key


def test_local_prompt_retains_contract_route_and_failed_mechanism_excludes_old_route(local):
    ws, primary, root, obligation = local
    child, = attempt(local)
    attempt(local)
    failure = add_work(ws, "obstruction", route=root, state="blocked", related=(child,))
    old = add_work(ws, "proof_attempt", related=(primary,))
    mark(old, **{ROUTE_IDS: json.dumps([old]), STARTED_AT: "1", SUPERSEDED_BY: json.dumps([root])})
    stale = add_work(ws, "failed_approach", route=old, state="refuted", related=(primary,))
    with connect() as con:
        add_relation(con, failure, "REFUTES", child)
        con.execute("UPDATE entities SET body='FAILED_LOCAL_MECHANISM' WHERE id=?", (failure,))
        con.execute("UPDATE entities SET body='STALE_ROUTE_SENTINEL' WHERE id=?", (stale,))
    full = for_workstream(ws)
    focused = _obligation_ideation_context(full, ws, primary, obligation)
    assert {primary, root, obligation, child, failure}.issubset({e["id"] for e in focused.entities})
    assert stale not in {e["id"] for e in focused.entities}
    from theory.research import _problem_contract
    prompt = build_ideation_prompt(focused, _problem_contract(full, ws), trigger(local)).render()
    state = json.loads(prompt.split("IDEATION STATE\n", 1)[1])
    assert state["focus_obligation_id"] == obligation
    assert state["problem_contract"][0]["body"] == "FULL_CONTRACT π\n" * 80
    assert "FAILED_LOCAL_MECHANISM" in prompt and "STALE_ROUTE_SENTINEL" not in prompt
    assert "Prefer simpler or weaker sufficient mechanisms" in prompt
    assert "Do not merely repeat recorded failed approaches" in prompt


def test_selected_local_idea_has_focused_move_strong_model_and_original_route(local, monkeypatch):
    ws, primary, root, obligation = local
    attempt(local)
    attempt(local)
    provider = install(monkeypatch, batch(local))
    outcome = research(ws, max_calls=2)
    assert outcome.ideation_calls_made == outcome.strategy_calls_made == 1
    moves = provider.offered[0]
    ideas = [m for m in moves if m["idea"]]
    assert len(ideas) == 3
    assert all(m["operation"] == "develop" and m["target_entity_id"] == m["focus_obligation_id"] == obligation for m in ideas)
    assert any(m["operation"] == "reframe" and m["target_entity_id"] == obligation for m in moves)
    assert not any(m["operation"] == "develop" and m["target_entity_id"] == primary for m in moves)
    execution = next(r for r in provider.requests if r["response_model"].__name__ == "ResearchStepReport")
    assert execution["model"] == "gpt-6-sol"
    decision = decision_from_prompt(execution["prompt"])
    assert decision["target_entity_id"] == decision["focus_obligation_id"] == obligation
    context = for_workstream(ws)
    attrs = context.attributes[outcome.artifact_ids[0]]
    assert json.loads(attrs[ROUTE_IDS]) == [root]
    assert attrs["research_focus_obligation_id"] == str(obligation)
    assert attrs["research_develop_provenance"] == "idea"
    assert json.loads(attrs["research_selected_idea"])["idea_id"] == ideas[0]["idea"]["idea_id"]
    assert live_construction_route_ids(context) == frozenset({root})
    trace, = ideation_telemetry(ws)
    assert trace["planning"]["trigger"] == "repeated_obligation_failure"
    assert trace["planning"]["focus_obligation_id"] == obligation
    assert int(attrs["research_ideation_call_id"]) == trace["id"]
    assert json.loads(trace["context_scope_json"])["target_entity_id"] == obligation
    with connect() as con:
        receipt = con.execute("SELECT * FROM research_iterations ORDER BY iteration_number DESC LIMIT 1 OFFSET 1").fetchone()
        assert receipt["target_entity_id"] == receipt["focus_obligation_id"] == obligation
        assert receipt["develop_provenance"] == "idea"


def test_declined_local_ideas_remain_telemetry_only(local, monkeypatch):
    attempt(local)
    attempt(local)
    provider = install(monkeypatch, batch(local), select_idea=False)
    research(local[0], max_calls=2)
    trace, = ideation_telemetry(local[0])
    assert trace["planning"]["selected_idea_id"] is None
    assert len(trace["generated_ideas"]) == 3
    assert all("research_selected_idea" not in attrs for attrs in for_workstream(local[0]).attributes.values())
    assert all(not decision_from_prompt(r["prompt"]).get("selected_idea")
               for r in provider.requests if r["response_model"].__name__ == "ResearchStepReport")


def test_local_ideation_cooldown_persists_across_invocations(local, monkeypatch):
    attempt(local)
    attempt(local)
    install(monkeypatch, batch(local))
    assert research(local[0], max_calls=2).ideation_calls_made == 1
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (local[0],))
    resumed = research(local[0], max_calls=2)
    assert resumed.ideation_calls_made == 0
    assert len(ideation_telemetry(local[0])) == 1


@pytest.mark.parametrize("ablation", ["off", "openai", "anthropic", "one_call"])
def test_local_ideation_preserves_ablations(local, monkeypatch, ablation):
    attempt(local)
    attempt(local)
    provider = install(monkeypatch, batch(local))
    outcome = research(local[0], strategy="off" if ablation == "off" else "auto",
        provider_name=ablation if ablation in {"openai", "anthropic"} else "auto",
        max_calls=1 if ablation == "one_call" else 2)
    assert outcome.ideation_calls_made == 0
    assert not ideation_telemetry(local[0])
    assert all(r["response_model"] is not IdeaBatch for r in provider.requests)


def test_local_ideas_must_ground_on_the_obligation(local):
    raw = copy.deepcopy(batch(local))
    raw["ideas"][0]["exploits"] = [u for u in raw["ideas"][0]["exploits"] if u["entity_id"] != local[3]]
    with pytest.raises(ModelOutputError, match="focus obligation"):
        validate_ideas(IdeaBatch.model_validate(raw), for_workstream(local[0]), {local[1]}, focus_obligation_id=local[3])
