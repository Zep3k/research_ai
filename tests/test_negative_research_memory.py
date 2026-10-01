"""Remember scoped route failures without restoring unrelated graph history."""
import json
from dataclasses import replace

import pytest

from theory.db import connect
from theory.graph import add_relation, set_attribute
from theory.research import (
    OperationChoice, _history, _obligation_ideation_context, _problem_contract,
    _research_prompt, _top_level_ideation_context, research,
)
from theory.research_context import focus_research_context, for_workstream
from theory.research_ideation import IdeationTrigger, build_ideation_prompt
from theory.research_negative_memory import select_negative_memory
from theory.research_routes import STARTED_AT
from test_execution_context_scope import generated, routed
from test_research import DynamicProvider, artifact, make_research_workstream, step_report


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Real provider forbidden"))


@pytest.fixture
def branch(routed):
    ws, primary, _, root, old, *_ = routed
    with connect() as con:
        obligation = generated(con, ws, "uniform publication obligation", route=root, kind="OpenQuestion")
        set_attribute(con, obligation, "is_proof_obligation", "true")
        set_attribute(con, obligation, "research_obligation_state", "open")
        failed = generated(con, ws, "echo implementation", route=root)
        direct = generated(con, ws, "old scoped echo failure", route=old, kind="Obstruction",
                           related=(obligation,), status="blocked")
        con.execute("UPDATE entities SET body=? WHERE id=?", (
            "Including every valid echo received by B fails for this implementation.\n"
            "This does not establish impossibility of uniform publication. ∑ π", direct,
        ))
        sibling = generated(con, ws, "same route lesson", route=root, kind="FailedApproach",
                            related=(failed,), status="failed")
        audit = generated(con, ws, "route specific stronger premise", kind="Finding", related=(primary,))
        set_attribute(con, audit, "research_necessity_audit", json.dumps({
            "argument": "The proposed global exclusion premise is sufficient on this route; the contract permits abstention."
        }))
        foreign = generated(con, ws, "foreign route failure", route=old, kind="Counterexample", related=(primary,))
        # Fill the continuation window after the old lessons have been recorded.
        current = tuple(generated(con, ws, f"recent component {n}", route=root) for n in range(6))
    history = (*_history(ws), {
        "id": 100, "workstream_id": ws, "iteration_number": 100, "status": "completed",
        "operation": "reframe", "target_entity_id": obligation, "focus_obligation_id": obligation,
        "artifact_ids_json": json.dumps([audit]), "consumed_entity_ids_json": "[]",
        "necessity_outcome": "alternative_route_found",
    })
    return ws, primary, root, old, obligation, direct, sibling, audit, foreign, current, history


def params(branch, *, root=False):
    ws, primary, _, _, obligation, *_ = branch
    return dict(workstream_id=ws, primary_entity_id=primary,
                target_entity_id=primary if root else obligation,
                focus_obligation_id=None if root else obligation, history=branch[-1])


def memory_records(section):
    memory = section.split("NEGATIVE RESEARCH MEMORY\n", 1)[1]
    return json.JSONDecoder().raw_decode(memory.split("\n", 1)[1])[0]


@pytest.mark.parametrize("operation", ["develop", "synthesize", "prove", "reframe"])
def test_focused_generation_retains_old_direct_route_and_reframe_lessons(branch, operation, monkeypatch):
    ws, primary, _, _, obligation, direct, sibling, audit, foreign, *_ = branch
    full = for_workstream(ws)
    before = full.as_dict()
    monkeypatch.setattr("theory.research_context.connect", lambda: pytest.fail("Selection cannot query DB"))
    scoped = focus_research_context(full, **params(branch), operation=operation)
    assert scoped.selections["negative_memory_ids"] == (direct, sibling, audit)
    assert scoped.context_scope["negative_memory_ids"] == [direct, sibling, audit]
    assert {direct, sibling, audit}.issubset(scoped.context_scope["included_entity_ids"])
    assert foreign not in scoped.context_scope["negative_memory_ids"]
    assert {direct, sibling, audit}.isdisjoint(scoped.context_scope["expansion_anchor_ids"])
    assert full.as_dict() == before
    choice = OperationChoice(operation, obligation, "Work on the same scientific branch.", focus_obligation_id=obligation)
    goal = next(e for e in full.entities if e["id"] == primary)
    prompt = _research_prompt(scoped, goal, choice)
    supplied = memory_records(prompt)
    assert [record["entity"]["id"] for record in supplied] == [direct, sibling, audit]
    for record in supplied:
        original = next(e for e in full.entities if e["id"] == record["entity"]["id"])
        assert record["entity"] == original
    assert "Do not treat them as universal impossibility results" in prompt
    assert "explicitly identify what premise or mechanism has materially changed" in prompt


def test_continuation_memory_is_not_limited_to_four_recent_live_components(branch):
    ws, _, root, _, _, direct, sibling, audit, foreign, current, history = branch
    scoped = focus_research_context(for_workstream(ws), **params(branch, root=True),
                                    operation="develop", continuation_route_ids=(root,))
    assert scoped.context_scope["continuation_artifact_ids"] == list(reversed(current[-4:]))
    # A same-route failure is memory even though terminal and older than all four seeds.
    assert sibling in scoped.selections["negative_memory_ids"]
    assert audit in scoped.selections["negative_memory_ids"]
    assert foreign not in scoped.selections["negative_memory_ids"]


def test_root_escape_remembers_live_and_last_departed_route_only(branch):
    ws, _, _, old, _, _, sibling, audit, foreign, *_ = branch
    full = for_workstream(ws)
    selected = select_negative_memory(full, **params(branch, root=True))
    assert selected == (audit, sibling)
    # A completed controller receipt establishes the particular older route left.
    recent = ({"id": 101, "iteration_number": 101, "status": "completed", "operation": "attack",
               "target_entity_id": old, "artifact_ids_json": "[]"},)
    selected = select_negative_memory(full, **{**params(branch, root=True), "history": recent})
    assert foreign in selected


@pytest.mark.parametrize("relation", ["REFUTES", "BLOCKS", "FAILS_AT", "CONTRADICTS"])
def test_explicit_closure_scope_has_first_priority_without_route_ownership(branch, relation):
    ws, _, _, _, obligation, direct, *_ = branch
    with connect() as con:
        evidence = generated(con, ws, "explicit refutation", kind="Counterexample")
        add_relation(con, obligation if relation == "FAILS_AT" else evidence, relation,
                     evidence if relation == "FAILS_AT" else obligation)
    assert select_negative_memory(for_workstream(ws), **params(branch))[0] == evidence


def test_only_accepted_completed_reframe_findings_are_eligible(branch):
    ws, _, _, _, _, direct, sibling, audit, *_ = branch
    full = for_workstream(ws)
    for invalid in ("error", "running"):
        history = (*branch[-1][:-1], {**branch[-1][-1], "status": invalid})
        assert select_negative_memory(full, **{**params(branch), "history": history}) == (direct, sibling)
    assert audit in select_negative_memory(full, **params(branch))


def test_cap_priority_newest_ties_and_entity_relation_order_are_deterministic(branch):
    ws, _, root, old, obligation, direct, *_ = branch
    with connect() as con:
        newest = tuple(generated(con, ws, f"direct lesson {n}", route=old, kind="Obstruction",
                                 related=(obligation,), status="blocked") for n in range(8))
        generated(con, ws, "newer route lesson", route=root, kind="Counterexample")
    full = for_workstream(ws)
    expected = tuple(reversed(newest[-6:]))
    assert select_negative_memory(full, **params(branch)) == expected
    reordered = replace(full, entities=full.entities[::-1], relations=full.relations[::-1],
                        attributes=dict(reversed(list(full.attributes.items()))))
    assert select_negative_memory(reordered, **params(branch)) == expected


@pytest.mark.parametrize("local", [False, True])
def test_ideation_uses_same_memory_selector_and_foregrounds_reframe_lesson(branch, local):
    ws, primary, _, _, obligation, direct, sibling, audit, foreign, *_ = branch
    full = for_workstream(ws)
    if local:
        scoped = _obligation_ideation_context(full, ws, primary, obligation, branch[-1])
    else:
        scoped = _top_level_ideation_context(full, ws, primary, (), branch[-1])
    assert audit in scoped.selections["negative_memory_ids"]
    assert foreign not in scoped.selections["negative_memory_ids"]
    expected = select_negative_memory(full, **params(branch, root=not local))
    assert scoped.selections["negative_memory_ids"] == expected
    trigger = IdeationTrigger("repeated_obligation_failure" if local else "concrete_refutation", (),
                              focus_obligation_id=obligation if local else None)
    prompt = build_ideation_prompt(scoped, _problem_contract(full, ws), trigger).render()
    supplied = memory_records(prompt)
    assert tuple(record["entity"]["id"] for record in supplied) == expected
    state = json.loads(prompt.split("IDEATION STATE\n", 1)[1])
    assert state["graph"]["context_scope"]["negative_memory_ids"] == list(expected)


def test_attack_scope_and_prompt_do_not_receive_route_memory(branch):
    ws, primary, root, _, obligation, direct, sibling, audit, *_ = branch
    with connect() as con:
        candidate = generated(con, ws, "testable publication candidate", route=root, kind="ProofAttempt",
                              related=(obligation,))
    full = for_workstream(ws)
    args = {**params(branch), "target_entity_id": candidate, "operation": "attack"}
    with_history = focus_research_context(full, **args)
    without_history = focus_research_context(full, **{**args, "history": ()})
    assert with_history == without_history
    assert "negative_memory_ids" not in with_history.selections
    assert "negative_memory_ids" not in with_history.context_scope
    assert {sibling, audit}.isdisjoint(with_history.context_scope["included_entity_ids"])
    choice = OperationChoice("attack", candidate, "Test explicit premises.", focus_obligation_id=obligation)
    goal = next(e for e in full.entities if e["id"] == primary)
    assert "NEGATIVE RESEARCH MEMORY" not in _research_prompt(with_history, goal, choice)


def test_hundreds_of_unrelated_failures_do_not_expand_memory_or_prompt(branch):
    ws, primary, _, old, _, *_ = branch
    full = for_workstream(ws)
    scoped = focus_research_context(full, **params(branch, root=True), operation="develop")
    goal = next(e for e in full.entities if e["id"] == primary)
    choice = OperationChoice("develop", primary, "Explore a distinct route.", develop_provenance="frontier")
    before = _research_prompt(scoped, goal, choice)
    with connect() as con:
        foreign = tuple(generated(con, ws, f"unrelated failure {n}", route=old, kind="FailedApproach",
                                  related=(primary,), status="failed") for n in range(200))
        for entity_id in foreign:
            con.execute("UPDATE entities SET body=? WHERE id=?", ("Unrelated historical derivation.\n" * 100, entity_id))
    after = focus_research_context(for_workstream(ws), **params(branch, root=True), operation="develop")
    assert after.entities == scoped.entities
    assert after.selections["negative_memory_ids"] == scoped.selections["negative_memory_ids"]
    assert set(foreign).isdisjoint(after.context_scope["included_entity_ids"])
    assert abs(len(_research_prompt(after, goal, choice).encode()) - len(before.encode())) <= 8


def test_memory_does_not_seed_dependencies_or_reverse_ideation_neighborhoods(branch):
    ws, primary, _, old, obligation, direct, *_ = branch
    with connect() as con:
        old_dependency = generated(con, ws, "old failure ancestry", route=old, kind="Lemma")
        add_relation(con, direct, "DEPENDS_ON", old_dependency)
        neighbor = generated(con, ws, "unrelated inbound failure", kind="Counterexample", related=(direct,))
    full = for_workstream(ws)
    scoped = _obligation_ideation_context(full, ws, primary, obligation, branch[-1])
    assert direct in scoped.selections["negative_memory_ids"]
    assert {old_dependency, neighbor}.isdisjoint(scoped.context_scope["included_entity_ids"])
    assert direct not in scoped.context_scope["expansion_anchor_ids"]


def test_foreign_workstream_live_route_is_not_root_escape_memory(routed):
    ws, primary, *_ = routed
    other_ws, _ = make_research_workstream()
    with connect() as con:
        foreign_root = generated(con, other_ws, "foreign live construction")
        set_attribute(con, foreign_root, STARTED_AT, "10000")
        foreign_failure = generated(con, other_ws, "foreign live route failure", route=foreign_root,
                                    kind="Counterexample", related=(primary,))
        # Neighbor loading sees these records, but their workstream is foreign.
        add_relation(con, foreign_root, "SUPPORTS", primary)
        add_relation(con, foreign_failure, "SUPPORTS", primary)
    full = for_workstream(ws)
    assert {foreign_root, foreign_failure}.issubset({e["id"] for e in full.entities})
    assert select_negative_memory(full, workstream_id=ws, primary_entity_id=primary,
                                  target_entity_id=primary, history=_history(ws)) == ()


def test_primary_synthesis_remembers_consumed_route_instead_of_unrelated_live_route(branch):
    ws, primary, _, _, _, _, sibling, audit, _, current, *_ = branch
    with connect() as con:
        other_root = generated(con, ws, "independent live route")
        set_attribute(con, other_root, STARTED_AT, "200")
        other_failure = generated(con, ws, "independent failure", route=other_root, kind="Counterexample")
    scoped = focus_research_context(for_workstream(ws), **params(branch, root=True),
                                    operation="synthesize", consumed_entity_ids=current[-2:])
    assert scoped.selections["negative_memory_ids"] == (sibling, audit)
    assert other_failure not in scoped.context_scope["included_entity_ids"]


def test_initial_route_neutral_negative_receipt_is_available_to_root_escape(routed):
    ws, primary, *_ = routed
    with connect() as con:
        lesson = generated(con, ws, "initial failed root mechanism", kind="Obstruction",
                           related=(primary,), status="blocked")
    history = ({"id": 101, "iteration_number": 101, "status": "completed", "operation": "develop",
                "target_entity_id": primary, "artifact_ids_json": json.dumps([lesson])},)
    full = for_workstream(ws)
    assert select_negative_memory(full, workstream_id=ws, primary_entity_id=primary,
                                  target_entity_id=primary, history=history) == (lesson,)


def test_controller_receipt_records_exact_memory_without_new_calls_or_routes(routed, monkeypatch):
    ws, primary, _, root, *_ = routed
    with connect() as con:
        lesson = generated(con, ws, "prior local failed rule", route=root, kind="FailedApproach", status="failed")
    provider = DynamicProvider(lambda decision, _: step_report(decision, [artifact(
        "protocol_component", "Replace the failed rule with an explicit deadline predicate.",
        "changed_deadline_predicate", [primary], branch_status="unresolved",
    )]))
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    outcome = research(ws, strategy="off", max_calls=1)
    assert outcome.calls_made == outcome.total_api_calls_made == len(provider.calls) == 1
    assert outcome.strategy_calls_made == outcome.ideation_calls_made == 0
    assert provider.calls[0]["model"] == "gpt-6-sol"
    with connect() as con:
        receipt = con.execute("SELECT * FROM api_calls").fetchone()
    scope = json.loads(receipt["context_scope_json"])
    assert scope["negative_memory_ids"] == [lesson]
    assert scope["negative_memory_ids"] == [record["entity"]["id"] for record in
        memory_records(provider.calls[0]["prompt"])]
