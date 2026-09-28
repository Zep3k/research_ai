import copy
import json

import pytest

from theory.db import connect
from theory.errors import ModelOutputError, TheoryError
from theory.graph import add_entity, add_relation, add_source, link_workstream_entity, set_attribute
from theory.research import OperationChoice, ResearchStepReport, _research_prompt, _validate_step_report, research
from theory.research_context import focus_research_context, for_workstream
from test_research import (
    DynamicProvider, add_linked_research_entity, artifact, decision_from_prompt,
    init_workspace, make_research_workstream, mark_candidate_attempt, step_report,
)


@pytest.fixture
def branches(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, primary = make_research_workstream()
    ids = {"primary": primary}
    ids["definition"] = add_linked_research_entity(workstream, "Definition", "Shared model", role="input")
    for branch in ("a", "b"):
        ids[branch] = add_linked_research_entity(workstream, "OpenQuestion", f"Obligation {branch}", proof_obligation=True)
        ids[f"lemma_{branch}"] = add_linked_research_entity(
            workstream, "Lemma", f"Lemma {branch}", related_entity_ids=(ids[branch],),
        )
        ids[f"proof_{branch}"] = add_linked_research_entity(
            workstream, "ProofAttempt", f"Proof {branch}", related_entity_ids=(ids[branch],),
        )
        mark_candidate_attempt(ids[f"proof_{branch}"], ids[branch])
    ids["blocker_a"] = add_linked_research_entity(workstream, "Obstruction", "A concrete blocker")
    ids["two_hops"] = add_linked_research_entity(workstream, "Finding", "Do not expand recursively")
    ids["retired_only"] = add_linked_research_entity(workstream, "Finding", "Only a retired edge")
    with connect() as con:
        add_relation(con, ids["blocker_a"], "BLOCKS", ids["proof_a"], trust_state="quarantined")
        add_relation(con, ids["two_hops"], "SUPPORTS", ids["blocker_a"])
        add_relation(con, ids["retired_only"], "SUPPORTS", ids["a"], status="retired")
        # Deliberate lexical overlap must not count as provenance.
        con.execute("UPDATE entities SET body=? WHERE id=?", ("UNRELATED_BRANCH_SENTINEL Obligation a proof overlap " * 30, ids["lemma_b"]))
        sources = {branch: add_source(con, ids[f"lemma_{branch}"], external_url=f"https://example.test/{branch}") for branch in ("a", "b")}
    full = for_workstream(workstream)
    params = dict(workstream_id=workstream, primary_entity_id=primary,
                  target_entity_id=ids["proof_a"], focus_obligation_id=ids["a"])
    return workstream, ids, sources, full, params


def test_focus_is_pure_stable_one_hop_and_filters_all_subobjects(branches, monkeypatch):
    workstream, ids, sources, full, params = branches
    snapshot = copy.deepcopy(full.as_dict())
    def forbidden(*args, **kwargs):
        pytest.fail("Focusing must use only the already-loaded context")
    monkeypatch.setattr("theory.research_context.connect", forbidden)
    monkeypatch.setattr("theory.research.get_provider", forbidden)
    focused = focus_research_context(full, **params)
    expected = {ids[key] for key in ("primary", "definition", "a", "lemma_a", "proof_a", "blocker_a")}
    assert {entity["id"] for entity in focused.entities} == expected
    assert focused.entities == tuple(entity for entity in full.entities if entity["id"] in expected)
    assert focused.target_entity["id"] == ids["proof_a"]
    assert focused.workstream == full.workstream
    assert focused.attributes == {key: value for key, value in full.attributes.items() if key in expected}
    assert {source["id"] for source in focused.sources} == {sources["a"]}
    assert all(link["entity_id"] in expected for link in focused.workstream_links)
    assert all(relation["status"] == "active" and
               {relation["source_entity_id"], relation["target_entity_id"]} <= expected
               for relation in focused.relations)
    assert ids["two_hops"] not in expected
    assert ids["retired_only"] not in expected
    assert focused.selections == {key: tuple(i for i in values if i in expected) for key, values in full.selections.items()}
    for state, group in focused.epistemic.items():
        assert group.entities == tuple(entity for entity in focused.entities if entity["trust_state"] == state)
        assert group.relations == tuple(relation for relation in focused.relations if relation["trust_state"] == state)
        assert group.model_instruction == full.epistemic[state].model_instruction
    assert focused.epistemic["quarantined"].relations
    assert full.as_dict() == snapshot
    assert json.dumps(focused.as_model_payload()) == json.dumps(focus_research_context(full, **params).as_model_payload())
    scope = focused.as_model_payload()["context_scope"]
    assert scope["mode"] == "focused_research_operation"
    assert scope["focused_entity_count"] == 6
    assert scope["full_workstream_entity_count"] == len(full.entities)
    assert scope["included_entity_ids"] == [entity["id"] for entity in focused.entities]


@pytest.mark.parametrize("key", ["related_entity_ids", "addresses_obligation_ids", "research_related_obligation_ids", "research_focus_obligation_id"])
def test_each_explicit_provenance_attribute_establishes_one_hop(branches, key):
    _, ids, _, full, params = branches
    full.attributes.setdefault(ids["two_hops"], {})[key] = str(ids["a"]) if key == "research_focus_obligation_id" else json.dumps([ids["a"]])
    assert ids["two_hops"] in {entity["id"] for entity in focus_research_context(full, **params).entities}


@pytest.mark.parametrize("parameter", ["primary_entity_id", "target_entity_id", "focus_obligation_id", "consumed_entity_ids"])
def test_missing_mandatory_context_fails(branches, parameter):
    _, _, _, full, params = branches
    params[parameter] = (99999,) if parameter == "consumed_entity_ids" else 99999
    with pytest.raises(TheoryError, match="Mandatory research context entities are absent"):
        focus_research_context(full, **params)


def test_missing_workstream_input_also_fails(branches):
    workstream, _, _, full, params = branches
    from dataclasses import replace
    full = replace(full, workstream_links=(*full.workstream_links, dict(workstream_id=workstream, entity_id=99999, role="input")))
    with pytest.raises(TheoryError, match="Mandatory"):
        focus_research_context(full, **params)


def test_synthesis_consumed_entities_are_mandatory_even_across_branches(branches):
    _, ids, _, full, params = branches
    consumed = (ids["lemma_a"], ids["lemma_b"])
    focused = focus_research_context(full, **{**params, "target_entity_id": ids["a"], "consumed_entity_ids": consumed})
    assert set(consumed) <= {entity["id"] for entity in focused.entities}
    choice = OperationChoice("synthesize", ids["a"], "test", consumed_entity_ids=consumed,
                             focus_obligation_id=ids["a"], open_obligation_ids=(ids["a"], ids["b"]))
    primary = next(entity for entity in full.entities if entity["id"] == ids["primary"])
    prompt = _research_prompt(focused, primary, choice)
    decision = decision_from_prompt(prompt)
    report = ResearchStepReport.model_validate(step_report(decision, [artifact(
        "obstruction", "These two estimates do not compose.", "noncomposing_estimates",
        [ids["a"], *consumed], branch_status="blocked",
    )]))
    _validate_step_report(report, focused, choice)


@pytest.mark.parametrize("omit_kind", ["entity", "source"])
def test_model_cannot_reference_omitted_full_context_objects(branches, omit_kind):
    _, ids, sources, full, params = branches
    focused = focus_research_context(full, **params)
    choice = OperationChoice("attack", ids["proof_a"], "test", open_obligation_ids=(ids["a"], ids["b"]), focus_obligation_id=ids["a"])
    primary = next(entity for entity in full.entities if entity["id"] == ids["primary"])
    decision = decision_from_prompt(_research_prompt(focused, primary, choice))
    output = artifact("obstruction", "A critical issue.", "critical_issue_a", [ids["proof_a"]], branch_status="blocked")
    if omit_kind == "entity":
        output["related_entity_ids"].append(ids["lemma_b"])
    else:
        output["source_ids"] = [sources["b"]]
    report = ResearchStepReport.model_validate(step_report(decision, [output], attack_outcome="critical_issue"))
    _validate_step_report(report, full, choice)  # Exists and was legal in full context.
    with pytest.raises(ModelOutputError, match="unknown/out-of-context"):
        _validate_step_report(report, focused, choice)


def test_context_size_and_prompt_exclude_unrelated_branch(branches):
    _, ids, _, full, params = branches
    focused = focus_research_context(full, **params)
    full_json = json.dumps(full.as_model_payload())
    focused_json = json.dumps(focused.as_model_payload())
    assert len(focused_json) < len(full_json)
    choice = OperationChoice("attack", ids["proof_a"], "test", focus_obligation_id=ids["a"])
    primary = next(entity for entity in full.entities if entity["id"] == ids["primary"])
    prompt = _research_prompt(focused, primary, choice)
    assert "UNRELATED_BRANCH_SENTINEL" not in prompt
    assert "UNRELATED_BRANCH_SENTINEL" in full_json
    assert '"context_scope"' in prompt
    print(f"Context regression: {len(full_json)} -> {len(focused_json)} serialized characters")


def test_research_rejects_reference_to_omitted_entity_at_execution_boundary(branches, monkeypatch):
    workstream, ids, _, _, _ = branches
    def respond(decision, _):
        assert decision["operation"] == "attack"
        return step_report(decision, [artifact(
            "obstruction", "A critical gap.", "critical_gap", [ids["proof_a"], ids["lemma_b"]], branch_status="blocked",
        )], attack_outcome="critical_issue")
    provider = DynamicProvider(respond)
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    with pytest.raises(ModelOutputError, match="unknown/out-of-context"):
        research(workstream, max_calls=1, strategy="off")
    assert len(provider.calls) == 1


def test_duplicate_detection_uses_omitted_full_context_branch(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    obligation = add_linked_research_entity(workstream, "OpenQuestion", "Focus on availability", proof_obligation=True)
    with connect() as con:
        omitted = add_entity(con, "Finding", "UNRELATED_DUPLICATE_SENTINEL", body="Previously established communication bound.")
        set_attribute(con, omitted, "research_material_key", "existing_communication_bound")
        link_workstream_entity(con, workstream, omitted, "created")
    provider = DynamicProvider(lambda decision, _: step_report(decision, [artifact(
        "finding", "A paraphrase of the existing bound.", "existing_communication_bound", [obligation],
    )]))
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    outcome = research(workstream, max_calls=1, strategy="off")
    assert "UNRELATED_DUPLICATE_SENTINEL" not in provider.calls[0]["prompt"]
    assert outcome.artifact_ids == ()
    with connect() as con:
        row = con.execute("SELECT duplicate_count,material_progress FROM research_iterations").fetchone()
    assert tuple(row) == (1, 0)


def test_attack_closure_receives_full_context_and_other_obligations_stay_open(branches, monkeypatch):
    workstream, ids, _, full, _ = branches
    import importlib
    controller = importlib.import_module("theory.research")
    original = controller._candidate_can_complete_obligation_after_attack
    checked = []
    def check(context, candidate_id, obligation_id):
        checked.append(candidate_id)
        assert {entity["id"] for entity in context.entities} == {entity["id"] for entity in full.entities}
        return original(context, candidate_id, obligation_id)
    monkeypatch.setattr(controller, "_candidate_can_complete_obligation_after_attack", check)
    provider = DynamicProvider(lambda decision, _: step_report(decision, [], attack_outcome="no_critical_issue"))
    monkeypatch.setattr(controller, "get_provider", lambda _: provider)
    outcome = research(workstream, max_calls=1, strategy="off")
    assert checked == [ids["proof_a"]]
    assert "UNRELATED_BRANCH_SENTINEL" not in provider.calls[0]["prompt"]
    # A has an unresolved blocker; its candidate must not close it. B also stays open.
    assert outcome.stop_reason == "max_calls_exhausted"
    after = for_workstream(workstream)
    assert ids["a"] in controller._open_obligation_ids(after, workstream, ids["primary"])
    assert ids["b"] in controller._open_obligation_ids(after, workstream, ids["primary"])
