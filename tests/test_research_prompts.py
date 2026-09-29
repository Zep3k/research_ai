"""Layout and wording regressions for pure research prompt composition."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import pytest

from theory.research import (
    ControllerSummary, OperationChoice, ProblemContractBrief, ResearchEntityBrief,
    ResearchState,
)
from theory.research_prompts import (
    ResearchPromptFacts, build_research_prompt, build_research_sections,
    build_strategist_prompt, build_strategist_sections,
)


CASES = (
    "develop", "focused_develop", "escape_develop", "contract_develop",
    "prove", "open_prove", "focused_prove", "attack", "reframe_attack",
    "synthesize", "primary_synthesize", "reframe",
)
# Captured from the pre-refactor working tree on main, including its existing
# primary-synthesis changes. The word fingerprints allow only these prose edits:
# focused develop/prove and synthesis use controller references in stable text;
# the duplicated attack-precedence summary is removed (the full rule remains).
# Protocol-construction evaluation updates only develop/strategist prose fingerprints:
# prefer constructive continuation and avoid mandatory branch/obligation breadth.
# Case 02B fixes update strategist comparison prose and wrap attack examples in
# an outcome-variant envelope. Decision/context fingerprints remain unchanged.
BASELINE = json.loads(
    (Path(__file__).parent / "fixtures" / "research_prompt_baseline.json").read_text()
)


class PromptContext:
    def __init__(self, offset=0):
        self.payload = {"entities": [{"id": 101 + offset, "body": f"Supplied π premise {offset}"}],
                        "attributes": {}, "relations": []}

    def as_model_payload(self):
        return self.payload


def inputs(case, offset=0):
    target, focus = 101 + offset, 209 + offset
    operation = case.split("_")[-1]
    focused = case.startswith("focused_") or case == "reframe_attack"
    consumed = (307 + offset, 401 + offset) if operation == "synthesize" else ()
    open_ids = (focus,) if focused or case in {"open_prove", "escape_develop"} else ()
    choice = OperationChoice(
        operation, target, f"Bounded decision {offset}", consumed, open_ids,
        focus if focused else None,
    )
    facts = ResearchPromptFacts(
        primary_synthesis=case == "primary_synthesize",
        reframe_attack=case == "reframe_attack",
        human_judgment_allowed=case == "contract_develop",
        problem_contract=(ProblemContractBrief(
            id=503 + offset, entity_type="Definition", title=f"Contract {offset}",
            body=f"Exact supplied π contract {offset}.", trust_state="quarantined",
        ),) if case == "reframe" else (),
    )
    return PromptContext(offset), {"id": target}, choice, facts


def strategist_state(offset=0):
    return ResearchState(
        primary_target=ResearchEntityBrief(
            id=101 + offset, entity_type="ResearchIdea", title=f"Primary goal {offset}",
            status="active", trust_state="quarantined", branch_status=None,
            obligation_state=None, attack_state=None,
        ),
        problem_contract=(), open_obligations=(), blocked_or_terminal_branches=(),
        move_entities=(), legal_moves=(), recent_iterations=(),
        controller_summary=ControllerSummary(
            workstream_id=701 + offset, workstream_status="active",
            completed_iterations=offset, error_iterations=0, eligible_obligation_ids=(),
        ),
    )


def digest_words(text):
    # Order intentionally changes; preserve the full multiset of words (including
    # negatives, IDs, and multiplicity). Explicit semantic checks below complement
    # this coarse baseline check. Examples and controller data are checked exactly.
    return hashlib.sha256(" ".join(sorted(text.split())).encode()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("case", CASES)
def test_execution_layout_is_deterministic_and_prefix_ignores_iteration_data(case):
    context, primary, choice, facts = inputs(case)
    sections = build_research_sections(context, primary, choice, facts=facts)
    changed_context, changed_primary, changed_choice, changed_facts = inputs(case, 1000)
    changed = build_research_sections(
        changed_context, changed_primary, changed_choice, facts=changed_facts,
    )
    assert sections == build_research_sections(context, primary, choice, facts=facts)
    assert sections.render() == build_research_prompt(context, primary, choice, facts=facts)
    assert sections.stable_prefix == changed.stable_prefix
    assert sections.dynamic_context != changed.dynamic_context
    assert sections.render().startswith(sections.stable_prefix)
    assert changed.render().startswith(sections.stable_prefix)
    assert "Supplied π premise" not in sections.stable_prefix
    assert "Exact supplied π contract" not in sections.stable_prefix
    assert sections.render().index("BRANCH STATUS RULES") < sections.render().index("CONTROLLER DECISION")
    with pytest.raises(FrozenInstanceError):
        sections.output_example = "changed"
    with pytest.raises(FrozenInstanceError):
        facts.reframe_attack = True
    # Dict insertion order in graph payloads is immaterial.
    context.payload = dict(reversed(list(context.payload.items())))
    assert sections.render() == build_research_prompt(context, primary, choice, facts=facts)


@pytest.mark.parametrize("case", CASES)
def test_execution_preserves_baseline_wording_decision_context_and_example(case):
    context, primary, choice, facts = inputs(case)
    sections = build_research_sections(context, primary, choice, facts=facts)
    prompt = sections.render()
    assert digest_words(prompt) == BASELINE[case]["words"]
    decision_and_graph = prompt.split("CONTROLLER DECISION\n", 1)[1].split(
        "\n\nReturn ONLY strict JSON", 1,
    )[0]
    assert digest(decision_and_graph) == BASELINE[case]["decision_and_graph"]
    assert digest(sections.output_example) == BASELINE[case]["example"]
    example = json.loads(sections.output_example.split("shape:\n", 1)[1])
    if choice.operation == "attack":
        assert set(example) == {"report"}
        example = example["report"]
    assert example["operation"] == choice.operation
    assert example["target_entity_id"] == choice.target_entity_id
    assert example["consumed_entity_ids"] == list(choice.consumed_entity_ids)


def test_strategist_layout_preserves_instructions_and_state():
    state = strategist_state()
    sections = build_strategist_sections(state)
    assert sections == build_strategist_sections(state)
    assert sections.render() == build_strategist_prompt(state)
    assert sections.stable_prefix == build_strategist_sections(strategist_state(1000)).stable_prefix
    assert digest_words(sections.render()) == BASELINE["strategist"]["words"]
    assert sections.render().split("RESEARCH STATE\n", 1)[1] == state.model_dump_json()
    for rule in (
        "Select exactly one offered legal move_id. Do not invent moves or operation parameters.",
        "A bypass is reversible if its alternative route fails.",
        "Do not invent requirements absent from the contract or automatically prefer reframe.",
        "use scientific context, not a rule to maximize a level or minimize obligation count.",
        "Quarantined artifacts are not assumptions; sourced never means theorem-verified.",
    ):
        assert rule in sections.stable_prefix


@pytest.mark.parametrize(("case", "required", "absent"), (
    ("reframe", ("Do not weaken the contract", "expose EVERY unresolved premise",
                 "not absence of evidence", "Quotes must be exact nonempty substrings",
                 "Return addressed_obligation_ids=[]", "never universal non-necessity or verification"),
     ("ATTACK OUTCOME PRECEDENCE",)),
    ("reframe_attack", ("independent attack on a necessity-audit finding",
                        "Replacement premises are open proof obligations, not assumed facts",
                        "Do not require the original sufficient route to hold"),
     ("For reframe, choose",)),
    ("attack", ("1. CONCRETE DEFECT FOUND", "Use this outcome even if uncertainty also remains",
                "2. NO CONCRETE DEFECT, BUT MATERIAL UNCERTAINTY REMAINS",
                "3. NEITHER A CONCRETE DEFECT NOR MATERIAL UNCERTAINTY",
                "it is never verification"), ("independent attack on a necessity-audit finding",)),
    ("focused_develop", ("focus proof obligation directly", "not escape to unrelated frontier material"),
     ("Develop a genuinely different top-level route",)),
    ("escape_develop", ("Do not assume the current open obligations are necessary",
                        "Do not mark existing obligations resolved merely because a new branch exists"),
     ("focus proof obligation directly",)),
    ("focused_prove", ("Expose every new obligation", "this prove pass must make a concrete transition",
                       "must reference both the prove target and that focus obligation"),
     ("SYNTHESIS OUTPUT RULES",)),
    ("synthesize", ("EVERY artifact emitted by this synthesis operation MUST include EVERY",
                    "successful synthesis artifacts AND failed_approach or obstruction",
                    'Do not call partial progress "addressed"'), ("PRIMARY SYNTHESIS OUTPUT RULES",)),
    ("primary_synthesize", ("hypothesis formation, not obligation closure",
                            "addressed_obligation_ids MUST be []", "never assumed facts"),
     ("attempt to close obligation",)),
))
def test_operation_specific_constraints(case, required, absent):
    context, primary, choice, facts = inputs(case)
    sections = build_research_sections(context, primary, choice, facts=facts)
    prefix = " ".join(sections.stable_prefix.split())
    for text in required:
        assert text in prefix
    for text in absent:
        assert text not in prefix
    assert '"blocked" is legal ONLY for obstruction or failed_approach' in prefix
    assert "ONLY if THIS response also emits" in prefix
    assert "NEVER use quarantined objects as facts" in prefix


def test_section_composition_does_not_mutate_original():
    original = build_strategist_sections(strategist_state())
    changed = replace(original, dynamic_context=("alternate state",))
    assert changed.stable_prefix == original.stable_prefix
    assert changed.render().endswith("alternate state")
    assert "RESEARCH STATE" in original.render()


def test_strategist_applies_same_scientific_criteria_to_every_move():
    prompt = build_strategist_sections(strategist_state()).stable_prefix
    for criterion in (
        "greatest expected contribution toward the exact primary",
        "Evaluate every offered move using the same scientific criteria",
        "directly its mechanism engages the current failure or uncertainty",
        "reduces dependence on unsupported assumptions",
        "few new unresolved premises",
        "concretely its claims or mechanism can be tested",
        "materially distinct it is from already explored routes",
        "preserves all stated contract requirements",
        "compare them symmetrically with all ordinary legal moves",
        "Do not prefer or penalize a move merely because it is an ideation candidate",
        "synthesize, develop, prove, attack, or reframe",
        "not because ideation was triggered",
        "Complementary or unreconciled artifacts alone do not make synthesis preferable",
    ):
        assert criterion in prompt
    for biased_rule in (
        "Give extra strategic weight to an idea", "Prefer synthesis",
        "Prefer root development", "prefer prove/attack", "Prefer a legal construction-continuation",
    ):
        assert biased_rule not in prompt
