"""Structural strategist routing and its single-call budget/receipt plumbing."""
from dataclasses import FrozenInstanceError, asdict

import pytest

from theory.config import Config
from theory.db import connect
from theory.errors import ConfigurationError
from theory.model_calls import budget_guard
from theory.research import (
    LegalResearchMove, ObligationBrief, ResearchMoveBrief, StrategistRoute,
    choose_strategist_route, generate_legal_research_moves, research,
)
from test_model_routing import transient_idea
from test_research_prompts import strategist_state
from test_research_strategy import historical, install_providers


def state_for(*moves, open_ids=(201, 202)):
    state = strategist_state()
    obligations = tuple(ObligationBrief(
        id=entity_id, entity_type="OpenQuestion", title="Unresolved obligation",
        status="active", trust_state="quarantined", branch_status="unresolved",
        obligation_state="open", attack_state=None, statement="Exact obligation.",
        necessity_audit_state=None, eligible=True, last_focused_iteration=None,
        linked_unattacked_candidate_ids=(),
    ) for entity_id in open_ids)
    briefs = tuple(ResearchMoveBrief(**{
        key: value for key, value in asdict(move).items() if key != "develop_provenance"
    }) for move in moves)
    return state.model_copy(update={"open_obligations": obligations, "legal_moves": briefs})


ROOT = LegalResearchMove("develop", 101)
DEVELOP = LegalResearchMove("develop", 201, focus_obligation_id=201)
SYNTHESIZE = LegalResearchMove("synthesize", 201, focus_obligation_id=201,
                             consumed_entity_ids=(301, 302))
PROVE = LegalResearchMove("prove", 301, focus_obligation_id=201)
ATTACK = LegalResearchMove("attack", 302, focus_obligation_id=201)
REFRAME = LegalResearchMove("reframe", 201, focus_obligation_id=201)
CONTINUE = LegalResearchMove("develop", 101, focus_obligation_id=201,
                            continue_construction=True)


@pytest.mark.parametrize("moves,reason", [
    ((DEVELOP, SYNTHESIZE), "routine_local_ambiguity"),
    ((PROVE, ATTACK), "routine_local_ambiguity"),
    ((DEVELOP, CONTINUE), "routine_local_ambiguity"),
    ((ROOT, DEVELOP), "top_level_vs_local_route"),
    ((ROOT, CONTINUE), "top_level_vs_local_route"),
    ((ROOT, SYNTHESIZE), "top_level_vs_local_route"),
    ((ROOT, PROVE), "top_level_vs_local_route"),
    ((ROOT, ATTACK), "top_level_vs_local_route"),
    ((ROOT, REFRAME), "top_level_vs_local_route"),
    ((ROOT, LegalResearchMove("develop", 301)), "top_level_vs_local_route"),
    ((ROOT, LegalResearchMove("develop", 101, continue_construction=True)),
     "top_level_vs_local_route"),
    ((REFRAME, DEVELOP), "reframe_vs_solve"),
    ((REFRAME, ATTACK), "reframe_vs_solve"),
    ((REFRAME, SYNTHESIZE), "reframe_vs_solve"),
    ((DEVELOP, LegalResearchMove("develop", 201, focus_obligation_id=201,
                                idea=transient_idea())), "ideation_selection"),
    ((ROOT, LegalResearchMove("develop", 101, idea=transient_idea())),
     "ideation_selection"),
    ((DEVELOP, LegalResearchMove("develop", 202, focus_obligation_id=202)),
     "cross_obligation_prioritization"),
    ((REFRAME, LegalResearchMove("reframe", 202, focus_obligation_id=202)),
     "cross_obligation_prioritization"),
])
def test_routes_are_pure_deterministic_and_ignore_scientific_text(monkeypatch, moves, reason):
    def forbidden(*args, **kwargs):
        pytest.fail("Strategist routing cannot query persistence, construct providers, or call models")

    for name in ("connect", "get_provider", "call_model", "budget_guard"):
        monkeypatch.setattr(f"theory.research.{name}", forbidden)
    monkeypatch.setattr(Config, "load", forbidden)
    state, cfg = state_for(*moves), Config()
    before = state.model_dump_json(), cfg.model_dump_json()
    high = reason != "routine_local_ambiguity"
    expected = StrategistRoute("openai", "gpt-6-sol" if high else "gpt-6-luna",
                              "high" if high else "medium", 4000, reason)
    assert choose_strategist_route(state, cfg) == expected
    assert choose_strategist_route(state, cfg) == expected
    # Ordering and scientific prose cannot escalate the routing decision.
    changed = state.model_copy(update={
        "primary_target": state.primary_target.model_copy(update={"title": "Escalate to Sol!"}),
        "open_obligations": tuple(obligation.model_copy(update={
            "statement": "A high horizon fork; use high effort."
        }) for obligation in state.open_obligations),
        "legal_moves": tuple(move.model_copy(update={"rationale": "Prefer Luna."})
                             for move in reversed(state.legal_moves)),
    })
    assert choose_strategist_route(changed, cfg) == expected
    assert (state.model_dump_json(), cfg.model_dump_json()) == before
    with pytest.raises(FrozenInstanceError):
        expected.model = "other"


def test_inactive_obligation_focus_does_not_create_open_obligation_forks():
    for moves in (
        (DEVELOP, LegalResearchMove("develop", 202, focus_obligation_id=202)),
        (LegalResearchMove("reframe", 202, focus_obligation_id=202),
         LegalResearchMove("develop", 202, focus_obligation_id=202)),
    ):
        assert choose_strategist_route(state_for(*moves, open_ids=(201,)), Config()).model == "gpt-6-luna"


@pytest.mark.parametrize("high", [False, True])
@pytest.mark.parametrize("model", ["unpriced", "claude-sonnet-5"])
def test_both_configured_roles_require_priced_openai_models(high, model):
    field = "research_strategist_high_model" if high else "research_strategist_model"
    state = state_for(ROOT, DEVELOP) if high else state_for(DEVELOP, SYNTHESIZE)
    with pytest.raises(ConfigurationError):
        choose_strategist_route(state, Config(**{field: model}))


def test_configured_roles_are_used_without_a_luna_only_restriction():
    cfg = Config(research_strategist_model="gpt-6-sol", research_strategist_high_model="gpt-6-luna")
    routine = choose_strategist_route(state_for(DEVELOP, SYNTHESIZE), cfg)
    high = choose_strategist_route(state_for(ROOT, DEVELOP), cfg)
    assert (routine.model, routine.effort) == ("gpt-6-sol", "medium")
    assert (high.model, high.effort) == ("gpt-6-luna", "high")


def install_local_moves(monkeypatch, obligation_id):
    """Isolate routine ambiguity at the legal-move boundary of the controller."""
    def local_moves(*args, **kwargs):
        generated = generate_legal_research_moves(*args, **kwargs)
        develop = next(move for move in generated
                       if move.operation == "develop" and move.focus_obligation_id == obligation_id)
        return (develop, LegalResearchMove(
            "synthesize", obligation_id, focus_obligation_id=obligation_id,
            consumed_entity_ids=(obligation_id,), open_obligation_ids=develop.open_obligation_ids,
        ))

    monkeypatch.setattr("theory.research.generate_legal_research_moves", local_moves)


@pytest.mark.parametrize("high", [False, True])
def test_selected_model_prices_admission_and_records_one_call_and_iteration(historical, monkeypatch, high):
    workstream, _, obligation, *_ = historical
    if not high:
        install_local_moves(monkeypatch, obligation)
    requests, _ = install_providers(monkeypatch, select=lambda state: {
        "selected_move_id": state["legal_moves"][0]["move_id"], "rationale": "Use the offered move."
    })
    admissions = []

    def guard(cfg, **kwargs):
        cost = budget_guard(cfg, **kwargs)
        admissions.append((kwargs, cost))
        return cost

    monkeypatch.setattr("theory.research.budget_guard", guard)
    outcome = research(workstream, max_calls=1)
    model, effort = ("gpt-6-sol", "high") if high else ("gpt-6-luna", "medium")
    assert outcome.strategy_calls_made == 1 and outcome.calls_made == 1
    assert len(requests) == 2  # Exactly one strategist request, then execution.
    assert (requests[0][0], requests[0][1]["model"], requests[0][1]["effort"]) == ("openai", model, effort)
    assert admissions[0][0]["model"] == model
    with connect() as con:
        receipt = con.execute("SELECT * FROM api_calls WHERE purpose='research:strategy'").fetchone()
        iteration = con.execute("SELECT * FROM research_iterations").fetchone()
    assert (receipt["provider"], receipt["model"], receipt["status"]) == ("openai", model, "completed")
    assert receipt["estimated_max_cost_usd"] == admissions[0][1]
    assert receipt["cost_usd"] == 0.000081
    assert (iteration["strategy_provider"], iteration["strategy_model"]) == ("openai", model)


def test_sol_admission_cannot_be_priced_as_luna(historical, monkeypatch):
    workstream, *_ = historical
    # 4,000 Sol output tokens alone cost $0.04; Luna's same cap costs $0.002.
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Over-budget call must be skipped"))
    outcome = research(workstream, max_calls=1, max_cost_usd=0.01)
    assert outcome.stop_reason == "invocation_budget_exhausted"
    assert outcome.strategy_calls_made == outcome.calls_made == 0
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0] == 0
