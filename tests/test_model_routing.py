import json

import pytest
from typer.testing import CliRunner

from theory.cli import app
from theory.config import Config
from theory.db import connect
from theory.errors import ConfigurationError, ModelOutputError
from theory.model_calls import budget_guard
from theory.providers import estimate_cost, get_model_spec
from theory.research import (
    ModelRoute,
    OperationChoice,
    RESEARCH_MAX_OUTPUT_TOKENS,
    ResearchStepReport,
    choose_model_route,
    research,
)
from theory.research_ideation import CandidateIdea, IdeaUse
from test_research import (
    DynamicProvider,
    add_linked_research_entity,
    artifact,
    init_workspace,
    make_research_workstream,
    step_report,
)


ROUTES = [
    ("develop", None, "openai", "gpt-6-luna", "high", "auto:develop"),
    ("synthesize", 9, "openai", "gpt-6-sol", "high", "auto:synthesize"),
    ("prove", 9, "openai", "gpt-6-sol", "high", "auto:prove"),
    ("attack", None, "openai", "gpt-6-sol", "high", "auto:attack"),
    ("attack", 9, "anthropic", "claude-opus-5-5", "medium", "auto:critical_attack"),
]


def transient_idea() -> CandidateIdea:
    return CandidateIdea(
        idea_id="candidate_route", mechanism="Test a provisional alternate route.",
        exploits=[IdeaUse(entity_id=1, exploitation="Uses the supplied contract.")],
        route_change="Replace the current unsupported premise.",
        main_risk="The replacement may introduce another premise.",
    )


@pytest.mark.parametrize("operation,focus,provider,model,effort,rationale", ROUTES)
def test_exact_routes_are_pure_and_deterministic(
    monkeypatch, operation, focus, provider, model, effort, rationale
):
    def forbidden(*args, **kwargs):
        pytest.fail("Routing cannot construct providers, load state, or make calls")

    monkeypatch.setattr("theory.research.get_provider", forbidden)
    monkeypatch.setattr("theory.research.call_model", forbidden)
    monkeypatch.setattr("theory.research.connect", forbidden)
    choice = OperationChoice(operation, 1, "test", focus_obligation_id=focus)
    cfg = Config()
    expected = ModelRoute(provider, model, effort, 12_000, rationale)
    assert choose_model_route(choice, cfg) == expected
    assert choose_model_route(choice, cfg) == expected
    assert RESEARCH_MAX_OUTPUT_TOKENS == 12_000


def test_selected_idea_develop_uses_its_own_configured_role():
    cfg = Config(research_develop_model="gpt-6-luna",
                 research_idea_develop_model="claude-sonnet-5")
    ordinary = OperationChoice("develop", 1, "Continue ordinary development")
    selected_idea = OperationChoice("develop", 1, "Explore selected idea", idea=transient_idea())
    assert choose_model_route(ordinary, cfg) == ModelRoute(
        "openai", "gpt-6-luna", "high", 12_000, "auto:develop"
    )
    assert choose_model_route(selected_idea, cfg) == ModelRoute(
        "anthropic", "claude-sonnet-5", "high", 12_000, "auto:idea_develop"
    )
    assert choose_model_route(selected_idea, Config()) == ModelRoute(
        "openai", "gpt-6-sol", "high", 12_000, "auto:idea_develop"
    )


def test_idea_origin_continuation_routes_without_replaying_idea():
    cfg = Config(research_develop_model="gpt-6-luna",
                 research_idea_develop_model="claude-sonnet-5")
    continuation = OperationChoice(
        "develop", 1, "Continue unfinished construction",
        continue_construction=True, idea_origin=True,
    )
    ordinary_continuation = OperationChoice(
        "develop", 1, "Continue ordinary construction", continue_construction=True,
    )
    assert continuation.idea is None
    assert choose_model_route(continuation, cfg) == ModelRoute(
        "anthropic", "claude-sonnet-5", "high", 12_000, "auto:idea_develop"
    )
    assert choose_model_route(ordinary_continuation, cfg) == ModelRoute(
        "openai", "gpt-6-luna", "high", 12_000, "auto:develop"
    )


@pytest.mark.parametrize("override,model", [
    ("openai", "gpt-5.6-terra"), ("anthropic", "claude-sonnet-5"),
])
def test_forced_provider_uses_single_configured_model_for_every_operation(override, model):
    cfg = Config(**{f"{override}_model": model})
    choices = [OperationChoice(operation, 1, "test", focus_obligation_id=focus)
               for operation, focus, *_ in ROUTES]
    choices.append(OperationChoice("develop", 1, "test", idea=transient_idea()))
    choices.append(OperationChoice("develop", 1, "test", continue_construction=True,
                                   idea_origin=True))
    for choice in choices:
        route = choose_model_route(
            choice, cfg, provider_override=override,
        )
        assert route == ModelRoute(override, model, "high", 12_000, f"forced:{override}")


def test_role_provider_ownership_comes_from_model_specs():
    cfg = Config(research_develop_model="claude-sonnet-5")
    route = choose_model_route(OperationChoice("develop", 1, "test"), cfg)
    assert route.provider == get_model_spec(cfg.research_develop_model).provider


@pytest.mark.parametrize("model", ["gpt-6-astra", "claude-fable-5-1"])
def test_frontier_models_require_explicit_override(model):
    with pytest.raises(ConfigurationError, match="explicit provider-override"):
        choose_model_route(
            OperationChoice("develop", 1, "test"), Config(research_develop_model=model)
        )
    provider = get_model_spec(model).provider
    cfg = Config(**{f"{provider}_model": model})
    assert choose_model_route(
        OperationChoice("develop", 1, "test"), cfg, provider_override=provider
    ).model == model


@pytest.mark.parametrize("override,cfg_kwargs,error", [
    ("unknown", {}, "Unknown provider override"),
    ("auto", {"research_develop_model": "unpriced"}, "No trusted pricing"),
    ("openai", {"openai_model": "unpriced"}, "No trusted pricing"),
    ("openai", {"openai_model": "claude-sonnet-5"}, "belongs to anthropic"),
])
def test_invalid_routing_fails_before_provider_or_iteration(
    monkeypatch, tmp_path, override, cfg_kwargs, error
):
    init_workspace(monkeypatch, tmp_path)
    Config(**cfg_kwargs).save()
    workstream, _ = make_research_workstream()
    monkeypatch.setattr(
        "theory.research.get_provider", lambda _: pytest.fail("No provider should be created")
    )
    with pytest.raises(ConfigurationError, match=error):
        research(workstream, override, max_calls=1)
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0] == 0


class PricedProvider(DynamicProvider):
    def complete(self, **kwargs):
        result = super().complete(**kwargs)
        result.cost_usd = estimate_cost(kwargs["model"], result.input_tokens, result.output_tokens)
        result.uncached_input_cost_usd = estimate_cost(kwargs["model"], result.input_tokens, 0)
        result.output_cost_usd = estimate_cost(kwargs["model"], 0, result.output_tokens)
        return result


@pytest.mark.parametrize("operation,focus,provider_name,model,effort,rationale", ROUTES)
def test_selected_route_reaches_budget_provider_and_receipts(
    monkeypatch, tmp_path, operation, focus, provider_name, model, effort, rationale
):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream(
        "Theorem" if operation == "attack" and focus is None else "ResearchIdea"
    )
    obligation = None
    if focus is not None:
        obligation = add_linked_research_entity(
            workstream, "OpenQuestion", "Prove the overlap bound", proof_obligation=True
        )
        if operation in {"prove", "attack"}:
            add_linked_research_entity(
                workstream, "Lemma" if operation == "prove" else "ProofAttempt",
                "Candidate for the overlap bound", related_entity_ids=(obligation,),
            )
        else:
            for title in ("Honest signer bound", "Certificate availability"):
                add_linked_research_entity(
                    workstream, "Finding", title, related_entity_ids=(obligation,),
                )

    def responder(decision, _):
        assert decision["operation"] == operation
        if operation == "attack":
            return step_report(decision, [], attack_outcome="inconclusive", unresolved=["Boundary case"])
        if operation == "prove":
            return step_report(decision, [artifact(
                "proof_attempt", "Candidate overlap argument", "overlap_candidate",
                [decision["target_entity_id"], obligation],
            )], addressed=[obligation])
        if operation == "synthesize":
            return step_report(decision, [artifact(
                "obstruction", "Composition still requires a missing boundary estimate.",
                "composition_boundary_gap",
                [obligation, *decision["required_consumed_entity_ids"]],
                branch_status="blocked",
            )])
        return step_report(decision, [artifact(
            "finding", "A counting argument bounds quorum overlap.",
            "quorum_counting_bound", [target],
        )])

    provider = PricedProvider(responder)
    initialized = []
    admissions = []

    def factory(name):
        initialized.append(name)
        return provider

    def guard(cfg, **kwargs):
        value = budget_guard(cfg, **kwargs)
        admissions.append((kwargs, value))
        return value

    monkeypatch.setattr("theory.research.get_provider", factory)
    monkeypatch.setattr("theory.research.budget_guard", guard)
    outcome = research(workstream, max_calls=1, strategy="off")

    assert outcome.calls_made == len(provider.calls) == 1
    assert initialized == [provider_name]
    request = provider.calls[0]
    assert (request["model"], request["effort"], request["max_output_tokens"]) == (model, effort, 12_000)
    expected_response = (
        "FlatAttackReport" if operation == "attack" and provider_name == "anthropic"
        else "ResearchAttackResponse" if operation == "attack" else "ResearchStepReport"
    )
    assert request["response_model"].__name__ == expected_response
    assert admissions[0][0]["model"] == model
    assert admissions[0][0]["max_output_tokens"] == 12_000
    with connect() as con:
        receipt = con.execute("SELECT * FROM api_calls").fetchone()
        assert con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0] == 1
    assert (receipt["provider"], receipt["model"], receipt["purpose"]) == (provider_name, model, f"research:{operation}")
    assert (receipt["input_tokens"], receipt["output_tokens"], receipt["status"]) == (500, 250, "completed")
    assert receipt["cost_usd"] == pytest.approx(estimate_cost(model, 500, 250))
    assert receipt["estimated_max_cost_usd"] == admissions[0][1]


def test_mixed_run_lazily_caches_providers_and_records_actual_review_model(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    obligation = add_linked_research_entity(
        workstream, "OpenQuestion", "Certificate availability", proof_obligation=True
    )
    operations = []

    def responder(decision, _):
        operation = decision["operation"]
        operations.append(operation)
        if operation == "develop":
            return step_report(decision, [artifact(
                "lemma", "Quorum intersection contains an honest signer.",
                "honest_overlap_lemma", [obligation],
            )])
        if operation == "prove":
            return step_report(decision, [artifact(
                "proof_attempt", "Pigeonhole counting constructs the desired certificate.",
                "pigeonhole_certificate_argument", [decision["target_entity_id"], obligation],
            )], addressed=[obligation])
        assert operation == "attack"
        return step_report(decision, [], attack_outcome="no_critical_issue")

    providers = {name: PricedProvider(responder) for name in ("openai", "anthropic")}
    initialized = []

    def factory(name):
        # Anthropic is constructed only after both constructive calls finished.
        initialized.append((name, len(operations)))
        return providers[name]

    monkeypatch.setattr("theory.research.get_provider", factory)
    outcome = research(workstream, max_calls=8, strategy="off")

    assert operations == ["develop", "prove", "attack"]
    assert initialized == [("openai", 0), ("anthropic", 2)]
    assert outcome.calls_made == 3
    assert outcome.stop_reason == "candidate_survived_attack"
    with connect() as con:
        receipts = con.execute("SELECT provider,model,purpose FROM api_calls ORDER BY id").fetchall()
        review = con.execute("SELECT provider,model FROM reviews").fetchone()
        iterations = con.execute("SELECT operation,status FROM research_iterations ORDER BY id").fetchall()
    assert [tuple(row) for row in receipts] == [
        ("openai", "gpt-6-luna", "research:develop"),
        ("openai", "gpt-6-sol", "research:prove"),
        ("anthropic", "claude-opus-5-5", "research:attack"),
    ]
    assert tuple(review) == ("anthropic", "claude-opus-5-5")
    assert [tuple(row) for row in iterations] == [(op, "completed") for op in operations]


def test_openai_only_run_never_constructs_anthropic(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, target = make_research_workstream()
    statements = ["Quorum intersection supplies an honest witness.",
                  "Message batching reduces communication overhead.",
                  "Timeout bounds require partial synchrony.",
                  "Signature aggregation preserves signer attribution."]
    provider = DynamicProvider(lambda decision, n: step_report(decision, [artifact(
        "finding", statements[n - 1], f"distinct_finding_{n}", [target],
    )]))
    initialized = []

    def factory(name):
        initialized.append(name)
        assert name == "openai"
        return provider

    monkeypatch.setattr("theory.research.get_provider", factory)
    # Exercise execution routing alone; generated findings now admit strategic synthesis.
    assert research(workstream, max_calls=4, strategy="off").calls_made == 4
    assert initialized == ["openai"]


@pytest.mark.parametrize("args,expected_provider,expected_model", [
    ([], "openai", "gpt-6-luna"),
    (["--provider", "auto"], "openai", "gpt-6-luna"),
    (["--provider", "openai"], "openai", "gpt-5.6-terra"),
    (["--provider", "anthropic"], "anthropic", "claude-sonnet-5"),
])
def test_cli_auto_default_and_explicit_ablation_models(
    monkeypatch, tmp_path, args, expected_provider, expected_model
):
    init_workspace(monkeypatch, tmp_path)
    Config(openai_model="gpt-5.6-terra", anthropic_model="claude-sonnet-5").save()
    workstream, _ = make_research_workstream()
    provider = DynamicProvider(lambda decision, _: step_report(decision, [artifact(
        "finding", "A counting argument bounds quorum overlap.",
        "quorum_counting_bound", [decision["target_entity_id"]],
    )]))
    names = []

    def factory(name):
        names.append(name)
        return provider

    monkeypatch.setattr("theory.research.get_provider", factory)
    result = CliRunner().invoke(app, ["research", str(workstream), "--max-calls", "1", *args])
    assert result.exit_code == 0, result.output
    assert names == [expected_provider]
    assert len(provider.calls) == 1
    assert provider.calls[0]["model"] == expected_model


def test_old_workspace_config_loads_new_role_defaults_without_rewrite(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    path = tmp_path / ".theory" / "config.json"
    legacy = json.dumps({"monthly_budget_usd": 15, "openai_model": "gpt-5.6-sol",
                         "anthropic_model": "claude-opus-5-5"})
    path.write_text(legacy)
    cfg = Config.load()
    assert cfg.openai_model == "gpt-5.6-sol"
    assert cfg.research_develop_model == "gpt-6-luna"
    assert cfg.research_idea_develop_model == "gpt-6-sol"
    assert cfg.research_synthesize_model == cfg.research_prove_model == "gpt-6-sol"
    assert cfg.research_attack_model == "gpt-6-sol"
    assert cfg.research_critical_attack_model == "claude-opus-5-5"
    assert path.read_text() == legacy


def test_explicit_legacy_attack_model_remains_readable_without_rewrite(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    path = tmp_path / ".theory" / "config.json"
    configured = json.dumps({
        "monthly_budget_usd": 15,
        "openai_model": "gpt-6-sol",
        "anthropic_model": "claude-opus-5-5",
        "research_attack_model": "claude-sonnet-5",
    })
    path.write_text(configured)
    cfg = Config.load()
    assert cfg.research_attack_model == "claude-sonnet-5"
    assert choose_model_route(OperationChoice("attack", 1, "test"), cfg) == ModelRoute(
        "anthropic", "claude-sonnet-5", "high", 12_000, "auto:attack"
    )
    assert path.read_text() == configured


def test_auto_invalid_output_is_a_single_paid_attempt_with_no_escalation(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream("Theorem", "Concrete target")
    # Syntactically valid, but violates the unchanged attack output contract.
    provider = DynamicProvider(lambda decision, _: step_report(decision, [],
        attack_outcome="no_critical_issue", unresolved=["Material uncertainty"]))
    monkeypatch.setattr("theory.research.get_provider", lambda _: provider)
    with pytest.raises(ModelOutputError, match="NoCriticalIssueAttackReport.*could_not_determine"):
        # Isolate automatic execution routing from the competing root-develop move.
        research(workstream, max_calls=8, strategy="off")
    assert len(provider.calls) == 1
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 1
        assert con.execute("SELECT status FROM research_iterations").fetchone()[0] == "error"
