"""Offline strategic-controller regressions. No provider credentials are used."""
import copy
import json
import sqlite3
from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from theory.cli import app
from theory.config import Config
from theory.db import SCHEMA_VERSION, RESEARCH_SELECTION_COLUMNS, SCHEMA, connect, utcnow
from theory.errors import BudgetExceededError, ConfigurationError, ModelOutputError, TheoryError
from theory.graph import set_attribute
from theory.model_calls import budget_guard
from theory.models import ModelResult
from theory.prompts import render_prompt
from theory.research_report import build_research_report
from theory.research import (
    RESEARCH_MAX_OUTPUT_TOKENS,
    LegalResearchMove, OperationChoice, ResearchStepReport, StrategistDecision,
    STRATEGIST_MAX_OUTPUT_TOKENS, _history, _strategist_prompt, build_research_state,
    choose_next_operation, generate_legal_research_moves, research,
    select_research_move, validate_strategist_decision,
)
from theory.research_context import for_workstream
from test_research import (
    add_linked_research_entity, artifact, completed_history, decision_from_prompt,
    init_workspace, make_research_workstream, mark_candidate_attempt, step_report,
)


@pytest.fixture(autouse=True)
def no_real_providers(monkeypatch):
    monkeypatch.setattr("theory.research.get_provider", lambda _: pytest.fail("Real provider forbidden"))


@pytest.fixture
def historical(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, primary = make_research_workstream()
    a = add_linked_research_entity(workstream, "OpenQuestion", "Obligation A: bound the expansion", proof_obligation=True)
    b = add_linked_research_entity(workstream, "OpenQuestion", "Obligation B: justify certificate overlap", proof_obligation=True)
    proof = add_linked_research_entity(workstream, "ProofAttempt", "Pigeonhole certificate overlap argument")
    mark_candidate_attempt(proof, b)
    with connect() as con:
        set_attribute(con, b, "research_obligation_state", "candidate_pending_attack")
        set_attribute(con, b, "research_branch_status", "unresolved")
    return workstream, primary, a, b, proof


def load_moves(workstream, primary_id, history=()):
    context = for_workstream(workstream)
    primary = next(entity for entity in context.entities if entity["id"] == primary_id)
    moves = generate_legal_research_moves(context, workstream, primary, history)
    baseline = choose_next_operation(context, workstream, primary, history)
    assert LegalResearchMove.from_choice(baseline) in moves
    assert moves == generate_legal_research_moves(context, workstream, primary, history)
    assert len({move.move_id for move in moves}) == len(moves)
    return context, primary, moves, baseline


class StrategyProvider:
    def __init__(self, name, requests, select=None, execute=None):
        self.name = name
        self.requests = requests
        self.select = select or (lambda state: {
            "selected_move_id": next((move["move_id"] for move in state["legal_moves"]
                                      if move["operation"] == "attack"), state["legal_moves"][0]["move_id"]),
            "rationale": "Test the concrete candidate before further expansion.",
        })
        self.execute = execute or (lambda decision, _: step_report(
            decision, [] if decision["operation"] == "attack" else [artifact(
                "finding", "Counting bounds the required certificate size.", "certificate_counting",
                [decision["target_entity_id"]],
            )], attack_outcome="inconclusive" if decision["operation"] == "attack" else "not_applicable",
            unresolved=["A boundary estimate remains undecided."] if decision["operation"] == "attack" else [],
        ))
        self.executions = 0

    def complete(self, **kwargs):
        kwargs["prompt"] = render_prompt(kwargs["prompt"])
        self.requests.append((self.name, kwargs))
        if kwargs["response_model"] is StrategistDecision:
            # Planning must never create a running iteration.
            with connect() as con:
                assert con.execute("SELECT COUNT(*) FROM research_iterations WHERE status='running'").fetchone()[0] == 0
            payload = self.select(json.loads(kwargs["prompt"].split("RESEARCH STATE\n", 1)[1]))
        else:
            assert kwargs["response_model"].__name__ in {
                "ResearchStepReport", "ResearchAttackResponse", "FlatAttackReport",
            }
            self.executions += 1
            payload = self.execute(decision_from_prompt(kwargs["prompt"]), self.executions)
        if kwargs["response_model"].__name__ == "ResearchAttackResponse" and isinstance(payload, dict):
            payload = {"report": payload}
        elif kwargs["response_model"].__name__ == "FlatAttackReport" and isinstance(payload, dict):
            payload.pop("attack_outcome", None)
        return ModelResult(
            text=payload if isinstance(payload, str) else json.dumps(payload),
            input_tokens=500, output_tokens=80, uncached_input_tokens=400,
            cache_read_input_tokens=100, cost_usd=0.000081,
            uncached_input_cost_usd=0.00004, cache_read_cost_usd=0.000001,
            output_cost_usd=0.00004,
        )


def install_providers(monkeypatch, **kwargs):
    requests, constructed = [], []
    providers = {name: StrategyProvider(name, requests, **kwargs) for name in ("openai", "anthropic")}

    def factory(name):
        constructed.append(name)
        return providers[name]

    monkeypatch.setattr("theory.research.get_provider", factory)
    return requests, constructed


def test_historical_attack_before_extra_expansion_and_strategy_telemetry(historical, monkeypatch):
    workstream, primary_id, a, b, proof = historical
    context, primary, moves, baseline = load_moves(workstream, primary_id)
    assert [(move.operation, move.target_entity_id, move.focus_obligation_id) for move in moves] == [
        ("develop", a, a), ("attack", proof, b),
        ("reframe", a, a), ("reframe", b, b),
        ("develop", primary_id, None),
    ]
    assert baseline.operation == "develop" and baseline.target_entity_id == a
    decision = StrategistDecision(selected_move_id=moves[1].move_id, rationale="Attack the concrete candidate now.")
    selected = validate_strategist_decision(decision, moves).to_operation_choice()
    assert selected == OperationChoice("attack", proof, moves[1].rationale, open_obligation_ids=(a, b), focus_obligation_id=b)

    requests, constructed = install_providers(monkeypatch)
    admissions = []

    def guard(cfg, **kwargs):
        admissions.append(kwargs)
        return budget_guard(cfg, **kwargs)

    monkeypatch.setattr("theory.research.budget_guard", guard)
    outcome = research(workstream, max_calls=1)
    assert (outcome.calls_made, outcome.strategy_calls_made, outcome.total_api_calls_made) == (1, 1, 2)
    assert len(outcome.iteration_ids) == 1
    assert constructed == ["openai", "anthropic"]
    assert [(name, request["model"], request["effort"], request["max_output_tokens"]) for name, request in requests] == [
        ("openai", "gpt-6-sol", "high", 4000),
        ("anthropic", "claude-opus-5-5", "medium", RESEARCH_MAX_OUTPUT_TOKENS),
    ]
    assert STRATEGIST_MAX_OUTPUT_TOKENS == 4000
    execution = decision_from_prompt(requests[1][1]["prompt"])
    assert execution["operation"] == "attack" and execution["target_entity_id"] == proof
    assert [admission["purpose"] for admission in admissions] == ["research:strategy", "research:attack"]
    with connect() as con:
        receipts = con.execute("SELECT * FROM api_calls ORDER BY id").fetchall()
        row = con.execute("SELECT * FROM research_iterations").fetchone()
        assert con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0] == 1
    receipt = receipts[0]
    assert [(r["provider"], r["model"], r["purpose"]) for r in receipts] == [
        ("openai", "gpt-6-sol", "research:strategy"),
        ("anthropic", "claude-opus-5-5", "research:attack"),
    ]
    assert receipt["workstream_id"] == workstream and receipt["run_id"] is None
    assert receipt["created_at"] and receipt["status"] == "completed"
    assert (receipt["input_tokens"], receipt["output_tokens"], receipt["cache_read_input_tokens"]) == (500, 80, 100)
    assert receipt["uncached_input_tokens"] == 400
    assert receipt["cost_usd"] == 0.000081
    assert admissions[0]["model"] == receipt["model"] == "gpt-6-sol"
    assert receipt["prompt_utf8_bytes"] == len(requests[0][1]["prompt"].encode("utf-8"))
    assert row["selection_mode"] == "strategist"
    assert json.loads(row["legal_move_ids_json"]) == [move.move_id for move in moves]
    assert row["selected_move_id"] == moves[1].move_id
    assert row["selection_rationale"] == "Test the concrete candidate before further expansion."
    assert (row["strategy_provider"], row["strategy_model"], row["focus_obligation_id"]) == ("openai", "gpt-6-sol", b)
    assert outcome.artifact_ids == ()  # The planning rationale is never an artifact.


@pytest.mark.parametrize("provider,strategy", [("auto", "off"), ("openai", "auto"), ("anthropic", "auto")])
def test_baseline_and_explicit_provider_ablations(historical, monkeypatch, provider, strategy):
    monkeypatch.setattr("theory.research.choose_strategist_route",
                        lambda *args: pytest.fail("Ablations must not route a strategist"))
    workstream, primary, *_ = historical
    _, _, moves, baseline = load_moves(workstream, primary)
    requests, constructed = install_providers(monkeypatch)
    outcome = research(workstream, provider, strategy=strategy, max_calls=1)
    assert outcome.calls_made == 1 and outcome.strategy_calls_made == 0
    assert len(requests) == 1 and constructed == ["openai" if provider == "auto" else provider]
    decision = decision_from_prompt(requests[0][1]["prompt"])
    assert (decision["operation"], decision["target_entity_id"]) == (baseline.operation, baseline.target_entity_id)
    if provider != "auto":
        assert requests[0][1]["model"] == getattr(Config(), f"{provider}_model")
        assert requests[0][1]["effort"] == "high"
    with connect() as con:
        row = con.execute("SELECT * FROM research_iterations").fetchone()
    assert row["selection_mode"] == "deterministic_baseline"
    assert row["selected_move_id"] == LegalResearchMove.from_choice(baseline).move_id
    assert row["selection_rationale"] == baseline.rationale
    assert row["strategy_provider"] is row["strategy_model"] is None
    assert f"develop:{primary}:none:none" not in json.loads(row["legal_move_ids_json"])


def test_single_move_costs_zero_strategy_calls(monkeypatch, tmp_path):
    monkeypatch.setattr("theory.research.choose_strategist_route",
                        lambda *args: pytest.fail("Singletons must skip strategist routing"))
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    requests, constructed = install_providers(monkeypatch)
    outcome = research(workstream, max_calls=1)
    assert (outcome.calls_made, outcome.strategy_calls_made, outcome.total_api_calls_made) == (1, 0, 1)
    assert len(requests) == 1 and constructed == ["openai"]
    with connect() as con:
        row = con.execute("SELECT * FROM research_iterations").fetchone()
        assert con.execute("SELECT purpose FROM api_calls").fetchone()[0] == "research:develop"
    assert row["selection_mode"] == "single_legal_move"
    assert json.loads(row["legal_move_ids_json"]) == [row["selected_move_id"]]
    assert row["strategy_provider"] is row["strategy_model"] is None


@pytest.mark.parametrize("payload", [
    {"selected_move_id": "attack:99999:12345", "rationale": "invented"},
    {"selected_move_id": "bad", "rationale": "invented", "operation": "attack"},
    {"selected_move_id": "bad", "rationale": 4},
    "not JSON",
])
def test_invalid_strategy_has_paid_failed_receipt_without_iteration_or_execution(historical, monkeypatch, payload):
    workstream, *_ = historical
    requests, _ = install_providers(monkeypatch, select=lambda _: payload)
    with connect() as con:
        count = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    with pytest.raises(ModelOutputError):
        research(workstream, max_calls=3)
    assert len(requests) == 1
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == count
        assert con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0] == 0
        assert con.execute("SELECT status FROM workstreams").fetchone()[0] == "error"
        receipts = con.execute("SELECT * FROM api_calls").fetchall()
    assert len(receipts) == 1 and receipts[0]["purpose"] == "research:strategy"
    assert receipts[0]["status"] == "failed" and receipts[0]["error_message"]
    assert receipts[0]["input_tokens"] == 500 and receipts[0]["cost_usd"] == 0.000081


def test_strategy_provider_failure_does_not_retry_or_fallback(historical, monkeypatch):
    workstream, *_ = historical

    def fail(_):
        raise TimeoutError("offline timeout")

    requests, _ = install_providers(monkeypatch, select=fail)
    with pytest.raises(TheoryError, match="research:strategy call failed"):
        research(workstream, max_calls=3)
    assert len(requests) == 1
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0] == 0
        assert con.execute("SELECT status FROM api_calls").fetchone()[0] == "failed"
        assert con.execute("SELECT status FROM workstreams").fetchone()[0] == "error"


@pytest.mark.parametrize("field,value", [
    ("operation", "attack"), ("target_entity_id", 12), ("focus_obligation_id", 3),
    ("consumed_entity_ids", [2, 3]), ("model", "gpt-6-sol"), ("provider", "anthropic"),
    ("effort", "high"), ("artifacts", []),
])
def test_strategy_schema_accepts_only_id_and_rationale(field, value):
    assert set(StrategistDecision.model_json_schema()["properties"]) == {"selected_move_id", "rationale"}
    with pytest.raises(ValidationError):
        StrategistDecision.model_validate({"selected_move_id": "develop:1:none:none", "rationale": "reason", field: value})


def test_move_ids_are_transparent_frozen_and_never_fuzzy_matched():
    move = LegalResearchMove("synthesize", 10, 10, (6, 12), (10, 15), "reason")
    assert move.move_id == "synthesize:10:10:6,12"
    assert LegalResearchMove.from_choice(move.to_operation_choice()) == move
    with pytest.raises(FrozenInstanceError):
        move.target_entity_id = 99
    with pytest.raises(ModelOutputError):
        validate_strategist_decision(StrategistDecision(selected_move_id=move.move_id + " ", rationale="reason"), (move,))
    with pytest.raises(TheoryError, match="baseline is not a legal move"):
        select_research_move(baseline_choice=OperationChoice("develop", 1, "reason"), legal_moves=(move,), strategy_enabled=True)


@pytest.mark.parametrize("kind,expected", [
    ("empty", "develop"), ("proof", "attack"), ("synthesis", "synthesize"), ("precise", "prove"),
    ("attacked", "develop"), ("contradicted", "develop"), ("terminal", "develop"), ("consumed", "develop"),
])
def test_local_frontier_precedence_and_exclusions(monkeypatch, tmp_path, kind, expected):
    init_workspace(monkeypatch, tmp_path)
    workstream, primary = make_research_workstream()
    # Baseline remains the empty A branch, even for the inherited terminal-candidate edge case.
    add_linked_research_entity(workstream, "OpenQuestion", "Branch A", proof_obligation=True)
    b = add_linked_research_entity(workstream, "OpenQuestion", "Branch B", proof_obligation=True)
    history = ()
    candidate = None
    if kind in {"proof", "attacked", "contradicted", "terminal", "precise"}:
        candidate = add_linked_research_entity(workstream, "Lemma" if kind == "precise" else "ProofAttempt",
                                             "Concrete bound", related_entity_ids=(b,))
        if kind == "attacked":
            history = (completed_history("attack", candidate),)
        if kind in {"terminal", "contradicted"}:
            with connect() as con:
                if kind == "terminal":
                    set_attribute(con, candidate, "research_branch_status", "refuted")
                else:
                    con.execute("UPDATE entities SET trust_state='contradicted' WHERE id=?", (candidate,))
    if kind in {"synthesis", "consumed"}:
        inputs = tuple(add_linked_research_entity(workstream, "Finding", title, related_entity_ids=(b,))
                       for title in ("First bound", "Second bound"))
        if kind == "consumed":
            history = (completed_history("synthesize", b, consumed_entity_ids=inputs),)
    _, _, moves, _ = load_moves(workstream, primary, history)
    move = next(move for move in moves if move.focus_obligation_id == b)
    assert move.operation == expected
    if expected in {"attack", "prove"}:
        assert move.target_entity_id == candidate
    if expected == "synthesize":
        assert set(move.consumed_entity_ids) == set(inputs)
    if kind in {"attacked", "terminal", "contradicted"}:
        assert candidate not in {move.target_entity_id for move in moves}


def test_newest_proof_only_and_leaf_frontier(historical):
    workstream, primary, a, b, old_proof = historical
    new_proof = add_linked_research_entity(workstream, "ProofAttempt", "New precise candidate")
    mark_candidate_attempt(new_proof, b)
    child = add_linked_research_entity(workstream, "OpenQuestion", "Child of A", proof_obligation=True, related_entity_ids=(a,))
    _, _, moves, _ = load_moves(workstream, primary)
    assert {move.focus_obligation_id for move in moves} == {b, child, None}
    assert [move.target_entity_id for move in moves if move.operation == "attack"] == [new_proof]
    assert old_proof not in {move.target_entity_id for move in moves}


def test_no_obligations_preserves_alternative_attacks_with_root_develop(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, primary = make_research_workstream()
    proofs = [add_linked_research_entity(workstream, "ProofAttempt", f"Candidate {n}") for n in range(3)]
    history = (completed_history("attack", proofs[0]),)
    context, primary_entity, moves, _ = load_moves(workstream, primary, history)
    assert [(move.operation, move.target_entity_id, move.focus_obligation_id) for move in moves] == [
        ("attack", proofs[2], None), ("attack", proofs[1], None),
        ("develop", primary, None),
    ]
    assert generate_legal_research_moves(
        context, workstream, primary_entity, history, strategy_enabled=False,
    ) == moves[:-1]


@pytest.mark.parametrize("entity_type,operation", [("Lemma", "prove"), ("ProofAttempt", "attack")])
def test_no_obligations_strategist_can_develop_past_auxiliary_candidate(
    monkeypatch, tmp_path, entity_type, operation,
):
    init_workspace(monkeypatch, tmp_path)
    workstream, primary_id = make_research_workstream(title="Construct a weak agreement protocol")
    candidate = add_linked_research_entity(workstream, entity_type, "Auxiliary certificate counting bound")
    context, primary, moves, baseline = load_moves(workstream, primary_id)
    assert baseline.operation == operation and baseline.target_entity_id == candidate
    assert baseline.open_obligation_ids == ()
    root_id = f"develop:{primary_id}:none:none"
    assert [move.move_id for move in moves] == [f"{operation}:{candidate}:none:none", root_id]
    assert moves[0] == LegalResearchMove.from_choice(baseline)
    assert moves[1].to_operation_choice() == OperationChoice(
        "develop", primary_id,
        "Explore another top-level route from the supplied problem contract "
        "instead of committing immediately to the current local candidate.",
        develop_provenance="frontier",
    )
    before_entity = next(entity for entity in context.entities if entity["id"] == candidate)
    before_attributes = copy.deepcopy(context.attributes.get(candidate, {}))
    requests, _ = install_providers(
        monkeypatch,
        select=lambda _: {"selected_move_id": root_id, "rationale": "The counting lemma is auxiliary to the main construction."},
        execute=lambda decision, _: step_report(decision, [artifact(
            "protocol_component", "Expose certificate conflicts and abstain on visible conflict.",
            "visible_conflict_protocol", [primary_id],
        )]),
    )
    outcome = research(workstream, max_calls=1)
    assert (outcome.calls_made, outcome.strategy_calls_made, outcome.total_api_calls_made) == (1, 1, 2)
    assert outcome.stop_reason == "max_calls_exhausted"
    assert len(outcome.artifact_ids) == 1
    assert "Compare proving/attacking it against" in requests[0][1]["prompt"]
    execution = requests[1][1]
    assert (execution["model"], execution["effort"], execution["max_output_tokens"]) == ("gpt-6-sol", "high", RESEARCH_MAX_OUTPUT_TOKENS)
    assert decision_from_prompt(execution["prompt"])["target_entity_id"] == primary_id
    assert "Do not assume the current open obligations are necessary." not in execution["prompt"]
    after = for_workstream(workstream)
    assert next(entity for entity in after.entities if entity["id"] == candidate) == before_entity
    assert after.attributes.get(candidate, {}) == before_attributes
    assert after.attributes[outcome.artifact_ids[0]]["research_artifact_type"] == "protocol_component"
    assert choose_next_operation(after, workstream, primary, ()).target_entity_id == candidate


@pytest.mark.parametrize("entity_type,operation", [("Lemma", "prove"), ("ProofAttempt", "attack")])
@pytest.mark.parametrize("provider,strategy", [("auto", "off"), ("openai", "auto"), ("anthropic", "auto")])
def test_no_obligations_ablations_keep_local_candidate(
    monkeypatch, tmp_path, entity_type, operation, provider, strategy,
):
    init_workspace(monkeypatch, tmp_path)
    workstream, primary_id = make_research_workstream()
    candidate = add_linked_research_entity(workstream, entity_type, "Auxiliary bound")
    context, primary, _, baseline = load_moves(workstream, primary_id)
    assert generate_legal_research_moves(
        context, workstream, primary, (), strategy_enabled=False,
    ) == (LegalResearchMove.from_choice(baseline),)
    requests, _ = install_providers(monkeypatch, execute=lambda decision, _: step_report(
        decision, [] if operation == "attack" else [artifact(
            "proof_attempt", "A conditional proof of the auxiliary bound.", "auxiliary_proof", [candidate],
        )], attack_outcome="inconclusive" if operation == "attack" else "not_applicable",
        unresolved=["Boundary case is still uncertain."] if operation == "attack" else [],
    ))
    outcome = research(workstream, provider, strategy=strategy, max_calls=1)
    assert (outcome.calls_made, outcome.strategy_calls_made, len(requests)) == (1, 0, 1)
    decision = decision_from_prompt(requests[0][1]["prompt"])
    assert (decision["operation"], decision["target_entity_id"]) == (operation, candidate)
    with connect() as con:
        row = con.execute("SELECT * FROM research_iterations").fetchone()
    assert row["selection_mode"] == "deterministic_baseline"
    assert json.loads(row["legal_move_ids_json"]) == [LegalResearchMove.from_choice(baseline).move_id]


def test_terminal_candidate_is_absent_from_deterministic_baseline(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    workstream, primary = make_research_workstream()
    candidate = add_linked_research_entity(workstream, "ProofAttempt", "Terminal attempt", branch_status="refuted")
    context = for_workstream(workstream)
    primary_entity = next(entity for entity in context.entities if entity["id"] == primary)
    baseline = choose_next_operation(context, workstream, primary_entity, ())
    assert baseline.target_entity_id == primary
    moves = generate_legal_research_moves(context, workstream, primary_entity, ())
    assert LegalResearchMove.from_choice(baseline) in moves
    assert all(move.target_entity_id != candidate for move in moves)


def test_research_state_is_compact_pure_deterministic_and_records_only_persisted_states(historical, monkeypatch):
    workstream, primary_id, a, b, proof = historical
    terminal = add_linked_research_entity(workstream, "Obstruction", "Blocked branch", branch_status="blocked")
    unrelated = add_linked_research_entity(workstream, "Finding", "Unrelated archival material")
    with connect() as con:
        set_attribute(con, proof, "research_attack_state", "inconclusive")
        con.execute("UPDATE entities SET body=? WHERE id=?", ("LONG_UNRELATED_BODY " * 500, unrelated))
    history = tuple({**completed_history("develop", a), "iteration_number": n,
                     "material_progress": n % 2, "duplicate_count": 2, "attack_outcome": "not_applicable",
                     "stop_reason": None} for n in range(1, 9))
    context, primary, moves, _ = load_moves(workstream, primary_id, history)
    snapshot = copy.deepcopy(context.as_dict())

    def forbidden(*args, **kwargs):
        pytest.fail("Planning must use only already-loaded data")

    for name in ("connect", "get_provider", "call_model", "choose_model_route"):
        monkeypatch.setattr(f"theory.research.{name}", forbidden)
    assert generate_legal_research_moves(context, workstream, primary, history) == moves
    state = build_research_state(context, workstream_id=workstream, primary=primary, history=history, legal_moves=moves)
    again = build_research_state(context, workstream_id=workstream, primary=primary, history=history, legal_moves=moves)
    assert state.model_dump_json() == again.model_dump_json()
    assert context.as_dict() == snapshot
    assert state.primary_target.id == primary_id
    assert {o.id for o in state.open_obligations if o.eligible} == {a, b}
    assert [m.move_id for m in state.legal_moves] == [m.move_id for m in moves]
    assert state.open_obligations[0].last_focused_iteration == 8
    branch_b = next(o for o in state.open_obligations if o.id == b)
    assert (branch_b.obligation_state, branch_b.branch_status, branch_b.linked_unattacked_candidate_ids) == (
        "candidate_pending_attack", "unresolved", (proof,),
    )
    assert state.blocked_or_terminal_branches[0].id == terminal
    assert next(e for e in state.move_entities if e.id == proof).attack_state == "inconclusive"
    assert [row.iteration_number for row in state.recent_iterations] == list(range(3, 9))
    assert [row.material_progress for row in state.recent_iterations] == [True, False] * 3
    assert all(row.duplicate_count == 2 for row in state.recent_iterations)
    prompt = _strategist_prompt(state)
    assert "LONG_UNRELATED_BODY" not in prompt and "Unrelated archival material" not in prompt
    assert len(prompt) < len(json.dumps(context.as_model_payload()))
    assert "Select exactly one offered legal move_id" in prompt


@pytest.mark.parametrize("multiple", [False, True])
def test_call_bounds_and_openai_provider_reuse(monkeypatch, tmp_path, multiple):
    init_workspace(monkeypatch, tmp_path)
    workstream, _ = make_research_workstream()
    if multiple:
        for title in ("Obligation A", "Obligation B"):
            add_linked_research_entity(workstream, "OpenQuestion", title, proof_obligation=True)
    statements = ["Quorum overlap contains an honest signer.", "Message batching reduces packet overhead.", "Timeout requirements depend on network synchrony."]
    requests, constructed = install_providers(monkeypatch, execute=lambda decision, n: step_report(decision, [artifact(
        "obstruction" if decision["operation"] == "synthesize" else "finding",
        statements[n - 1], f"distinct_result_{n}",
        [decision["target_entity_id"], *decision["required_consumed_entity_ids"]],
    )]))
    outcome = research(workstream, max_calls=3)
    assert outcome.calls_made == 3
    # Without obligations, the first two develop results enable primary synthesis.
    assert outcome.strategy_calls_made == (3 if multiple else 1)
    assert outcome.total_api_calls_made == len(requests) <= 6
    assert outcome.strategy_calls_made <= outcome.calls_made <= 3
    assert constructed == ["openai"]  # Strategy and execution share the same instance.
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0] == 3
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == len(requests)


def test_strategy_budget_guard_runs_before_provider(historical):
    workstream, *_ = historical
    Config(monthly_budget_usd=0.000001).save()
    with pytest.raises(BudgetExceededError, match="research:strategy"):
        research(workstream, max_calls=1)
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0] == 0
        assert con.execute("SELECT status FROM workstreams").fetchone()[0] == "active"


@pytest.mark.parametrize("model", ["unpriced", "claude-fable-5-1", "claude-sonnet-5"])
def test_strategy_high_model_must_be_priced_openai(historical, model):
    workstream, *_ = historical
    Config(research_strategist_high_model=model).save()
    with pytest.raises(ConfigurationError):
        research(workstream, max_calls=1)
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 0


def test_legacy_config_default_without_rewrite(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    path = tmp_path / ".theory" / "config.json"
    old = '{"monthly_budget_usd": 10}'
    path.write_text(old)
    assert Config.load().research_strategist_model == "gpt-6-luna"
    assert Config.load().research_strategist_high_model == "gpt-6-sol"
    assert path.read_text() == old


def test_v8_migration_retains_null_strategy_metadata_and_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    old_schema = "\n".join(line for line in SCHEMA.splitlines()
                           if not any(line.strip().startswith(key + " ") for key in RESEARCH_SELECTION_COLUMNS))
    with sqlite3.connect(tmp_path / ".theory" / "research.db") as con:
        con.executescript(old_schema)
        con.execute("INSERT INTO projects VALUES(1,'Legacy','','t')")
        con.execute("INSERT INTO entities(project_id,entity_type,title,body,created_at,updated_at) VALUES(1,'ResearchIdea','Goal','',?,?)", (utcnow(), utcnow()))
        con.execute("INSERT INTO workstreams(project_id,workstream_type,status,goal,created_at,updated_at) VALUES(1,'research','active','Goal',?,?)", (utcnow(), utcnow()))
        con.execute("""INSERT INTO research_iterations(project_id,workstream_id,iteration_number,operation,target_entity_id,rationale,status,created_at)
                       VALUES(1,1,1,'develop',1,'Historical rationale','completed',?)""", (utcnow(),))
        original = con.execute("SELECT * FROM research_iterations").fetchone()
        old_columns = [row[1] for row in con.execute("PRAGMA table_info(research_iterations)")]
        con.execute("PRAGMA user_version=8")
    for _ in range(2):
        with connect() as con:
            assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert con.execute("SELECT name FROM schema_migrations WHERE version=9").fetchone()[0] == "research_strategy_selection"
            row = con.execute("SELECT * FROM research_iterations").fetchone()
            assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        assert tuple(row[key] for key in old_columns) == original
        assert all(row[key] is None for key in RESEARCH_SELECTION_COLUMNS)


@pytest.mark.parametrize("args,strategic", [([], True), (["--strategy", "off"], False),
    (["--provider", "openai"], False), (["--provider", "anthropic"], False)])
def test_cli_strategy_modes_and_call_visibility(historical, monkeypatch, args, strategic):
    workstream, *_ = historical
    requests, _ = install_providers(monkeypatch)
    result = CliRunner().invoke(app, ["research", str(workstream), "--max-calls", "1", *args])
    assert result.exit_code == 0, result.output
    normalized = " ".join(result.output.split())
    assert "1 execution call(s)" in normalized
    assert len(requests) == (2 if strategic else 1)
    if strategic:
        assert "1 strategist call(s)" in normalized and "2 total API call(s)" in normalized
        assert "Calls: 1 strategy / 1 execution" in normalized
    else:
        assert "Calls: 0 strategy / 1 execution" in normalized


def test_local_precedence_across_simultaneous_candidates(historical):
    workstream, primary, _, b, proof = historical
    lemma = add_linked_research_entity(workstream, "Lemma", "Precise lemma", related_entity_ids=(b,))
    add_linked_research_entity(workstream, "Finding", "Auxiliary estimate", related_entity_ids=(b,))
    _, _, moves, _ = load_moves(workstream, primary)
    assert moves[1].operation == "attack" and moves[1].target_entity_id == proof
    history = (completed_history("attack", proof),)
    _, _, moves, _ = load_moves(workstream, primary, history)
    assert moves[1].operation == "synthesize"
    history += (completed_history("synthesize", b, consumed_entity_ids=moves[1].consumed_entity_ids),)
    _, _, moves, _ = load_moves(workstream, primary, history)
    assert moves[1].operation == "prove" and moves[1].target_entity_id == lemma
    history += (completed_history("prove", lemma),)
    _, _, moves, _ = load_moves(workstream, primary, history)
    assert moves[1].operation == "develop"


def test_strategy_execution_still_rejects_omitted_branch_references(historical, monkeypatch):
    workstream, _, a, _, proof = historical
    requests, _ = install_providers(monkeypatch, execute=lambda decision, _: step_report(decision, [artifact(
        "obstruction", "Concrete gap", "concrete_gap", [proof, a], branch_status="blocked",
    )], attack_outcome="critical_issue"))
    with pytest.raises(ModelOutputError, match="unknown/out-of-context"):
        research(workstream, max_calls=1)
    assert len(requests) == 2
    assert "Obligation A: bound the expansion" not in requests[1][1]["prompt"]
    with connect() as con:
        row = con.execute("SELECT status,selection_mode FROM research_iterations").fetchone()
    assert tuple(row) == ("error", "strategist")


@pytest.mark.parametrize("complete_link", [False, True])
def test_strategy_execution_closure_still_uses_full_context(historical, monkeypatch, complete_link):
    import importlib
    controller = importlib.import_module("theory.research")
    workstream, _, a, b, proof = historical
    if complete_link:
        with connect() as con:
            set_attribute(con, proof, "related_entity_ids", json.dumps([b]))
    original = controller._candidate_can_complete_obligation_after_attack
    checked = []

    def check(context, candidate_id, obligation_id):
        assert {a, b, proof} <= {entity["id"] for entity in context.entities}
        checked.append((candidate_id, obligation_id))
        return original(context, candidate_id, obligation_id)

    monkeypatch.setattr(controller, "_candidate_can_complete_obligation_after_attack", check)
    requests, _ = install_providers(monkeypatch, execute=lambda decision, _: step_report(
        decision, [], attack_outcome="no_critical_issue",
    ))
    outcome = research(workstream, max_calls=1)
    assert checked == [(proof, b)]
    assert outcome.stop_reason == "max_calls_exhausted"  # A still remains open.
    assert "Obligation A: bound the expansion" not in requests[1][1]["prompt"]
    context = for_workstream(workstream)
    assert context.attributes[b]["research_obligation_state"] == ("resolved_candidate" if complete_link else "open")
    if complete_link:
        assert context.attributes[b]["research_surviving_candidate_id"] == str(proof)
    else:
        assert "research_surviving_candidate_id" not in context.attributes[b]
    assert context.attributes[proof]["research_attack_state"] == "survived_attack"


def test_strategy_execution_duplicates_still_use_full_context(historical, monkeypatch):
    workstream, _, a, _, proof = historical
    previous = add_linked_research_entity(workstream, "Finding", "Elsewhere in branch A", related_entity_ids=(a,))
    with connect() as con:
        set_attribute(con, previous, "research_material_key", "existing_material")
    requests, _ = install_providers(monkeypatch, execute=lambda decision, _: step_report(decision, [artifact(
        "finding", "A paraphrase of existing material", "existing_material", [proof],
    )], attack_outcome="inconclusive", unresolved=["Still missing a boundary case."]))
    outcome = research(workstream, max_calls=1)
    assert outcome.artifact_ids == ()
    assert "Elsewhere in branch A" not in requests[1][1]["prompt"]
    with connect() as con:
        row = con.execute("SELECT duplicate_count,material_progress FROM research_iterations").fetchone()
    # The duplicate is still rejected; the inconclusive attack itself is validation.
    assert tuple(row) == (1, 1)


def test_strategy_cost_is_spent_before_execution_budget_admission(historical, monkeypatch):
    workstream, *_ = historical
    Config(monthly_budget_usd=0.07).save()  # Admits Sol strategy, not the Opus execution cap.
    requests, _ = install_providers(monkeypatch)
    with pytest.raises(BudgetExceededError, match="research:attack"):
        research(workstream, max_calls=1)
    assert len(requests) == 1
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM research_iterations").fetchone()[0] == 0
        assert con.execute("SELECT status FROM workstreams").fetchone()[0] == "active"
        receipt = con.execute("SELECT purpose,cost_usd FROM api_calls").fetchone()
    assert tuple(receipt) == ("research:strategy", 0.000081)


def test_recent_error_is_absent_from_scientific_state_and_fairness(historical):
    workstream, primary_id, a, b, proof = historical
    history = ({"iteration_number": 1, "status": "error", "operation": "attack",
                "target_entity_id": proof, "material_progress": 0, "duplicate_count": 0,
                "attack_outcome": "not_applicable", "stop_reason": None},)
    context, primary, moves, baseline = load_moves(workstream, primary_id, history)
    assert baseline.target_entity_id == a
    state = build_research_state(context, workstream_id=workstream, primary=primary, history=history, legal_moves=moves)
    assert state.recent_iterations == ()
    assert "error_iterations" not in state.model_dump_json()
    assert all(obligation.last_focused_iteration is None for obligation in state.open_obligations)


def test_failed_attack_calls_do_not_change_scientific_strategy_but_success_does(
    historical, monkeypatch,
):
    workstream, primary_id, _, _, proof = historical
    scientific_states = []
    execution_attempts = 0

    def select(state):
        scientific_states.append(state)
        attack = next(move for move in state["legal_moves"]
                      if move["operation"] == "attack" and move["target_entity_id"] == proof)
        return {"selected_move_id": attack["move_id"],
                "rationale": "Test the same concrete proof candidate."}

    def execute(decision, _):
        nonlocal execution_attempts
        execution_attempts += 1
        if execution_attempts <= 2:
            raise TimeoutError("offline attack transport failure")
        return step_report(decision, [], attack_outcome="inconclusive",
                           unresolved=["The boundary estimate remains undecided."])

    requests, _ = install_providers(monkeypatch, select=select, execute=execute)
    context, primary, moves, baseline = load_moves(workstream, primary_id)
    initial = build_research_state(context, workstream_id=workstream, primary=primary,
                                   history=(), legal_moves=moves)
    assert baseline.target_entity_id != proof
    for expected_error_count in (1, 2):
        with pytest.raises(TheoryError, match="offline attack transport failure"):
            research(workstream, max_calls=1)
        with connect() as con:
            rows = [dict(row) for row in con.execute(
                "SELECT * FROM research_iterations WHERE workstream_id=? ORDER BY id",
                (workstream,),
            )]
            assert len(rows) == expected_error_count
            assert all(row["status"] == "error" and row["material_progress"] == 0
                       and row["resolution_progress"] is None for row in rows)
            con.execute("UPDATE workstreams SET status='active' WHERE id=?", (workstream,))
        context, primary, moves, next_baseline = load_moves(workstream, primary_id, _history(workstream))
        state = build_research_state(context, workstream_id=workstream, primary=primary,
                                     history=_history(workstream), legal_moves=moves)
        assert state.model_dump() == initial.model_dump()
        assert next_baseline == baseline
        assert state.recent_iterations == ()
        assert next(e for e in context.entities if e["id"] == proof)["status"] == "active"
        assert context.attributes[proof].get("research_attack_state") is None
        assert "error_iterations" not in state.model_dump_json()
        assert scientific_states[-1] == json.loads(initial.model_dump_json())

    audit = build_research_report(workstream)
    assert len(audit.recent_iterations) == 2
    assert all(row.status == "error" for row in audit.recent_iterations)
    cli = CliRunner().invoke(app, ["workstream", "show", str(workstream)])
    assert cli.exit_code == 0, cli.output
    assert "offline attack transport failure" in cli.output
    with connect() as con:
        assert con.execute("SELECT COUNT(*) FROM api_calls WHERE status='failed'").fetchone()[0] == 2

    result = research(workstream, max_calls=1)
    assert result.calls_made == 1 and execution_attempts == 3
    assert scientific_states[0] == scientific_states[1] == scientific_states[2]
    assert [request[1]["response_model"] is StrategistDecision for request in requests].count(True) == 3
    with connect() as con:
        con.execute("UPDATE workstreams SET status='active' WHERE id=?", (workstream,))
    context, primary, moves, _ = load_moves(workstream, primary_id, _history(workstream))
    after_success = build_research_state(context, workstream_id=workstream, primary=primary,
                                          history=_history(workstream), legal_moves=moves)
    assert after_success.model_dump() != initial.model_dump()
    assert after_success.controller_summary.completed_iterations == 1
    assert len(after_success.recent_iterations) == 1
    assert after_success.recent_iterations[0].attack_outcome == "inconclusive"
    assert context.attributes[proof]["research_attack_state"] == "inconclusive"


def test_frontier_develop_and_consecutive_continuations_keep_route_after_resume(
    historical, monkeypatch,
):
    workstream, primary_id, _, _, _ = historical

    def select(state):
        assert "develop_provenance" not in json.dumps(state)
        frontier = next(move for move in state["legal_moves"]
                        if move["operation"] == "develop"
                        and move["target_entity_id"] == primary_id
                        and not move["continue_construction"] and move["idea"] is None)
        return {"selected_move_id": frontier["move_id"],
                "rationale": "Try the offered different top-level route."}

    def execute(decision, number):
        return step_report(decision, [artifact(
            "protocol_component", f"Unfinished route component {number}.",
            f"frontier_component_{number}", [decision["target_entity_id"]],
            branch_status="unresolved",
        )])

    requests, _ = install_providers(monkeypatch, select=select, execute=execute)
    first = research(workstream, max_calls=1)
    assert first.calls_made == 1
    for _ in range(2):
        with connect() as con:
            con.execute("UPDATE workstreams SET status='active' WHERE id=?", (workstream,))
        assert research(workstream, strategy="off", max_calls=1).calls_made == 1

    executions = [kwargs for _, kwargs in requests
                  if kwargs["response_model"] is ResearchStepReport]
    assert [request["model"] for request in executions] == ["gpt-6-sol"] * 3
    assert all(request["max_output_tokens"] == RESEARCH_MAX_OUTPUT_TOKENS for request in executions)
    assert all("research_develop_provenance" not in request["prompt"] for request in executions)
    with connect() as con:
        rows = [dict(row) for row in con.execute(
            "SELECT * FROM research_iterations WHERE workstream_id=? ORDER BY iteration_number",
            (workstream,),
        )]
    assert [row["develop_provenance"] for row in rows] == ["frontier"] * 3
    assert [row["selected_move_id"].endswith(":continue") for row in rows] == [
        False, True, True,
    ]
    context = for_workstream(workstream)
    assert all(context.attributes[entity_id]["research_develop_provenance"] == "frontier"
               for row in rows for entity_id in json.loads(row["artifact_ids_json"]))


def test_leaving_frontier_route_restores_ordinary_develop_routing(historical, monkeypatch):
    workstream, primary_id, obligation_id, _, _ = historical
    selections = 0

    def select(state):
        nonlocal selections
        selections += 1
        if selections == 1:
            chosen = next(move for move in state["legal_moves"]
                          if move["operation"] == "develop"
                          and move["target_entity_id"] == primary_id
                          and not move["continue_construction"])
        elif selections == 2:
            chosen = next(move for move in state["legal_moves"]
                          if move["operation"] == "develop"
                          and move["target_entity_id"] == obligation_id
                          and not move["continue_construction"])
        else:
            chosen = next(move for move in state["legal_moves"]
                          if move["operation"] == "develop"
                          and move["target_entity_id"] == obligation_id
                          and move["continue_construction"])
        return {"selected_move_id": chosen["move_id"],
                "rationale": "Select the offered route for this bounded step."}

    def execute(decision, number):
        return step_report(decision, [artifact(
            "protocol_component", f"Unfinished distinct route component {number}.",
            f"distinct_route_component_{number}", [decision["target_entity_id"]],
            branch_status="unresolved",
        )])

    requests, _ = install_providers(monkeypatch, select=select, execute=execute)
    assert research(workstream, max_calls=3).calls_made == 3
    executions = [kwargs for _, kwargs in requests
                  if kwargs["response_model"] is ResearchStepReport]
    assert [request["model"] for request in executions] == [
        "gpt-6-sol", "gpt-6-luna", "gpt-6-luna",
    ]
    with connect() as con:
        rows = [dict(row) for row in con.execute(
            "SELECT target_entity_id,develop_provenance,selected_move_id "
            "FROM research_iterations WHERE workstream_id=? ORDER BY iteration_number",
            (workstream,),
        )]
    assert [(row["target_entity_id"], row["develop_provenance"]) for row in rows] == [
        (primary_id, "frontier"), (obligation_id, "ordinary"), (obligation_id, "ordinary"),
    ]
    assert rows[2]["selected_move_id"].endswith(":continue")
