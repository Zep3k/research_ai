from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .config import Config
from .db import connect, utcnow
from .errors import ConfigurationError, ModelOutputError, TheoryError
from .graph import (
    add_entity,
    add_relation,
    add_review,
    link_workstream_entity,
    set_attribute,
    set_workstream_status,
)
from .jsonutil import parse_json_model
from .model_calls import InvocationBudget, budget_guard, call_model
from .paths import STATE_DIR
from .providers import Provider, get_model_spec, get_provider
from .research_context import ResearchContext, focus_research_context, for_workstream
from .research_prompts import (
    PromptSections, ResearchPromptFacts, build_research_sections, build_strategist_sections,
    build_strategist_prompt as _strategist_prompt,
)
from .research_ideation import (
    CandidateIdea, IdeaBatch, IDEATION_MAX_OUTPUT_TOKENS, build_ideation_prompt,
    choose_ideation_trigger, previous_ideations, record_idea_selection, validate_ideas,
)
from .research_progress import (
    ProgressEvent, ProgressEventKind, ProgressLevel, ProgressRecord, progress_event_kinds,
)
from .research_routes import (
    CONSTRUCTION_TYPES, ROUTE_IDS, STARTED_AT, record_construction_routes,
    live_construction_route_ids, persisted_id_set as _stored_id_set,
    entity_has_closing_relation,
    route_entity_is_live as _route_entity_is_live,
)


STRATEGIST_MAX_OUTPUT_TOKENS = 4000
RESEARCH_MAX_OUTPUT_TOKENS = 16_000
MAX_CONTROLLER_CALLS = 20
MAX_CONSTRUCTIVE_CONTINUATIONS = 3
INTERRUPTED_ITERATION_ERROR = (
    "Interrupted before controller completion; reconciled as abandoned before a new "
    "research invocation."
)
INTERRUPTED_API_CALL_ERROR = (
    "Interrupted provider request; reconciled as abandoned before a new research invocation."
)
OPERATIONS = ("develop", "attack", "synthesize", "prove", "reframe")
ROOT_TARGET_TYPES = {
    "Conjecture",
    "Finding",
    "Lemma",
    "OpenQuestion",
    "ProofAttempt",
    "ResearchIdea",
    "Technique",
    "Theorem",
}
ARTIFACT_ENTITY_TYPES = {
    "consequence": "Finding",
    "lemma": "Lemma",
    "protocol_component": "Technique",
    "parameter_analysis": "Finding",
    "proof_obligation": "OpenQuestion",
    "open_question": "OpenQuestion",
    "proof_attempt": "ProofAttempt",
    "synthesis": "Finding",
    "counterexample": "Counterexample",
    "obstruction": "Obstruction",
    "failed_approach": "FailedApproach",
    "finding": "Finding",
}
CRITICAL_ARTIFACT_TYPES = {"counterexample", "obstruction", "failed_approach"}
PROOF_ARTIFACT_TYPES = {"lemma", "proof_attempt"}
TERMINAL_BRANCH_STATES = {"blocked", "failed", "refuted"}
LIVE_BRANCH_STATES = {"promising", "unresolved"}
INACTIVE_OBLIGATION_STATES = {"blocked", "resolved_candidate", "bypassed", "unnecessary"}
PRECISE_ENTITY_TYPES = {"Theorem", "Lemma", "ProofAttempt"}
SYNTHESIS_INPUT_TYPES = {
    "Assumption",
    "Conjecture",
    "Definition",
    "Finding",
    "Lemma",
    "Model",
    "ProofAttempt",
    "Technique",
    "Theorem",
}
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "under",
    "with",
}


def _strip_nonempty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("text cannot be blank")
    return stripped


def _positive_unique_ids(values: list[int]) -> list[int]:
    if any(value <= 0 for value in values):
        raise ValueError("IDs must be positive")
    if len(values) != len(set(values)):
        raise ValueError("IDs must not be duplicated")
    return values


ArtifactType = Literal[
    "consequence",
    "lemma",
    "protocol_component",
    "parameter_analysis",
    "proof_obligation",
    "open_question",
    "proof_attempt",
    "synthesis",
    "counterexample",
    "obstruction",
    "failed_approach",
    "finding",
]
GeneralArtifactType = Literal[
    "consequence",
    "lemma",
    "protocol_component",
    "parameter_analysis",
    "proof_obligation",
    "open_question",
    "proof_attempt",
    "synthesis",
    "counterexample",
    "finding",
]
BranchStatus = Literal[
    "promising", "blocked", "failed", "refuted", "unresolved"
] | None


class ResearchArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    artifact_type: ArtifactType
    statement: str = Field(min_length=1, max_length=2_000)
    reasoning_summary: str = Field(min_length=1, max_length=5_000)
    material_key: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{2,79}$")
    epistemic_status: Literal["inference", "speculation", "unresolved"]
    related_entity_ids: list[int] = Field(min_length=1, max_length=24)
    refutes_entity_ids: list[int] = Field(default_factory=list, max_length=24)
    source_ids: list[int] = Field(default_factory=list, max_length=20)
    branch_status: BranchStatus

    @field_validator("statement", "reasoning_summary")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        return _strip_nonempty(value)

    @field_validator("related_entity_ids", "refutes_entity_ids", "source_ids")
    @classmethod
    def positive_unique_ids(cls, values: list[int]) -> list[int]:
        return _positive_unique_ids(values)

    @model_validator(mode="after")
    def validate_branch_status(self) -> "ResearchArtifact":
        if self.refutes_entity_ids and self.artifact_type not in CRITICAL_ARTIFACT_TYPES:
            raise ValueError("Only negative artifacts may explicitly refute existing entities")
        if self.branch_status in {"failed", "refuted"} and self.artifact_type != "failed_approach":
            raise ValueError("failed/refuted branches must be failed_approach artifacts")
        if self.branch_status == "blocked" and self.artifact_type not in {
            "obstruction",
            "failed_approach",
        }:
            raise ValueError("blocked branches must be obstruction or failed_approach artifacts")
        return self


class GeneralResearchArtifact(ResearchArtifact):
    artifact_type: GeneralArtifactType
    branch_status: Literal["promising", "unresolved"] | None


class ObstructionResearchArtifact(ResearchArtifact):
    artifact_type: Literal["obstruction"]
    branch_status: Literal["promising", "blocked", "unresolved"] | None


class FailedApproachResearchArtifact(ResearchArtifact):
    artifact_type: Literal["failed_approach"]
    branch_status: BranchStatus


ResearchArtifactVariant = (
    GeneralResearchArtifact
    | ObstructionResearchArtifact
    | FailedApproachResearchArtifact
)


class ContractClause(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    entity_id: int = Field(gt=0)
    quote: str = Field(min_length=1)


class NecessityAudit(BaseModel):
    """Inspectable route argument, not a model-supplied controller verdict."""
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    parent_requirement: ContractClause
    contract_clauses: list[ContractClause] = Field(min_length=1, max_length=24)
    argument: str = Field(min_length=1, max_length=5000)
    replacement_obligation_keys: list[str] = Field(max_length=4)

    @field_validator("argument")
    @classmethod
    def nonempty_argument(cls, value: str) -> str:
        return _strip_nonempty(value)


class ResearchStepReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    operation: Literal["develop", "attack", "synthesize", "prove", "reframe"]
    target_entity_id: int = Field(gt=0)
    summary: str = Field(min_length=1, max_length=4_000)
    artifacts: list[ResearchArtifactVariant] = Field(default_factory=list, max_length=4)
    consumed_entity_ids: list[int] = Field(default_factory=list, max_length=8)
    addressed_obligation_ids: list[int] = Field(default_factory=list, max_length=12)
    attack_outcome: Literal[
        "not_applicable", "critical_issue", "no_critical_issue", "inconclusive"
    ]
    could_not_determine: list[str] = Field(default_factory=list, max_length=12)
    necessity_outcome: Literal["not_applicable", "required_on_current_routes", "alternative_route_found", "inconclusive"] = "not_applicable"
    necessity_contract_entity_ids: list[int] = Field(default_factory=list, max_length=24)
    necessity_audit: NecessityAudit | None = None
    human_judgment_required: bool
    human_judgment_reason: str | None

    @field_validator("summary")
    @classmethod
    def strip_summary(cls, value: str) -> str:
        return _strip_nonempty(value)

    @field_validator("consumed_entity_ids", "addressed_obligation_ids", "necessity_contract_entity_ids")
    @classmethod
    def positive_unique_ids(cls, values: list[int]) -> list[int]:
        return _positive_unique_ids(values)

    @field_validator("could_not_determine")
    @classmethod
    def clean_unresolved(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("could_not_determine entries cannot be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_human_judgment_reason(self) -> "ResearchStepReport":
        if self.human_judgment_required:
            if self.human_judgment_reason is None or not self.human_judgment_reason.strip():
                raise ValueError("human_judgment_reason is required when human judgment is needed")
            self.human_judgment_reason = self.human_judgment_reason.strip()
        elif self.human_judgment_reason is not None:
            raise ValueError("human_judgment_reason must be null when judgment is not required")
        return self


class NoncriticalAttackArtifact(GeneralResearchArtifact):
    artifact_type: Literal[
        "consequence", "lemma", "protocol_component", "parameter_analysis",
        "proof_obligation", "open_question", "proof_attempt", "synthesis", "finding",
    ]


class CriticalAttackReport(ResearchStepReport):
    operation: Literal["attack"]
    attack_outcome: Literal["critical_issue"]
    artifacts: list[ResearchArtifactVariant] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def require_critical_artifact(self) -> "CriticalAttackReport":
        if not any(a.artifact_type in CRITICAL_ARTIFACT_TYPES for a in self.artifacts):
            raise ValueError(
                "critical_issue requires a concrete counterexample, obstruction, or failed approach."
            )
        return self


class InconclusiveAttackReport(ResearchStepReport):
    operation: Literal["attack"]
    attack_outcome: Literal["inconclusive"]
    artifacts: list[NoncriticalAttackArtifact] = Field(default_factory=list, max_length=4)
    could_not_determine: list[Annotated[str, Field(pattern=r"\S")]] = Field(min_length=1, max_length=12)


class NoCriticalIssueAttackReport(ResearchStepReport):
    operation: Literal["attack"]
    attack_outcome: Literal["no_critical_issue"]
    artifacts: list[NoncriticalAttackArtifact] = Field(default_factory=list, max_length=4)
    could_not_determine: list[str] = Field(max_length=0)


class ResearchAttackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    # Literal outcome tags discriminate the variants using provider-supported anyOf.
    # Concrete critical-artifact presence is also checked locally; no output repair.
    report: CriticalAttackReport | InconclusiveAttackReport | NoCriticalIssueAttackReport


class FlatAttackReport(BaseModel):
    """Anthropic attack evidence; the controller classifies its outcome locally."""

    model_config = ConfigDict(extra="forbid", strict=True)

    operation: Literal["attack"]
    target_entity_id: int = Field(gt=0)
    summary: str = Field(min_length=1, max_length=4_000)
    artifacts: list[ResearchArtifact] = Field(default_factory=list, max_length=4)
    consumed_entity_ids: list[int] = Field(
        default_factory=list, max_length=0,
    )
    addressed_obligation_ids: list[int] = Field(
        default_factory=list, max_length=0,
    )
    could_not_determine: list[str] = Field(default_factory=list, max_length=12)
    necessity_outcome: Literal["not_applicable"] = "not_applicable"
    necessity_contract_entity_ids: list[int] = Field(
        default_factory=list, max_length=0,
    )
    necessity_audit: Literal[None] = None
    human_judgment_required: Literal[False]
    human_judgment_reason: Literal[None]

    @field_validator("summary")
    @classmethod
    def strip_summary(cls, value: str) -> str:
        return _strip_nonempty(value)

    @field_validator("could_not_determine")
    @classmethod
    def clean_unresolved(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("could_not_determine entries cannot be blank")
        return cleaned


def _execution_response_model(operation: str, provider: str) -> type[BaseModel]:
    if operation != "attack":
        return ResearchStepReport
    if provider == "openai":
        return ResearchAttackResponse
    if provider == "anthropic":
        return FlatAttackReport
    raise ConfigurationError(f"Unknown attack response provider: {provider}")


def _parse_execution_report(text: str, response_model: type[BaseModel]) -> ResearchStepReport:
    if response_model is ResearchAttackResponse:
        return parse_json_model(text, ResearchAttackResponse).report
    if response_model is FlatAttackReport:
        try:
            evidence = FlatAttackReport.model_validate_json(text)
            if any(artifact.artifact_type in CRITICAL_ARTIFACT_TYPES
                   for artifact in evidence.artifacts):
                outcome = "critical_issue"
            elif evidence.could_not_determine:
                outcome = "inconclusive"
            else:
                outcome = "no_critical_issue"
            return ResearchStepReport.model_validate({
                **evidence.model_dump(), "attack_outcome": outcome,
            })
        except ValidationError as exc:
            raise ModelOutputError(f"Model JSON did not match FlatAttackReport: {exc}") from exc
    return parse_json_model(text, response_model)


DevelopProvenance = Literal["ordinary", "idea", "frontier"]


@dataclass(frozen=True)
class OperationChoice:
    operation: str
    target_entity_id: int
    rationale: str
    consumed_entity_ids: tuple[int, ...] = ()
    open_obligation_ids: tuple[int, ...] = ()
    focus_obligation_id: int | None = None
    continue_construction: bool = False
    idea: CandidateIdea | None = None
    ideation_call_id: int | None = None
    develop_provenance: DevelopProvenance = "ordinary"

    def __post_init__(self) -> None:
        if self.operation == "develop" and self.idea is not None:
            object.__setattr__(self, "develop_provenance", "idea")


@dataclass(frozen=True)
class LegalResearchMove:
    operation: Literal["develop", "attack", "synthesize", "prove", "reframe"]
    target_entity_id: int
    focus_obligation_id: int | None = None
    consumed_entity_ids: tuple[int, ...] = ()
    open_obligation_ids: tuple[int, ...] = ()
    rationale: str = ""
    continue_construction: bool = False
    idea: CandidateIdea | None = None
    ideation_call_id: int | None = None
    develop_provenance: DevelopProvenance = "ordinary"
    move_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.operation == "develop" and self.idea is not None:
            object.__setattr__(self, "develop_provenance", "idea")
        # Open obligations are shared by every move in a legal set. The operation,
        # target, focus, exact ordered inputs and construction intent distinguish moves.
        focus = self.focus_obligation_id if self.focus_obligation_id is not None else "none"
        inputs = ",".join(map(str, self.consumed_entity_ids)) or "none"
        suffix = ":continue" if self.continue_construction else ""
        if self.idea is not None:
            suffix += f":idea:{self.ideation_call_id}:{self.idea.idea_id}"
        object.__setattr__(self, "move_id", f"{self.operation}:{self.target_entity_id}:{focus}:{inputs}{suffix}")

    @classmethod
    def from_choice(cls, choice: OperationChoice) -> "LegalResearchMove":
        return cls(**asdict(choice))

    def to_operation_choice(self) -> OperationChoice:
        return OperationChoice(
            operation=self.operation,
            target_entity_id=self.target_entity_id,
            focus_obligation_id=self.focus_obligation_id,
            consumed_entity_ids=self.consumed_entity_ids,
            open_obligation_ids=self.open_obligation_ids,
            rationale=self.rationale,
            continue_construction=self.continue_construction,
            idea=self.idea, ideation_call_id=self.ideation_call_id,
            develop_provenance=self.develop_provenance,
        )


class StrategistDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    selected_move_id: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=1500)

    @field_validator("selected_move_id", "rationale")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be blank")
        return value  # Never normalize or repair a model-selected ID.


class PlanningBrief(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ResearchEntityBrief(PlanningBrief):
    id: int
    entity_type: str
    title: str
    status: str
    trust_state: str
    branch_status: str | None
    obligation_state: str | None
    attack_state: str | None


class ObligationBrief(ResearchEntityBrief):
    statement: str
    necessity_audit_state: str | None
    eligible: bool
    last_focused_iteration: int | None
    linked_unattacked_candidate_ids: tuple[int, ...]


class ResearchArtifactBrief(ResearchEntityBrief):
    statement: str


class ResearchMoveBrief(PlanningBrief):
    continue_construction: bool = False
    idea: CandidateIdea | None = None
    ideation_call_id: int | None = None
    move_id: str
    operation: str
    target_entity_id: int
    focus_obligation_id: int | None
    consumed_entity_ids: tuple[int, ...]
    open_obligation_ids: tuple[int, ...]
    rationale: str


class RecentIterationBrief(PlanningBrief):
    iteration_number: int | None
    operation: str
    target_entity_id: int
    focus_obligation_id: int | None
    status: str
    material_progress: bool
    duplicate_count: int
    attack_outcome: str | None
    stop_reason: str | None
    progress_class: ProgressEventKind | None = None
    progress_level: ProgressLevel | None = None
    progress_event_kinds: tuple[ProgressEventKind, ...] | None = None
    resolution_progress: bool | None = None
    open_obligations_before: int | None = None
    open_obligations_after: int | None = None
    resolved_obligation_count: int | None = None
    new_obligation_count: int | None = None
    candidate_created_count: int | None = None
    candidate_tested_count: int | None = None
    closed_branch_count: int | None = None
    accepted_artifact_count: int | None = None


class ControllerSummary(PlanningBrief):
    workstream_id: int
    workstream_status: str | None
    completed_iterations: int
    eligible_obligation_ids: tuple[int, ...]


class ProblemContractBrief(PlanningBrief):
    id: int
    entity_type: str
    title: str
    body: str
    trust_state: str


class ResearchState(PlanningBrief):
    """Transient planning view, never a durable scientific entity or execution context."""

    primary_target: ResearchEntityBrief
    problem_contract: tuple[ProblemContractBrief, ...]
    open_obligations: tuple[ObligationBrief, ...]
    blocked_or_terminal_branches: tuple[ResearchEntityBrief, ...]
    move_entities: tuple[ResearchArtifactBrief, ...]
    legal_moves: tuple[ResearchMoveBrief, ...]
    recent_iterations: tuple[RecentIterationBrief, ...]
    controller_summary: ControllerSummary


@dataclass(frozen=True)
class ResearchSelection:
    move: LegalResearchMove
    selection_mode: Literal["single_legal_move", "strategist", "deterministic_baseline"]
    legal_move_ids: tuple[str, ...]
    rationale: str
    strategy_provider: str | None = None
    strategy_model: str | None = None


@dataclass(frozen=True)
class ModelRoute:
    provider: str
    model: str
    effort: str
    max_output_tokens: int
    rationale: str


def choose_model_route(
    choice: OperationChoice,
    cfg: Config,
    *,
    provider_override: str = "auto",
) -> ModelRoute:
    """Pure operation-aware baseline: no API calls, availability probes, or fallback."""
    if provider_override not in {"auto", "openai", "anthropic"}:
        raise ConfigurationError(f"Unknown provider override: {provider_override}")
    if choice.operation not in OPERATIONS:
        raise ConfigurationError(f"Unknown research operation: {choice.operation}")
    if provider_override != "auto":
        model = getattr(cfg, f"{provider_override}_model")
        spec = get_model_spec(model, provider_override)
        return ModelRoute(
            spec.provider, model, "high", RESEARCH_MAX_OUTPUT_TOKENS,
            f"forced:{provider_override}",
        )

    role = choice.operation
    if role == "develop" and choice.develop_provenance != "ordinary":
        role = f"{choice.develop_provenance}_develop"
    elif role == "attack" and choice.focus_obligation_id is not None:
        role = "critical_attack"
    model = getattr(cfg, f"research_{role}_model")
    spec = get_model_spec(model)
    if model in {"gpt-6-astra", "claude-fable-5-1"}:
        raise ConfigurationError(
            f"Model {model!r} is reserved for explicit provider-override experiments; "
            "it cannot be used in automatic research routing."
        )
    return ModelRoute(
        spec.provider, model, "medium" if role == "critical_attack" else "high",
        RESEARCH_MAX_OUTPUT_TOKENS, f"auto:{role}",
    )


@dataclass(frozen=True)
class AcceptedArtifactBrief:
    entity_id: int
    artifact_type: str
    branch_status: str | None
    attempted_obligation_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PersistedStep:
    artifact_ids: tuple[int, ...]
    duplicate_count: int
    accepted_artifacts: tuple[AcceptedArtifactBrief, ...]


@dataclass(frozen=True)
class ResearchOutcome:
    workstream_id: int
    calls_made: int
    iteration_ids: tuple[int, ...]
    artifact_ids: tuple[int, ...]
    stop_reason: str
    final_status: str
    strategy_calls_made: int = 0
    ideation_calls_made: int = 0
    invocation_spend_usd: float = 0.0
    invocation_budget_usd: float = 1.50

    @property
    def total_api_calls_made(self) -> int:
        return self.calls_made + self.strategy_calls_made + self.ideation_calls_made


def _linked_ids(context: ResearchContext, workstream_id: int) -> set[int]:
    return {
        int(link["entity_id"])
        for link in context.workstream_links
        if int(link["workstream_id"]) == workstream_id
    }


def _input_ids(context: ResearchContext, workstream_id: int) -> set[int]:
    return {
        int(link["entity_id"])
        for link in context.workstream_links
        if int(link["workstream_id"]) == workstream_id and link["role"] == "input"
    }


def _human_judgment_allowed(context: ResearchContext, choice: OperationChoice) -> bool:
    workstream = getattr(context, "workstream", None)
    return bool(
        choice.operation == "develop"
        and workstream is not None
        and choice.target_entity_id in _input_ids(context, int(workstream["id"]))
        and any(
            int(entity["id"]) == choice.target_entity_id and entity.get("entity_type") != "Paper"
            for entity in context.entities
        )
    )


def _primary_target(context: ResearchContext, workstream_id: int) -> dict:
    input_ids = _input_ids(context, workstream_id)
    if not input_ids:
        raise TheoryError(f"Research workstream #{workstream_id} has no input entity.")
    candidates = [
        entity
        for entity in context.entities
        if int(entity["id"]) in input_ids and entity["entity_type"] in ROOT_TARGET_TYPES
    ]
    if not candidates:
        allowed = ", ".join(sorted(ROOT_TARGET_TYPES))
        raise TheoryError(
            f"Research workstream #{workstream_id} has no eligible primary target; "
            f"expected one input of type: {allowed}."
        )
    if len(candidates) > 1:
        ids = ", ".join(f"#{entity['id']}" for entity in candidates)
        raise TheoryError(
            f"Research workstream #{workstream_id} has ambiguous primary targets: {ids}. "
            "Keep exactly one eligible input target."
        )
    return candidates[0]


def _history(workstream_id: int) -> tuple[dict, ...]:
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM research_iterations WHERE workstream_id=? ORDER BY iteration_number",
            (workstream_id,),
        ).fetchall()
    return tuple(dict(row) for row in rows)


def _scientific_history(history: tuple[dict, ...]) -> tuple[dict, ...]:
    """Execution failures are audit records, never scientific planning evidence."""
    return tuple(row for row in history if row["status"] == "completed")


def _completed_for_target(history: tuple[dict, ...], operation: str, entity_id: int) -> bool:
    return any(
        row["status"] == "completed"
        and row["operation"] == operation
        and int(row["target_entity_id"]) == entity_id
        for row in history
    )


def _completed_no_issue_attack(
    history: tuple[dict, ...], candidate_id: int
) -> bool:
    return any(
        row["status"] == "completed"
        and row["operation"] == "attack"
        and int(row["target_entity_id"]) == candidate_id
        and row.get("attack_outcome") == "no_critical_issue"
        for row in history
    )


def _completed_synthesis_input_sets(
    history: tuple[dict, ...], target_entity_id: int
) -> set[frozenset[int]]:
    return {
        _stored_id_set(row.get("consumed_entity_ids_json"))
        for row in history
        if row["status"] == "completed"
        and row["operation"] == "synthesize"
        and int(row["target_entity_id"]) == target_entity_id
    }


def _is_new_synthesis_input_set(
    history: tuple[dict, ...], target_entity_id: int, consumed_entity_ids: tuple[int, ...]
) -> bool:
    return frozenset(consumed_entity_ids) not in _completed_synthesis_input_sets(
        history, target_entity_id
    )


def _attribute(context: ResearchContext, entity_id: int, key: str) -> str | None:
    return context.attributes.get(entity_id, {}).get(key)


def _is_precise_candidate(context: ResearchContext, entity: dict) -> bool:
    entity_id = int(entity["id"])
    if not _route_entity_is_live(context, entity_id):
        return False
    if entity["entity_type"] in PRECISE_ENTITY_TYPES:
        return True
    if entity["entity_type"] not in {"Conjecture", "Technique"}:
        return False
    if (_attribute(context, entity_id, "precise_candidate") or "").casefold() == "true":
        return True
    return _attribute(context, entity_id, "develop_item_type") == "protocol_component"


def _obligation_ids(
    context: ResearchContext, workstream_id: int, primary_entity_id: int, *,
    include_inactive_entities: bool = False,
) -> tuple[int, ...]:
    linked = _linked_ids(context, workstream_id)
    obligations: list[int] = []
    for entity in context.entities:
        entity_id = int(entity["id"])
        if (
            entity_id not in linked
            or entity_id == primary_entity_id
            or (entity["status"] != "active" and not include_inactive_entities)
        ):
            continue
        attrs = context.attributes.get(entity_id, {})
        is_open_obligation = entity["entity_type"] == "OpenQuestion" and (
            attrs.get("research_artifact_type") == "proof_obligation"
            or attrs.get("develop_item_type") == "proof_obligation"
            or attrs.get("is_proof_obligation", "").casefold() == "true"
        )
        is_explicit_legacy_obstruction_obligation = (
            entity["entity_type"] == "Obstruction"
            and attrs.get("is_proof_obligation", "").casefold() == "true"
        )
        if is_open_obligation or is_explicit_legacy_obstruction_obligation:
            obligations.append(entity_id)
    return tuple(sorted(obligations))


def _persisted_open_obligation_ids(
    context: ResearchContext, workstream_id: int, primary_entity_id: int
) -> tuple[int, ...]:
    return tuple(
        entity_id
        for entity_id in _obligation_ids(context, workstream_id, primary_entity_id)
        if _attribute(context, entity_id, "research_obligation_state")
        not in INACTIVE_OBLIGATION_STATES
    )


def _obligation_route_activity(
    context: ResearchContext, workstream_id: int, primary_entity_id: int,
) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    """Derive route attachment from construction ownership, bypasses and parents.

    An obligation's persisted state is unchanged. Replacement premises depend on
    a surviving owning bypass; other descendants depend on their explicit parent
    obligations. A shared premise remains attached through any live route.
    """
    obligation_ids = set(_obligation_ids(
        context, workstream_id, primary_entity_id, include_inactive_entities=True,
    ))
    persisted_open = set(_persisted_open_obligation_ids(context, workstream_id, primary_entity_id))
    owners: dict[int, set[int]] = {}
    owner_parents: dict[int, set[int]] = {}
    linked = _linked_ids(context, workstream_id)
    for candidate_id in sorted(linked):
        attrs = context.attributes.get(candidate_id, {})
        raw = attrs.get("research_bypass_replacement_obligation_ids")
        if raw is None:
            continue
        parent = _explicit_obligation_id(attrs.get("research_reframe_target_obligation_id"), obligation_ids)
        if parent is None:
            continue
        replacements = json.loads(raw)
        if not isinstance(replacements, list) or any(type(i) is not int or i <= 0 for i in replacements):
            raise TheoryError("Invalid persisted bypass replacement provenance.")
        for replacement in replacements:
            if replacement in obligation_ids:
                owners.setdefault(replacement, set()).add(candidate_id)
                owner_parents.setdefault(replacement, set()).add(parent)

    parents: dict[int, tuple[int, ...]] = {}
    for obligation_id in obligation_ids - owners.keys():
        attrs = context.attributes.get(obligation_id, {})
        focus = _explicit_obligation_id(attrs.get("research_focus_obligation_id"), obligation_ids)
        parent_ids = ({focus} if focus is not None else
                      set(_stored_id_set(attrs.get("related_entity_ids")) & obligation_ids))
        parent_ids.discard(obligation_id)
        parents[obligation_id] = tuple(sorted(parent_ids))

    route_owned = {
        i: _stored_id_set(_attribute(context, i, ROUTE_IDS)) for i in obligation_ids
    }
    live_routes = live_construction_route_ids(context)
    route_allowed = {
        i for i, roots in route_owned.items()
        if (not roots or roots & live_routes) and not entity_has_closing_relation(context, i)
    }
    # Bypass descendants still require an attached parent. Ordinary ownership
    # may be shared across routes even when an older parent route was superseded.
    bypass_descendants = set(owners)
    while True:
        added = {i for i, parent_ids in parents.items()
                 if set(parent_ids) & bypass_descendants} - bypass_descendants
        if not added:
            break
        bypass_descendants.update(added)
    attached = {
        i for i, parent_ids in parents.items() if i in route_allowed and (
            not parent_ids or (i not in bypass_descendants
                              and _attribute(context, i, ROUTE_IDS) is not None)
        )
    }
    while True:
        added = {
            i for i in (obligation_ids - attached) & route_allowed
            if (any(parent in attached for parent in parents.get(i, ()))
                if i not in owners else _route_entity_is_live(context, i) and any(
                    parent in attached and bypass_route_is_live(context, parent, candidate_id)
                    for candidate_id in owners[i] for parent in owner_parents[i]
                    if _attribute(context, candidate_id, "research_reframe_target_obligation_id") == str(parent)
                ))
        }
        if not added:
            break
        attached.update(added)

    active = tuple(sorted(persisted_open & attached))
    return (active,
            {i: tuple(sorted(ids)) for i, ids in owners.items()},
            parents)


def _construction_entity_is_active(context: ResearchContext, entity_id: int) -> bool:
    roots = _stored_id_set(_attribute(context, entity_id, ROUTE_IDS))
    return not roots or bool(roots & live_construction_route_ids(context))


def _open_obligation_ids(
    context: ResearchContext, workstream_id: int, primary_entity_id: int
) -> tuple[int, ...]:
    """Open obligations supported by at least one live construction route."""
    return _obligation_route_activity(context, workstream_id, primary_entity_id)[0]


def _candidate_has_unresolved_critical_issue(
    context: ResearchContext, candidate_id: int
) -> bool:
    prior_attack_state = _attribute(context, candidate_id, "research_attack_state")
    if prior_attack_state in {"challenged", "inconclusive"}:
        return True
    return entity_has_closing_relation(context, candidate_id)


def _candidate_meets_obligation_closure_structure(
    context: ResearchContext, candidate_id: int, obligation_id: int
) -> bool:
    candidate = next(
        (entity for entity in context.entities if int(entity["id"]) == candidate_id),
        None,
    )
    if (
        candidate is None
        or candidate["entity_type"] not in {"ProofAttempt", "Lemma"}
        or candidate["status"] != "active"
        or candidate["trust_state"] == "contradicted"
    ):
        return False
    attrs = context.attributes.get(candidate_id, {})
    if obligation_id not in _stored_id_set(attrs.get("related_entity_ids")):
        return False
    if obligation_id not in _stored_id_set(attrs.get("addresses_obligation_ids")):
        return False
    return any(
        relation["relation_type"] == "ATTEMPTS"
        and int(relation["source_entity_id"]) == candidate_id
        and int(relation["target_entity_id"]) == obligation_id
        for relation in context.relations
    )


def _candidate_can_complete_obligation_after_attack(
    context: ResearchContext, candidate_id: int, obligation_id: int
) -> bool:
    return _candidate_meets_obligation_closure_structure(
        context, candidate_id, obligation_id
    ) and not _candidate_has_unresolved_critical_issue(context, candidate_id)


def _all_obligations_have_completed_candidates(
    context: ResearchContext,
    history: tuple[dict, ...],
    workstream_id: int,
    primary_entity_id: int,
) -> bool:
    obligation_ids = _obligation_ids(context, workstream_id, primary_entity_id)
    if not obligation_ids:
        return False
    for obligation_id in obligation_ids:
        attrs = context.attributes.get(obligation_id, {})
        if attrs.get("research_obligation_state") != "resolved_candidate":
            return False
        try:
            candidate_id = int(attrs["research_surviving_candidate_id"])
        except (KeyError, TypeError, ValueError):
            return False
        if (
            _attribute(context, candidate_id, "research_attack_state")
            != "survived_attack"
            or not _completed_no_issue_attack(history, candidate_id)
            or not _candidate_meets_obligation_closure_structure(
                context, candidate_id, obligation_id
            )
            or _candidate_has_unresolved_critical_issue(context, candidate_id)
        ):
            return False
    return True


def _all_branches_terminal(context: ResearchContext, workstream_id: int) -> bool:
    """Exhaust established construction routes, never standalone negative evidence.

    An obligation-only root becomes substantive when construction is persisted
    on it. Terminal artifacts remain evidence; only existing structural route
    liveness and active work determine whether the whole frontier is exhausted.
    """
    linked = _linked_ids(context, workstream_id)
    roots = {i for i in linked if STARTED_AT in context.attributes.get(i, {})}
    substantive = {i for i in roots
                   if _attribute(context, i, "research_artifact_type") in CONSTRUCTION_TYPES}
    for entity_id in linked:
        if _attribute(context, entity_id, "research_artifact_type") in CONSTRUCTION_TYPES:
            substantive.update(_stored_id_set(_attribute(context, entity_id, ROUTE_IDS)) & roots)
    if not substantive or roots & live_construction_route_ids(context):
        return False
    primary_id = int(_primary_target(context, workstream_id)["id"])
    if _open_obligation_ids(context, workstream_id, primary_id):
        return False

    inputs = _input_ids(context, workstream_id)
    for entity in context.entities:
        entity_id = int(entity["id"])
        if entity_id not in linked - inputs or not _route_entity_is_live(context, entity_id):
            continue
        attrs = context.attributes.get(entity_id, {})
        if (_construction_entity_is_active(context, entity_id)
                and (_is_precise_candidate(context, entity)
                     or attrs.get("research_artifact_type") in CONSTRUCTION_TYPES)):
            return False
        parent = _explicit_obligation_id(attrs.get("research_reframe_target_obligation_id"), linked)
        if parent is not None and (
            (_attribute(context, parent, "research_obligation_state") == "reframe_pending_attack"
             and _attribute(context, parent, "research_reframe_candidate_id") == str(entity_id))
            or bypass_route_is_live(context, parent, entity_id)
        ):
            return False
    return True


def _entity_is_relevant_to_obligation(
    context: ResearchContext, entity_id: int, obligation_id: int
) -> bool:
    """Use only explicit, shallow graph provenance to establish obligation relevance."""
    attrs = context.attributes.get(entity_id, {})
    if obligation_id in _stored_id_set(attrs.get("related_entity_ids")):
        return True
    if obligation_id in _stored_id_set(attrs.get("addresses_obligation_ids")):
        return True
    if obligation_id in _stored_id_set(attrs.get("research_related_obligation_ids")):
        return True
    if attrs.get("research_focus_obligation_id") == str(obligation_id):
        return True

    obligation_attrs = context.attributes.get(obligation_id, {})
    if entity_id in _stored_id_set(obligation_attrs.get("related_entity_ids")):
        return True

    if any(
        {
            int(relation["source_entity_id"]),
            int(relation["target_entity_id"]),
        }
        == {entity_id, obligation_id}
        for relation in context.relations
    ):
        return True

    return any(
        {entity_id, obligation_id}
        <= set(_stored_id_set(other_attrs.get("related_entity_ids")))
        for other_attrs in context.attributes.values()
    )


def _has_terminal_branch_state(context: ResearchContext, entity_id: int) -> bool:
    attrs = context.attributes.get(entity_id, {})
    state = attrs.get("research_branch_status") or attrs.get("develop_branch_status")
    return state in TERMINAL_BRANCH_STATES


def _relevant_synthesis_inputs(
    context: ResearchContext,
    workstream_id: int,
    obligation_id: int,
) -> tuple[int, ...]:
    linked = _linked_ids(context, workstream_id)
    candidates = [
        entity
        for entity in context.entities
        if int(entity["id"]) in linked
        and int(entity["id"]) != obligation_id
        and entity["status"] == "active"
        and entity["trust_state"] != "contradicted"
        and not _has_terminal_branch_state(context, int(entity["id"]))
        and _construction_entity_is_active(context, int(entity["id"]))
        and entity["entity_type"] in SYNTHESIS_INPUT_TYPES
        and _entity_is_relevant_to_obligation(context, int(entity["id"]), obligation_id)
    ]
    candidates.sort(
        key=lambda entity: (
            entity["entity_type"] not in {"Lemma", "Theorem", "ProofAttempt", "Technique"},
            -int(entity["id"]),
        )
    )
    return tuple(int(entity["id"]) for entity in candidates[:4])


def _primary_synthesis_bundles(
    context: ResearchContext, workstream_id: int, primary_id: int, history: tuple[dict, ...],
) -> tuple[tuple[int, ...], ...]:
    """At most three complementary pairs; no Cartesian frontier enumeration.

    Retain active negative results even when they record a failed route. A terminal
    positive candidate, or an inactive/contradicted negative result, is not evidence.
    Material keys and the ordinary lexical duplicate policy suppress paraphrases.
    """
    inputs = _input_ids(context, workstream_id)
    linked = _linked_ids(context, workstream_id)
    negatives = {"Obstruction", "Counterexample", "FailedApproach"}
    roles = {"Technique": "mechanism", "Lemma": "proof", "Theorem": "proof",
             "ProofAttempt": "proof", "Conjecture": "candidate", "Finding": "finding",
             **{kind: "negative" for kind in negatives}}
    origins = {
        entity_id: ("iteration", index)
        for index, row in enumerate(history) if row["status"] == "completed"
        for entity_id in _stored_id_set(row.get("artifact_ids_json"))
    }
    candidates: list[int] = []
    role_by_id: dict[int, str] = {}
    branch_by_id: dict[int, object] = {}
    keys: set[str] = set()
    fingerprints: list[frozenset[str]] = []
    # Oldest representative retains important negative history and stable identity.
    for entity in sorted(context.entities, key=lambda entity: int(entity["id"])):
        entity_id = int(entity["id"])
        kind = entity["entity_type"]
        attrs = context.attributes.get(entity_id, {})
        if (entity_id not in linked or entity_id in inputs or entity_id == primary_id
                or kind not in roles or entity["status"] != "active"
                or entity["trust_state"] == "contradicted"
                or not (attrs.get("research_artifact_type") or attrs.get("develop_item_type")
                        or attrs.get("develop_branch_name") or entity_id in origins)
                or (kind not in negatives and _has_terminal_branch_state(context, entity_id))):
            continue
        key = attrs.get("research_material_key")
        statement = (attrs.get("research_statement")
                     or (entity.get("body") or "").split("\n\n", 1)[0] or entity["title"])
        tokens = _normalized_tokens(statement)
        if (key and key in keys) or _is_lexical_duplicate(tokens, fingerprints):
            continue
        if key:
            keys.add(key)
        fingerprints.append(tokens)
        candidates.append(entity_id)
        role_by_id[entity_id] = roles[kind]
        related = tuple(sorted(_stored_id_set(attrs.get("related_entity_ids")) - inputs))
        branch_by_id[entity_id] = (
            ("focus", attrs["research_focus_obligation_id"])
            if attrs.get("research_focus_obligation_id") else
            ("related", related) if related else origins.get(entity_id)
        )
    negative_ids = [i for i in candidates if role_by_id[i] == "negative"]
    positive_ids = [i for i in reversed(candidates) if role_by_id[i] != "negative"]
    # Three anchors, each with a linear partner search, bound the candidate frontier.
    anchors = list(dict.fromkeys(negative_ids[:1] + positive_ids))[:3]
    completed = _completed_synthesis_input_sets(history, primary_id)
    bundles: list[tuple[int, ...]] = []
    for anchor in anchors:
        partners = sorted(candidates, key=lambda i: (
            role_by_id[i] == role_by_id[anchor],
            role_by_id[i] not in {"mechanism", "proof", "candidate"},
            branch_by_id[i] == branch_by_id[anchor], -i,
        ))
        for partner in partners:
            if partner == anchor or role_by_id[anchor] == role_by_id[partner] == "negative":
                continue
            different_branch = (branch_by_id[anchor] is not None
                                and branch_by_id[partner] is not None
                                and branch_by_id[anchor] != branch_by_id[partner])
            if (role_by_id[anchor] == role_by_id[partner] and not different_branch
                    and role_by_id[anchor] != "mechanism"):
                continue
            bundle = tuple(sorted((anchor, partner)))
            if frozenset(bundle) in completed or bundle in bundles:
                continue
            bundles.append(bundle)
            break
    return tuple(bundles)


def _relevant_prove_candidates(
    context: ResearchContext,
    history: tuple[dict, ...],
    obligation_id: int,
) -> tuple[dict, ...]:
    if context.workstream is None:
        return ()
    workstream_id = int(context.workstream["id"])
    linked = _linked_ids(context, workstream_id)
    candidates = [
        entity
        for entity in context.entities
        if int(entity["id"]) in linked
        and entity["entity_type"] != "ProofAttempt"
        and _is_precise_candidate(context, entity)
        and _construction_entity_is_active(context, int(entity["id"]))
        and not _completed_for_target(history, "prove", int(entity["id"]))
        and _entity_is_relevant_to_obligation(
            context, int(entity["id"]), obligation_id
        )
    ]
    candidates.sort(
        key=lambda entity: (
            {"Lemma": 0, "Theorem": 1, "Technique": 2}.get(
                entity["entity_type"], 3
            ),
            -int(entity["id"]),
        )
    )
    return tuple(candidates)


def _relevant_unattacked_proof_attempts(
    context: ResearchContext,
    history: tuple[dict, ...],
    open_obligation_ids: tuple[int, ...],
) -> tuple[dict, ...]:
    """Return newest linked proof attempts connected to a currently open obligation."""
    if context.workstream is None or not open_obligation_ids:
        return ()
    workstream_id = int(context.workstream["id"])
    linked = _linked_ids(context, workstream_id)
    open_ids = set(open_obligation_ids)
    candidates: list[dict] = []
    for entity in context.entities:
        entity_id = int(entity["id"])
        if (
            entity_id not in linked
            or entity["entity_type"] != "ProofAttempt"
            or entity["status"] != "active"
            or entity["trust_state"] == "contradicted"
            or not _construction_entity_is_active(context, entity_id)
            or not _route_entity_is_live(context, entity_id)
            or _completed_for_target(history, "attack", entity_id)
        ):
            continue
        if any(
            _entity_is_relevant_to_obligation(context, entity_id, obligation_id)
            for obligation_id in open_ids
        ):
            candidates.append(entity)
    candidates.sort(key=lambda entity: -int(entity["id"]))
    return tuple(candidates)


def _explicit_obligation_id(raw: object, obligation_ids: set[int]) -> int | None:
    try:
        entity_id = int(raw)
    except (TypeError, ValueError):
        return None
    return entity_id if entity_id in obligation_ids else None


def _iteration_focus_obligation_id(
    context: ResearchContext,
    row: dict,
    obligation_ids: set[int],
) -> int | None:
    """Recover one completed iteration's focus from explicit, auditable provenance."""
    if row.get("status") != "completed":
        return None

    stored_focus = _explicit_obligation_id(
        row.get("focus_obligation_id"), obligation_ids
    )
    if stored_focus is not None:
        return stored_focus

    target_id = _explicit_obligation_id(row.get("target_entity_id"), obligation_ids)
    if target_id is not None:
        return target_id

    artifact_focuses = {
        focus_id
        for artifact_id in _stored_id_set(row.get("artifact_ids_json"))
        if (
            focus_id := _explicit_obligation_id(
                _attribute(context, artifact_id, "research_focus_obligation_id"),
                obligation_ids,
            )
        )
        is not None
    }
    if len(artifact_focuses) == 1:
        return next(iter(artifact_focuses))

    try:
        raw_target_id = int(row["target_entity_id"])
    except (KeyError, TypeError, ValueError):
        return None
    candidate_focus = _explicit_obligation_id(
        _attribute(context, raw_target_id, "research_focus_obligation_id"),
        obligation_ids,
    )
    if candidate_focus is not None:
        return candidate_focus

    directly_connected = set(
        _stored_id_set(_attribute(context, raw_target_id, "related_entity_ids"))
    ) | set(
        _stored_id_set(_attribute(context, raw_target_id, "addresses_obligation_ids"))
    )
    directly_connected.update(
        int(relation["target_entity_id"])
        for relation in context.relations
        if relation["relation_type"] == "ATTEMPTS"
        and int(relation["source_entity_id"]) == raw_target_id
    )
    connected_obligations = directly_connected & obligation_ids
    if len(connected_obligations) == 1:
        return next(iter(connected_obligations))
    return None


def _active_obligation_history(
    context: ResearchContext, workstream_id: int, primary_entity_id: int,
    history: tuple[dict, ...],
) -> tuple[dict, ...]:
    """Exclude work on detached premises from current strategy evidence."""
    completed = _scientific_history(history)
    inactive = (set(_persisted_open_obligation_ids(context, workstream_id, primary_entity_id))
                - set(_open_obligation_ids(context, workstream_id, primary_entity_id)))
    if not inactive:
        return completed
    all_ids = set(_obligation_ids(context, workstream_id, primary_entity_id))
    return tuple(row for row in completed
                 if _iteration_focus_obligation_id(context, row, all_ids) not in inactive)


def _obligation_ideation_context(
    context: ResearchContext, workstream_id: int, primary_id: int, obligation_id: int,
) -> ResearchContext:
    """Retain the local premise, live owners and relevant negative evidence exactly."""
    roots = _stored_id_set(_attribute(context, obligation_id, ROUTE_IDS)) & live_construction_route_ids(context)
    params = dict(workstream_id=workstream_id, primary_entity_id=primary_id,
                  target_entity_id=obligation_id, focus_obligation_id=obligation_id,
                  operation="develop")
    scoped = focus_research_context(context, additional_entity_ids=tuple(sorted(roots)), **params)
    local_ids = {int(e["id"]) for e in scoped.entities} - _input_ids(context, workstream_id)
    evidence = {
        int(e["id"]) for e in context.entities
        if e["entity_type"] in {"Obstruction", "Counterexample", "FailedApproach"}
        and _construction_entity_is_active(context, int(e["id"]))
        and (_stored_id_set(_attribute(context, int(e["id"]), "related_entity_ids")) & local_ids
             or any(r["status"] == "active" and (
                 r["relation_type"] in {"REFUTES", "BLOCKS", "CONTRADICTS"}
                 and int(r["source_entity_id"]) == int(e["id"])
                 and int(r["target_entity_id"]) in local_ids
                 or r["relation_type"] == "FAILS_AT"
                 and int(r["target_entity_id"]) == int(e["id"])
                 and int(r["source_entity_id"]) in local_ids
             ) for r in context.relations))
    }
    return focus_research_context(context, additional_entity_ids=tuple(sorted(roots | evidence)), **params)


def _top_level_ideation_context(
    context: ResearchContext, workstream_id: int, primary_id: int, trigger_ids: tuple[int, ...],
) -> ResearchContext:
    """Exact contract, live roots, latest four live artifacts per root and evidence."""
    roots = live_construction_route_ids(context)
    linked = _linked_ids(context, workstream_id) - _input_ids(context, workstream_id)
    recent: set[int] = set()
    for root in sorted(roots):
        recent.update(sorted((i for i in linked
            if root in _stored_id_set(_attribute(context, i, ROUTE_IDS))
            and _route_entity_is_live(context, i)), reverse=True)[:4])
    anchors = set(roots) | recent | set(trigger_ids)
    evidence = {int(e["id"]) for e in context.entities
                if e["status"] == "active"
                and e["entity_type"] in {"Obstruction", "Counterexample", "FailedApproach"}
                and _construction_entity_is_active(context, int(e["id"]))
                and (_stored_id_set(_attribute(context, int(e["id"]), ROUTE_IDS)) & roots
                     or _stored_id_set(_attribute(context, int(e["id"]), "related_entity_ids")) & anchors
                     or any(r["status"] == "active" and (
                         r["relation_type"] in {"REFUTES", "BLOCKS", "CONTRADICTS"}
                         and int(r["source_entity_id"]) == int(e["id"])
                         and int(r["target_entity_id"]) in anchors
                         or r["relation_type"] == "FAILS_AT"
                         and int(r["target_entity_id"]) == int(e["id"])
                         and int(r["source_entity_id"]) in anchors
                     ) for r in context.relations))}
    return focus_research_context(
        context, workstream_id=workstream_id, primary_entity_id=primary_id,
        target_entity_id=primary_id, operation="develop",
        additional_entity_ids=tuple(sorted(anchors | evidence)), forward_dependencies_only=True,
    )


def eligible_open_obligation_ids(
    context: ResearchContext,
    open_obligation_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Return the existing actionable leaf frontier in stable ID order."""
    if not open_obligation_ids:
        return ()

    open_ids = set(open_obligation_ids)
    parents_with_open_children: set[int] = set()
    for child_id in open_obligation_ids:
        attrs = context.attributes.get(child_id, {})
        explicit_parents = set(_stored_id_set(attrs.get("related_entity_ids")))
        focus_parent = _explicit_obligation_id(
            attrs.get("research_focus_obligation_id"), open_ids
        )
        if focus_parent is not None:
            explicit_parents.add(focus_parent)
        parents_with_open_children.update(
            parent_id
            for parent_id in explicit_parents & open_ids
            if parent_id != child_id
            # An exhausted replacement must not hide the route it reactivated.
            and not (
                _attribute(context, parent_id, "research_necessity_audit_state") == "reactivated"
                and not _route_entity_is_live(context, child_id)
            )
        )

    # A pending necessity challenge must remain actionable even if its audit
    # created replacement obligations referencing the original parent.
    pending = {entity_id for entity_id in open_ids
               if _attribute(context, entity_id, "research_obligation_state") == "reframe_pending_attack"}
    leaf_ids = sorted((open_ids - parents_with_open_children) | pending)
    # Malformed cyclic provenance has no leaf. Keep scheduling deterministic and live
    # without inventing a parent/child direction that is absent from the graph.
    return tuple(leaf_ids or sorted(open_ids))

def _select_focus_obligation(
    context: ResearchContext,
    history: tuple[dict, ...],
    open_obligation_ids: tuple[int, ...],
) -> int:
    """Select an open leaf obligation fairly using only persisted provenance."""
    if not open_obligation_ids:
        raise TheoryError("Cannot select a focus obligation when none are open.")
    open_ids = set(open_obligation_ids)
    eligible_ids = eligible_open_obligation_ids(context, open_obligation_ids)

    last_focused_at: dict[int, int] = {}
    for position, row in enumerate(history):
        focus_id = _iteration_focus_obligation_id(context, row, open_ids)
        if focus_id in eligible_ids:
            last_focused_at[focus_id] = position

    never_focused = [
        obligation_id
        for obligation_id in eligible_ids
        if obligation_id not in last_focused_at
    ]
    if never_focused:
        return min(never_focused)
    return min(
        eligible_ids,
        key=lambda obligation_id: (last_focused_at[obligation_id], obligation_id),
    )


def _obligation_frontier_choice(
    context: ResearchContext,
    workstream_id: int,
    history: tuple[dict, ...],
    open_obligations: tuple[int, ...],
    obligation_id: int,
    *,
    exclude_terminal: bool = False,
) -> OperationChoice:
    """The shared deterministic local precedence for one obligation branch."""
    if _attribute(context, obligation_id, "research_obligation_state") == "reframe_pending_attack":
        candidate_id = _reframe_candidate_id(context, workstream_id, obligation_id)
        if _completed_for_target(history, "attack", candidate_id):
            raise TheoryError("Pending reframe candidate already has a completed attack.")
        return OperationChoice(
            "attack", candidate_id, "Independently test the proposed bypass route.",
            open_obligation_ids=open_obligations, focus_obligation_id=obligation_id,
        )
    relevant_unattacked = tuple(
        entity for entity in _relevant_unattacked_proof_attempts(context, history, (obligation_id,))
        if not exclude_terminal or not _has_terminal_branch_state(context, int(entity["id"]))
    )
    if relevant_unattacked:
        target_id = int(relevant_unattacked[0]["id"])
        return OperationChoice(
            operation="attack",
            target_entity_id=target_id,
            open_obligation_ids=open_obligations,
            focus_obligation_id=obligation_id,
            rationale=(
                f"Proof attempt #{target_id} is linked to open obligation "
                f"#{obligation_id} "
                "and has not yet received one bounded adversarial attack."
            ),
        )
    consumed = _relevant_synthesis_inputs(context, workstream_id, obligation_id)
    if len(consumed) >= 2 and _is_new_synthesis_input_set(
        history, obligation_id, consumed
    ):
        return OperationChoice(
            operation="synthesize",
            target_entity_id=obligation_id,
            consumed_entity_ids=consumed,
            open_obligation_ids=open_obligations,
            focus_obligation_id=obligation_id,
            rationale=(
                f"Open proof obligation #{obligation_id} has {len(consumed)} linked, "
                "non-terminal artifacts that can be combined in one bounded synthesis."
            ),
        )
    relevant_prove_candidates = tuple(
        entity for entity in _relevant_prove_candidates(context, history, obligation_id)
        if not exclude_terminal or not _has_terminal_branch_state(context, int(entity["id"]))
    )
    if relevant_prove_candidates:
        candidate_id = int(relevant_prove_candidates[0]["id"])
        return OperationChoice(
            operation="prove",
            target_entity_id=candidate_id,
            open_obligation_ids=open_obligations,
            focus_obligation_id=obligation_id,
            rationale=(
                f"Precise candidate #{candidate_id} exists while proof obligation "
                f"#{obligation_id} remains open; a rigorous proof attempt is the next "
                "material step."
            ),
        )
    return OperationChoice(
        operation="develop",
        target_entity_id=obligation_id,
        open_obligation_ids=open_obligations,
        focus_obligation_id=obligation_id,
        rationale=(
            f"Proof obligation #{obligation_id} is open, but its branch lacks either two "
            "relevant synthesis inputs or a relevant unproved precise candidate; refine "
            "this obligation directly."
        ),
    )


def _develop_resets_construction_streak(
    context: ResearchContext, iteration: dict,
) -> bool:
    return any(
        _attribute(context, entity_id, "research_artifact_type") in {"proof_attempt", "proof_obligation"}
        or _attribute(context, entity_id, "research_branch_status") == "promising"
        for entity_id in _stored_id_set(iteration.get("artifact_ids_json"))
    )


def _iteration_construction_routes(context: ResearchContext, iteration: dict) -> frozenset[int]:
    accepted = _stored_id_set(iteration.get("artifact_ids_json"))
    started = frozenset(i for i in accepted
                        if _attribute(context, i, STARTED_AT) == str(iteration.get("id")))
    if started:
        return started
    anchors = accepted | _stored_id_set(iteration.get("consumed_entity_ids_json"))
    anchors |= {iteration["target_entity_id"], iteration.get("focus_obligation_id")}
    return frozenset().union(*(
        _stored_id_set(context.attributes.get(i, {}).get(ROUTE_IDS)) for i in anchors
    ))


def _constructive_continuation_streak(
    context: ResearchContext, history: tuple[dict, ...],
) -> tuple[frozenset[int], int]:
    """Reconstruct consecutive completed continuations from route-owned receipts.

    Shared ancestry retains the same route through the intersection of owners.
    Failed calls do not interrupt the scientific execution path. No process-local
    counter or artifact-age cutoff is used.
    """
    route: frozenset[int] = frozenset()
    streak = 0
    previous: dict | None = None
    for row in _scientific_history(history):
        roots = _iteration_construction_routes(context, row)
        continuation = (row["operation"] == "develop"
                        and (row.get("selected_move_id") or "").endswith(":continue"))
        if (row["operation"] == "develop" and not roots and previous is not None
                and previous["operation"] == "develop"
                and previous["target_entity_id"] == row["target_entity_id"]):
            roots = route  # Duplicate-only development retains its route.
        common = route & roots
        if (row["operation"] == "develop" and roots
                and not _develop_resets_construction_streak(context, row)):
            streak = (streak if common else 0) + int(continuation)
            route = common or roots
        else:
            streak = 0
            route = roots
        previous = row
    return route, streak


def _route_consolidation_choice(
    context: ResearchContext, workstream_id: int, primary_id: int,
    history: tuple[dict, ...], open_obligations: tuple[int, ...],
) -> OperationChoice | None:
    route, streak = _constructive_continuation_streak(context, history)
    live_routes = route & live_construction_route_ids(context)
    if streak < MAX_CONSTRUCTIVE_CONTINUATIONS or not live_routes:
        return None
    linked = _linked_ids(context, workstream_id) - _input_ids(context, workstream_id) - {primary_id}
    candidates = [
        int(entity["id"]) for entity in context.entities
        if int(entity["id"]) in linked
        and entity["entity_type"] in SYNTHESIS_INPUT_TYPES
        and _stored_id_set(_attribute(context, int(entity["id"]), ROUTE_IDS)) & live_routes
        and _route_entity_is_live(context, int(entity["id"]))
    ]
    # Prefer construction components and their lemmas/findings, then newest
    # accepted artifacts within that group. IDs provide stable tie breaking.
    candidates.sort(key=lambda i: (
        _attribute(context, i, "research_artifact_type") not in {"protocol_component", "lemma", "finding"},
        -i,
    ))
    consumed = tuple(candidates[:4])
    if len(consumed) < 2 or not _is_new_synthesis_input_set(history, primary_id, consumed):
        return None
    return OperationChoice(
        "synthesize", primary_id,
        "Consolidate the unfinished construction into a testable candidate and/or explicit obligations.",
        consumed_entity_ids=consumed, open_obligation_ids=open_obligations,
    )


def _constructive_continuation(
    context: ResearchContext, workstream_id: int, history: tuple[dict, ...],
    open_obligations: tuple[int, ...],
) -> OperationChoice | None:
    """Continue only newly accepted, live, explicitly unfinished construction.

    Obligation-only expansion, duplicate output, a finished candidate, or an
    intervening operation does not earn another construction step. Existing
    call/stagnation bounds still apply. Three route-local continuations lead to
    a consolidation checkpoint; this does not close any obligation.
    """
    history = _scientific_history(history)
    if not history:
        return None
    previous = history[-1]
    if (previous["status"] != "completed" or previous["operation"] != "develop"
            or not previous.get("material_progress")):
        return None
    target = int(previous["target_entity_id"])
    focus = previous.get("focus_obligation_id")
    if focus is not None and focus not in open_obligations:
        return None
    if target not in _input_ids(context, workstream_id) and not _route_entity_is_live(context, target):
        return None
    accepted = _stored_id_set(previous.get("artifact_ids_json"))
    if any(
        _attribute(context, entity_id, "research_artifact_type") == "proof_attempt"
        or _attribute(context, entity_id, "research_branch_status") == "promising"
        for entity_id in accepted
    ) or _constructive_continuation_streak(context, history)[1] >= MAX_CONSTRUCTIVE_CONTINUATIONS:
        return None
    if not any(
        _attribute(context, entity_id, "research_artifact_type") == "protocol_component"
        and _attribute(context, entity_id, "research_branch_status") == "unresolved"
        and target in _stored_id_set(_attribute(context, entity_id, "related_entity_ids"))
        and _route_entity_is_live(context, entity_id)
        and _construction_entity_is_active(context, entity_id)
        for entity_id in accepted
    ):
        return None
    if focus is None:
        routes = _iteration_construction_routes(context, previous) & live_construction_route_ids(context)
        owning = tuple(
            obligation for obligation in eligible_open_obligation_ids(context, open_obligations)
            if _stored_id_set(_attribute(context, obligation, ROUTE_IDS)) & routes
        )
        if owning:
            focus = _select_focus_obligation(context, history, owning)
    return OperationChoice(
        "develop", target,
        "Continue the same unfinished protocol route: the previous develop step accepted "
        "a live, substantive protocol component. Extend its construction before routine "
        "branch expansion or proof work; retain all outstanding obligations.",
        open_obligation_ids=open_obligations, focus_obligation_id=focus,
        continue_construction=True,
        develop_provenance=previous.get("develop_provenance", "ordinary"),
    )


def choose_next_operation(
    context: ResearchContext,
    workstream_id: int,
    primary: dict,
    history: tuple[dict, ...],
) -> OperationChoice:
    """Choose the next bounded operation deterministically from current graph state."""
    history = _scientific_history(history)
    primary_id = int(primary["id"])
    linked = _linked_ids(context, workstream_id)
    entity_by_id = {int(entity["id"]): entity for entity in context.entities}
    open_obligations = _open_obligation_ids(context, workstream_id, primary_id)
    continuation = _constructive_continuation(context, workstream_id, history, open_obligations)
    if continuation is not None:
        return continuation
    precise = [
        entity_by_id[entity_id]
        for entity_id in linked
        if entity_id in entity_by_id and _is_precise_candidate(context, entity_by_id[entity_id])
        and _construction_entity_is_active(context, entity_id)
    ]
    precise.sort(
        key=lambda entity: (
            {"Lemma": 0, "Theorem": 1, "Technique": 2, "ProofAttempt": 3}.get(
                entity["entity_type"], 4
            ),
            -int(entity["id"]),
        )
    )
    prove_candidates = [
        entity
        for entity in precise
        if entity["entity_type"] != "ProofAttempt"
        and not _completed_for_target(history, "prove", int(entity["id"]))
    ]

    if open_obligations:
        obligation_id = _select_focus_obligation(context, history, open_obligations)
        return _obligation_frontier_choice(
            context, workstream_id, history, open_obligations, obligation_id
        )

    unattacked_proofs = [
        entity
        for entity in precise
        if entity["entity_type"] == "ProofAttempt"
        and not _completed_for_target(history, "attack", int(entity["id"]))
    ]
    if unattacked_proofs:
        target_id = int(unattacked_proofs[0]["id"])
        return OperationChoice(
            operation="attack",
            target_entity_id=target_id,
            rationale=(
                f"Proof candidate #{target_id} is precise and has not yet received a bounded "
                "adversarial attack."
            ),
        )

    if _is_precise_candidate(context, primary) and primary["entity_type"] in PRECISE_ENTITY_TYPES:
        if not _completed_for_target(history, "attack", primary_id):
            return OperationChoice(
                operation="attack",
                target_entity_id=primary_id,
                rationale=(
                    f"Primary {primary['entity_type']} #{primary_id} is concrete and has no "
                    "open graph-recorded proof obligation or prior attack."
                ),
            )

    non_primary_prove_candidates = [
        entity
        for entity in prove_candidates
        if not (
            int(entity["id"]) == primary_id
            and primary["entity_type"] in PRECISE_ENTITY_TYPES
        )
    ]
    consolidation = _route_consolidation_choice(context, workstream_id, primary_id, history, open_obligations)
    if consolidation is not None and all(
        _attribute(context, int(entity["id"]), "research_artifact_type") == "protocol_component"
        and _attribute(context, int(entity["id"]), "research_branch_status") == "unresolved"
        for entity in non_primary_prove_candidates
    ):
        return consolidation
    if non_primary_prove_candidates:
        target_id = int(non_primary_prove_candidates[0]["id"])
        return OperationChoice(
            operation="prove",
            target_entity_id=target_id,
            rationale=(
                f"Candidate #{target_id} is precise enough for proof work and has no recorded "
                "controller proof attempt."
            ),
        )

    return OperationChoice(
        operation="develop",
        target_entity_id=primary_id,
        rationale=(
            f"Primary object #{primary_id} has no unattacked proof candidate and no currently "
            "actionable synthesis obligation; extend the technical frontier."
        ),
    )


def _is_primary_synthesis(context: ResearchContext, choice: OperationChoice) -> bool:
    workstream = getattr(context, "workstream", None)
    return bool(
        choice.operation == "synthesize" and choice.focus_obligation_id is None
        and workstream is not None
        and choice.target_entity_id == int(_primary_target(context, int(workstream["id"]))["id"])
    )


def _can_reframe(
    context: ResearchContext, workstream_id: int, obligation_id: int, history: tuple[dict, ...],
) -> bool:
    return (
        obligation_id not in _input_ids(context, workstream_id)
        and not _has_terminal_branch_state(context, obligation_id)
        and not _attribute(context, obligation_id, "research_necessity_audit_state")
        and not _completed_for_target(history, "reframe", obligation_id)
        and any(int(entity["id"]) == obligation_id and entity["status"] == "active"
                and entity["trust_state"] != "contradicted" for entity in context.entities)
    )


def _reframe_candidate_id(context: ResearchContext, workstream_id: int, obligation_id: int) -> int:
    if obligation_id in _input_ids(context, workstream_id):
        raise TheoryError("A problem-contract input cannot be bypassed by a necessity audit.")
    raw = _attribute(context, obligation_id, "research_reframe_candidate_id")
    try:
        candidate_id = int(raw)
    except (TypeError, ValueError) as exc:
        raise TheoryError("Pending reframe obligation has no candidate ID.") from exc
    candidate = next((entity for entity in context.entities if int(entity["id"]) == candidate_id), None)
    if (candidate is None or candidate["entity_type"] != "Finding"
            or candidate["status"] != "active" or candidate["trust_state"] == "contradicted"
            or candidate_id not in _linked_ids(context, workstream_id)
            or _has_terminal_branch_state(context, candidate_id)
            or _attribute(context, candidate_id, "research_reframe_target_obligation_id") != str(obligation_id)):
        raise TheoryError("Pending reframe obligation has an invalid audit candidate.")
    return candidate_id


def _is_reframe_attack(context: ResearchContext, choice: OperationChoice) -> bool:
    focus = choice.focus_obligation_id
    return (
        choice.operation == "attack" and focus is not None
        and _attribute(context, focus, "research_obligation_state") == "reframe_pending_attack"
        and _attribute(context, focus, "research_reframe_candidate_id") == str(choice.target_entity_id)
        and _attribute(context, choice.target_entity_id, "research_reframe_target_obligation_id") == str(focus)
    )


def bypass_route_is_live(context: ResearchContext, obligation_id: int, candidate_id: int) -> bool:
    """A bounded surviving route, never a claim of universal non-necessity.

    Missing legacy route provenance cannot establish a live bypass. An explicit
    empty replacement list means a directly discharged parent requirement.
    """
    attrs = context.attributes.get(candidate_id, {})
    if (attrs.get("research_reframe_target_obligation_id") != str(obligation_id)
            or not attrs.get("research_bypass_activated_iteration_id")
            or not _route_entity_is_live(context, candidate_id)):
        return False
    raw = attrs.get("research_bypass_replacement_obligation_ids")
    if raw is None:
        return False
    replacements = json.loads(raw)
    if not isinstance(replacements, list) or any(type(i) is not int or i <= 0 for i in replacements):
        raise TheoryError("Invalid persisted bypass replacement provenance.")
    return not replacements or any(_route_entity_is_live(context, i) for i in replacements)


def reactivate_bypassed_obligations(workstream_id: int) -> tuple[ProgressEvent, ...]:
    """Reconcile persisted routes without a model call or synthetic iteration.

    Called before scheduling and after scientific writes. Out-of-iteration
    transitions retain append-only event metadata on the original obligation;
    in-iteration transitions also enter the ordinary typed progress record.
    """
    context = for_workstream(workstream_id)
    reopened: list[ProgressEvent] = []
    with connect() as con:
        for obligation_id in sorted(_linked_ids(context, workstream_id)):
            attrs = context.attributes.get(obligation_id, {})
            if attrs.get("research_obligation_state") not in {"bypassed", "unnecessary"}:
                continue
            routes = set(_stored_id_set(attrs.get("research_bypass_candidate_ids")))
            current = attrs.get("research_reframe_candidate_id")
            if current is not None:
                routes.add(int(current))
            if any(bypass_route_is_live(context, obligation_id, candidate) for candidate in sorted(routes)):
                continue
            event = ProgressEvent(kind="obligation_reactivated", entity_ids=(obligation_id,),
                                  obligation_ids=(obligation_id,))
            log = json.loads(attrs.get("research_bypass_reactivation_events", "[]"))
            log.append({**event.model_dump(mode="json"), "previous_state": attrs["research_obligation_state"],
                        "candidate_ids": sorted(routes), "timestamp": utcnow()})
            set_attribute(con, obligation_id, "research_bypass_reactivation_events", json.dumps(log))
            set_attribute(con, obligation_id, "research_obligation_state", "open")
            set_attribute(con, obligation_id, "research_necessity_audit_state", "reactivated")
            reopened.append(event)
    return tuple(reopened)


def generate_legal_research_moves(
    context: ResearchContext,
    workstream_id: int,
    primary: dict,
    history: tuple[dict, ...],
    *,
    strategy_enabled: bool = True,
) -> tuple[LegalResearchMove, ...]:
    """Pure frontier enumeration, with one optional strategic branch escape."""
    history = _scientific_history(history)
    baseline = choose_next_operation(context, workstream_id, primary, history)
    open_ids = _open_obligation_ids(context, workstream_id, int(primary["id"]))
    consolidation = _route_consolidation_choice(context, workstream_id, int(primary["id"]), history, open_ids)
    if open_ids:
        choices = ([baseline] if baseline.continue_construction else []) + [
            _obligation_frontier_choice(
                context, workstream_id, history, open_ids, obligation, exclude_terminal=True
            )
            for obligation in eligible_open_obligation_ids(context, open_ids)
        ]
        for obligation in eligible_open_obligation_ids(context, open_ids):
            if _can_reframe(context, workstream_id, obligation, history):
                choices.append(OperationChoice(
                    "reframe", obligation,
                    "Audit whether this provisional obligation is required by the problem contract.",
                    open_obligation_ids=open_ids, focus_obligation_id=obligation,
                ))
        if strategy_enabled:
            choices.append(OperationChoice(
                operation="develop",
                target_entity_id=int(primary["id"]),
                rationale="Explore a genuinely different top-level route from the supplied problem contract without assuming the current obligation decomposition.",
                open_obligation_ids=open_ids,
                focus_obligation_id=None,
                develop_provenance="frontier",
            ))
    else:
        choices = [baseline]
        # Preserve the existing concrete, unattacked proof-candidate alternatives.
        if baseline.operation == "attack":
            linked = _linked_ids(context, workstream_id)
            for entity in sorted(context.entities, key=lambda entity: -int(entity["id"])):
                entity_id = int(entity["id"])
                if (
                    entity_id in linked
                    and entity_id != baseline.target_entity_id
                    and entity["entity_type"] == "ProofAttempt"
                    and _is_precise_candidate(context, entity)
                    and _construction_entity_is_active(context, entity_id)
                    and not _has_terminal_branch_state(context, entity_id)
                    and not _completed_for_target(history, "attack", entity_id)
                ):
                    choices.append(OperationChoice(
                        "attack", entity_id,
                        f"Proof candidate #{entity_id} is precise and has no completed bounded attack.",
                    ))
        if strategy_enabled and (baseline.operation in {"prove", "attack"} or consolidation is not None):
            choices.append(OperationChoice(
                operation="develop",
                target_entity_id=int(primary["id"]),
                rationale=(
                    "Explore another top-level route from the supplied problem contract "
                    "instead of committing immediately to the current local candidate."
                ),
                develop_provenance="frontier",
            ))
    if consolidation is not None:
        choices.append(consolidation)
    if strategy_enabled:
        for bundle in _primary_synthesis_bundles(context, workstream_id, int(primary["id"]), history):
            choices.append(OperationChoice(
                operation="synthesize", target_entity_id=int(primary["id"]),
                rationale="Combine complementary branch results against the supplied problem contract to form a new top-level candidate.",
                consumed_entity_ids=bundle, open_obligation_ids=open_ids,
            ))
    moves_by_id: dict[str, LegalResearchMove] = {}
    for choice in choices:
        if (choice.operation in {"attack", "prove"}
                and _has_terminal_branch_state(context, choice.target_entity_id)):
            continue
        move = LegalResearchMove.from_choice(choice)
        moves_by_id.setdefault(move.move_id, move)
    moves = tuple(moves_by_id.values())
    _baseline_legal_move(baseline, moves)
    return moves


def _baseline_legal_move(
    baseline: OperationChoice, legal_moves: tuple[LegalResearchMove, ...]
) -> LegalResearchMove:
    if len({move.move_id for move in legal_moves}) != len(legal_moves):
        raise TheoryError("Internal controller error: duplicate legal move IDs.")
    expected = LegalResearchMove.from_choice(baseline)
    if expected not in legal_moves:
        raise TheoryError("Internal controller error: deterministic baseline is not a legal move.")
    return expected


def validate_strategist_decision(
    decision: StrategistDecision, legal_moves: tuple[LegalResearchMove, ...]
) -> LegalResearchMove:
    move_by_id = {move.move_id: move for move in legal_moves}
    if decision.selected_move_id not in move_by_id:
        raise ModelOutputError(
            f"Invalid strategist selected_move_id: {decision.selected_move_id!r}; "
            "expected exactly one offered legal move ID."
        )
    return move_by_id[decision.selected_move_id]


def select_research_move(
    *,
    baseline_choice: OperationChoice,
    legal_moves: tuple[LegalResearchMove, ...],
    strategy_enabled: bool,
    decision: StrategistDecision | None = None,
    strategy_model: str | None = None,
) -> ResearchSelection | None:
    """Pure selection; None requests one strategist call from the orchestrator."""
    baseline = _baseline_legal_move(baseline_choice, legal_moves)
    ids = tuple(move.move_id for move in legal_moves)
    # Explicit ablations retain their audit label even for a singleton frontier.
    if not strategy_enabled:
        return ResearchSelection(baseline, "deterministic_baseline", ids, baseline.rationale)
    if len(legal_moves) == 1:
        return ResearchSelection(
            legal_moves[0], "single_legal_move", ids,
            f"Only one legal move is available. {legal_moves[0].rationale}",
        )
    if decision is None:
        return None
    return ResearchSelection(
        validate_strategist_decision(decision, legal_moves), "strategist", ids,
        decision.rationale, "openai", strategy_model,
    )


def _problem_contract(context: ResearchContext, workstream_id: int) -> tuple[ProblemContractBrief, ...]:
    inputs = _input_ids(context, workstream_id)
    if inputs - {int(entity["id"]) for entity in context.entities}:
        raise TheoryError("Problem-contract input entities are missing from the supplied context.")
    return tuple(ProblemContractBrief(
        id=int(entity["id"]), entity_type=entity["entity_type"], title=entity["title"],
        body=entity.get("body") or "", trust_state=entity["trust_state"],
    ) for entity in sorted(context.entities, key=lambda entity: int(entity["id"]))
        if int(entity["id"]) in inputs and entity["entity_type"] != "Paper")


def _obligation_statement(context: ResearchContext, entity_id: int) -> str:
    entity = next(entity for entity in context.entities if int(entity["id"]) == entity_id)
    stored = _attribute(context, entity_id, "research_statement")
    if stored is not None:
        return stored
    body = entity.get("body") or ""
    # Legacy objects lack a separate exact statement: retain their complete body.
    return body or entity["title"]


def build_research_state(
    context: ResearchContext,
    *,
    workstream_id: int,
    primary: dict,
    history: tuple[dict, ...],
    legal_moves: tuple[LegalResearchMove, ...],
) -> ResearchState:
    """Build a deterministic compact view using only already-loaded state."""
    history = _active_obligation_history(context, workstream_id, int(primary["id"]), history)
    by_id = {int(entity["id"]): entity for entity in context.entities}

    def brief(entity: dict) -> ResearchEntityBrief:
        entity_id = int(entity["id"])
        attrs = context.attributes.get(entity_id, {})
        return ResearchEntityBrief(
            id=entity_id, entity_type=entity["entity_type"], title=entity["title"],
            status=entity["status"], trust_state=entity["trust_state"],
            branch_status=attrs.get("research_branch_status") or attrs.get("develop_branch_status"),
            obligation_state=attrs.get("research_obligation_state"),
            attack_state=attrs.get("research_attack_state"),
        )

    open_ids = _open_obligation_ids(context, workstream_id, int(primary["id"]))
    eligible = eligible_open_obligation_ids(context, open_ids)
    all_obligation_ids = set(_obligation_ids(context, workstream_id, int(primary["id"])))
    focuses = [_iteration_focus_obligation_id(context, row, all_obligation_ids) for row in history]
    obligations = tuple(ObligationBrief(
        **brief(by_id[obligation]).model_dump(),
        statement=_obligation_statement(context, obligation),
        necessity_audit_state=_attribute(context, obligation, "research_necessity_audit_state"),
        eligible=obligation in eligible,
        last_focused_iteration=next((
            row.get("iteration_number")
            for row, focus in reversed(list(zip(history, focuses))) if focus == obligation
        ), None),
        linked_unattacked_candidate_ids=tuple(int(entity["id"]) for entity in
            _relevant_unattacked_proof_attempts(context, history, (obligation,))
            if not _has_terminal_branch_state(context, int(entity["id"]))),
    ) for obligation in open_ids)
    move_entity_ids = {
        entity_id for move in legal_moves
        for entity_id in (move.target_entity_id, *move.consumed_entity_ids,
                          *(tuple(use.entity_id for use in move.idea.exploits) if move.idea else ()))
    } - {int(primary["id"]), *open_ids}
    linked = _linked_ids(context, workstream_id)
    terminal = tuple(brief(entity) for entity_id, entity in sorted(by_id.items())
        if entity_id in linked and (
            _has_terminal_branch_state(context, entity_id)
            or _attribute(context, entity_id, "research_obligation_state") in INACTIVE_OBLIGATION_STATES
            or entity["status"] != "active" or entity["trust_state"] == "contradicted"
        ))
    recent = tuple(RecentIterationBrief(
        iteration_number=row.get("iteration_number"), operation=row["operation"],
        target_entity_id=int(row["target_entity_id"]),
        focus_obligation_id=focus,
        status=row["status"], material_progress=bool(row.get("material_progress", False)),
        duplicate_count=row.get("duplicate_count", 0), attack_outcome=row.get("attack_outcome"),
        stop_reason=row.get("stop_reason"),
        progress_class=row.get("progress_class"),
        progress_level=row.get("progress_level"),
        progress_event_kinds=progress_event_kinds(row.get("progress_events_json")),
        resolution_progress=(bool(row["resolution_progress"])
                             if row.get("resolution_progress") is not None else None),
        open_obligations_before=row.get("open_obligations_before"),
        open_obligations_after=row.get("open_obligations_after"),
        resolved_obligation_count=row.get("resolved_obligation_count"),
        new_obligation_count=row.get("new_obligation_count"),
        candidate_created_count=row.get("candidate_created_count"),
        candidate_tested_count=row.get("candidate_tested_count"),
        closed_branch_count=row.get("closed_branch_count"),
        accepted_artifact_count=row.get("accepted_artifact_count"),
    ) for row, focus in list(zip(history, focuses))[-6:])
    return ResearchState(
        primary_target=brief(primary), problem_contract=_problem_contract(context, workstream_id),
        open_obligations=obligations,
        blocked_or_terminal_branches=terminal,
        move_entities=tuple(ResearchArtifactBrief(
            **brief(by_id[entity_id]).model_dump(),
            statement=context.attributes.get(entity_id, {}).get(
                "research_statement", by_id[entity_id].get("body") or ""
            ),
        ) for entity_id in sorted(move_entity_ids)),
        legal_moves=tuple(ResearchMoveBrief(**{
            key: value for key, value in asdict(move).items() if key != "develop_provenance"
        }) for move in legal_moves),
        recent_iterations=recent,
        controller_summary=ControllerSummary(
            workstream_id=workstream_id,
            workstream_status=context.workstream["status"] if context.workstream else None,
            completed_iterations=sum(row["status"] == "completed" for row in history),
            eligible_obligation_ids=eligible,
        ),
    )


def _strategist_model(cfg: Config) -> str:
    model = cfg.research_strategist_model
    get_model_spec(model)  # Refuse unpriced models through the shared registry.
    if model != "gpt-6-luna":
        raise ConfigurationError(
            "Automatic research strategy requires OpenAI gpt-6-luna; "
            f"{model!r} is not permitted for this milestone."
        )
    get_model_spec(model, "openai")
    return model


def _research_prompt(
    context: ResearchContext, primary: dict, choice: OperationChoice
) -> str:
    return _research_prompt_sections(context, primary, choice).render()


def _research_prompt_sections(
    context: ResearchContext, primary: dict, choice: OperationChoice,
    *, attack_response_format: Literal["variant", "flat"] = "variant",
) -> PromptSections:
    return build_research_sections(context, primary, choice, facts=ResearchPromptFacts(
        constructive_continuation=choice.continue_construction,
        primary_synthesis=_is_primary_synthesis(context, choice),
        reframe_attack=_is_reframe_attack(context, choice),
        human_judgment_allowed=_human_judgment_allowed(context, choice),
        attack_response_format=attack_response_format,
        problem_contract=(
            _problem_contract(context, int(context.workstream["id"]))
            if choice.operation == "reframe" else ()
        ),
    ))


def _has_meaningful_open_obligation_transition(
    artifacts: Sequence[ResearchArtifact],
    addressed_obligation_ids: Sequence[int],
    open_obligation_ids: Sequence[int],
) -> bool:
    open_ids = set(open_obligation_ids)
    addressed_open_ids = open_ids & set(addressed_obligation_ids)
    if addressed_open_ids and any(
        artifact.artifact_type in PROOF_ARTIFACT_TYPES
        and addressed_open_ids & set(artifact.related_entity_ids)
        for artifact in artifacts
    ):
        return True
    return any(
        artifact.artifact_type
        in {"proof_obligation", "counterexample", "obstruction", "failed_approach"}
        for artifact in artifacts
    )


def _validate_step_report(
    report: ResearchStepReport,
    context: ResearchContext,
    choice: OperationChoice,
) -> None:
    primary_synthesis = _is_primary_synthesis(context, choice)
    if report.operation != choice.operation:
        raise ModelOutputError(
            f"Research output chose {report.operation!r}, expected {choice.operation!r}."
        )
    if report.target_entity_id != choice.target_entity_id:
        raise ModelOutputError(
            f"Research output targeted entity #{report.target_entity_id}, expected "
            f"#{choice.target_entity_id}."
        )
    if (
        (report.human_judgment_required or report.human_judgment_reason is not None)
        and not _human_judgment_allowed(context, choice)
    ):
        raise ModelOutputError(
            "Human judgment is permitted only for develop targeting a supplied role=input "
            "problem-contract entity. This operation requires human_judgment_required=false "
            "and human_judgment_reason=null; record candidate-specific missing premises in artifacts."
        )
    allowed_entity_ids = {int(entity["id"]) for entity in context.entities}
    allowed_source_ids = {int(source["id"]) for source in context.sources}
    if choice.operation != "reframe":
        if (report.necessity_outcome != "not_applicable" or report.necessity_contract_entity_ids
                or report.necessity_audit is not None):
            raise ModelOutputError("Only reframe may report a necessity outcome or contract IDs.")
    else:
        if context.workstream is None:
            raise ModelOutputError("Reframe requires a workstream contract.")
        workstream_id = int(context.workstream["id"])
        contract_bodies = {entity.id: entity.body for entity in _problem_contract(context, workstream_id)}
        contract_ids = set(contract_bodies)
        cited = set(report.necessity_contract_entity_ids)
        if (choice.focus_obligation_id != choice.target_entity_id
                or choice.target_entity_id not in choice.open_obligation_ids
                or choice.target_entity_id in _input_ids(context, workstream_id)
                or not _can_reframe(context, workstream_id, choice.target_entity_id, ())):
            raise ModelOutputError("Reframe requires an unaudited generated open obligation.")
        if report.necessity_outcome == "not_applicable" or not cited or not cited <= contract_ids:
            raise ModelOutputError("Reframe requires a necessity outcome and supplied role=input contract IDs.")
        audit = report.necessity_audit
        if audit is None:
            raise ModelOutputError("Reframe requires an inspectable contract-grounded necessity_audit.")
        if {clause.entity_id for clause in audit.contract_clauses} != cited:
            raise ModelOutputError("Every materially used contract clause must be explicitly cited.")
        for clause in audit.contract_clauses:
            if not clause.quote.strip() or clause.quote not in contract_bodies[clause.entity_id]:
                raise ModelOutputError("Contract grounding requires exact quotations from the cited input body.")
        parent = audit.parent_requirement
        if parent.entity_id not in cited:
            raise ModelOutputError("The parent requirement must cite a contract entity in contract_clauses.")
        if not parent.quote.strip() or parent.quote not in contract_bodies[parent.entity_id]:
            raise ModelOutputError("The parent requirement must quote an exact nonempty contract substring.")
        if not any(
            clause.entity_id == parent.entity_id
            and (parent.quote in clause.quote or clause.quote in parent.quote)
            for clause in audit.contract_clauses
        ):
            raise ModelOutputError(
                "The parent requirement must have compatible textual scope with a cited contract clause."
            )
        used_by_artifacts = set().union(*(set(a.related_entity_ids) for a in report.artifacts)) & contract_ids
        if not used_by_artifacts <= cited:
            raise ModelOutputError("Artifacts reference materially used contract inputs omitted from the audit.")
        replacement_keys = [a.material_key for a in report.artifacts if a.artifact_type == "proof_obligation"]
        if (len(set(audit.replacement_obligation_keys)) != len(audit.replacement_obligation_keys)
                or set(audit.replacement_obligation_keys) != set(replacement_keys)):
            raise ModelOutputError("Every unresolved replacement premise must name an emitted proof obligation.")
        if report.addressed_obligation_ids:
            raise ModelOutputError("Reframe cannot claim to prove an obligation via addressed_obligation_ids.")
        for artifact in report.artifacts:
            if artifact.artifact_type == "proof_obligation" and not cited.intersection(artifact.related_entity_ids):
                raise ModelOutputError("Replacement obligations must reference a cited contract entity.")
        if report.necessity_outcome == "alternative_route_found" and not any(
            artifact.artifact_type == "finding" and cited <= set(artifact.related_entity_ids)
            for artifact in report.artifacts
        ):
            raise ModelOutputError("Alternative route requires a concrete finding citing the contract.")
    obligation_ids = set(choice.open_obligation_ids)
    if (
        choice.focus_obligation_id is not None
        and choice.focus_obligation_id not in obligation_ids
    ):
        raise ModelOutputError(
            f"Controller focus obligation #{choice.focus_obligation_id} is not open."
        )
    if primary_synthesis and report.addressed_obligation_ids:
        raise ModelOutputError("Primary synthesis cannot address proof obligations.")
    unknown_addressed = set(report.addressed_obligation_ids) - obligation_ids
    if unknown_addressed:
        raise ModelOutputError(
            "Research output addressed unknown or non-open obligation IDs: "
            + ", ".join(str(value) for value in sorted(unknown_addressed))
        )
    reported_consumed_ids = set(report.consumed_entity_ids)
    unknown_consumed = reported_consumed_ids - allowed_entity_ids
    if unknown_consumed:
        raise ModelOutputError(
            "Research output consumed unknown/out-of-context entity IDs: "
            + ", ".join(str(value) for value in sorted(unknown_consumed))
        )
    if choice.operation == "synthesize":
        if reported_consumed_ids != set(choice.consumed_entity_ids):
            raise ModelOutputError(
                "Research output did not return exactly the controller-selected "
                "consumed_entity_ids."
            )
    elif report.consumed_entity_ids:
        raise ModelOutputError(
            "Only synthesize may return non-empty consumed_entity_ids."
        )
    for index, artifact in enumerate(report.artifacts, start=1):
        unknown_entities = (set(artifact.related_entity_ids) | set(artifact.refutes_entity_ids)) - allowed_entity_ids
        if unknown_entities:
            raise ModelOutputError(
                f"Research artifact {index} referenced unknown/out-of-context entity IDs: "
                + ", ".join(str(value) for value in sorted(unknown_entities))
            )
        if not set(artifact.refutes_entity_ids) <= set(artifact.related_entity_ids):
            raise ModelOutputError(
                f"Research artifact {index} must include every explicitly refuted entity in related_entity_ids."
            )
        if choice.target_entity_id not in artifact.related_entity_ids:
            raise ModelOutputError(
                f"Research artifact {index} did not reference selected target "
                f"#{choice.target_entity_id}."
            )
        unknown_sources = set(artifact.source_ids) - allowed_source_ids
        if unknown_sources:
            raise ModelOutputError(
                f"Research artifact {index} referenced unknown/out-of-context source IDs: "
                + ", ".join(str(value) for value in sorted(unknown_sources))
            )

    if choice.operation == "attack":
        if report.addressed_obligation_ids:
            raise ModelOutputError("Attack output cannot claim to address proof obligations.")
        if report.attack_outcome == "not_applicable":
            raise ModelOutputError("Attack output must state its bounded attack outcome.")
        critical = any(
            artifact.artifact_type in CRITICAL_ARTIFACT_TYPES
            for artifact in report.artifacts
        )
        if report.attack_outcome == "critical_issue" and not critical:
            raise ModelOutputError(
                "critical_issue requires a concrete counterexample, obstruction, or failed approach."
            )
        if report.attack_outcome == "inconclusive":
            if critical:
                raise ModelOutputError(
                    "Critical attack artifacts require attack_outcome='critical_issue'."
                )
            if not report.could_not_determine:
                raise ModelOutputError(
                    "inconclusive requires non-empty could_not_determine uncertainty."
                )
        if report.attack_outcome == "no_critical_issue" and (
            critical or report.could_not_determine
        ):
            raise ModelOutputError(
                "no_critical_issue cannot accompany a critical artifact or unresolved point."
            )
    elif report.attack_outcome != "not_applicable":
        raise ModelOutputError("Only attack may return an attack outcome.")

    if choice.operation == "synthesize":
        if len(choice.consumed_entity_ids) < 2:
            raise ModelOutputError("Synthesize requires at least two existing artifacts.")
        if primary_synthesis and len(choice.consumed_entity_ids) > 4:
            raise ModelOutputError("Primary synthesis consumes at most four existing artifacts.")
        required_refs = set(choice.consumed_entity_ids) | {choice.target_entity_id}
        if not report.artifacts:
            raise ModelOutputError("Synthesize must persist an attempted result or failure.")
        for index, artifact in enumerate(report.artifacts, start=1):
            if not required_refs <= set(artifact.related_entity_ids):
                raise ModelOutputError(
                    f"Synthesis artifact {index} did not reference every consumed artifact "
                    + ("and the primary target." if primary_synthesis else "and the target obligation.")
                )
        if not primary_synthesis and choice.target_entity_id not in report.addressed_obligation_ids and not any(
            artifact.artifact_type in {"failed_approach", "obstruction"}
            for artifact in report.artifacts
        ):
            raise ModelOutputError(
                "An unsuccessful synthesis must persist a failed approach or obstruction."
            )

    if choice.operation == "prove":
        if not report.artifacts:
            raise ModelOutputError("Prove must persist a proof attempt, obligation, or failure.")
        if choice.focus_obligation_id is not None:
            for index, artifact in enumerate(report.artifacts, start=1):
                if (
                    artifact.artifact_type in PROOF_ARTIFACT_TYPES
                    and choice.focus_obligation_id not in artifact.related_entity_ids
                ):
                    raise ModelOutputError(
                        f"Focused prove artifact {index} did not reference obligation "
                        f"#{choice.focus_obligation_id}."
                    )
        if choice.open_obligation_ids and not _has_meaningful_open_obligation_transition(
            report.artifacts,
            report.addressed_obligation_ids,
            choice.open_obligation_ids,
        ):
            raise ModelOutputError(
                "Prove with open obligations must address one with a lemma/proof attempt, "
                "create a new proof obligation, or record a counterexample, obstruction, "
                "or failed approach."
            )
    if choice.operation == "develop" and not report.artifacts:
        raise ModelOutputError("Develop must propose at least one substantive artifact.")

    if report.addressed_obligation_ids:
        proof_artifacts = [
            artifact
            for artifact in report.artifacts
            if artifact.artifact_type in PROOF_ARTIFACT_TYPES
        ]
        for obligation_id in report.addressed_obligation_ids:
            if not any(
                obligation_id in artifact.related_entity_ids for artifact in proof_artifacts
            ):
                raise ModelOutputError(
                    f"Addressed obligation #{obligation_id} lacks a lemma or proof attempt."
                )


def _normalized_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in _STOP_WORDS
    )


def _is_lexical_duplicate(
    tokens: frozenset[str], existing_fingerprints: list[frozenset[str]]
) -> bool:
    if not tokens:
        return True
    for existing in existing_fingerprints:
        if tokens == existing:
            return True
        union = tokens | existing
        intersection = tokens & existing
        if len(intersection) >= 3 and len(intersection) / len(union) >= 0.86:
            return True
    return False


def _existing_duplicate_state(
    context: ResearchContext,
) -> tuple[set[str], list[frozenset[str]]]:
    keys = {
        attrs["research_material_key"]
        for attrs in context.attributes.values()
        if attrs.get("research_material_key")
    }
    fingerprints: list[frozenset[str]] = []
    for entity in context.entities:
        body = str(entity.get("body") or "")
        statement = body.split("\n\n", 1)[0].strip()
        if not statement:
            statement = str(entity["title"]).split(":", 1)[-1].strip()
        fingerprints.append(_normalized_tokens(statement))
    return keys, fingerprints


def _artifact_title(artifact: ResearchArtifact) -> str:
    prefix = artifact.artifact_type.replace("_", " ").title()
    return f"{prefix}: {artifact.statement}"[:240]


def _persist_step(
    *,
    iteration_id: int,
    workstream_id: int,
    provider_name: str,
    model: str,
    choice: OperationChoice,
    context: ResearchContext,
    report: ResearchStepReport,
) -> PersistedStep:
    existing_keys, fingerprints = _existing_duplicate_state(context)
    artifact_ids: list[int] = []
    accepted: list[tuple[ResearchArtifact, int]] = []
    attempted_by_entity: dict[int, tuple[int, ...]] = {}
    reused_obligation_ids: list[int] = []
    duplicate_count = 0
    with connect() as con:
        for artifact in report.artifacts:
            tokens = _normalized_tokens(artifact.statement)
            if artifact.material_key in existing_keys or _is_lexical_duplicate(
                tokens, fingerprints
            ):
                duplicate_count += 1
                reused_obligation_ids.extend(
                    i for i, attrs in context.attributes.items()
                    if artifact.artifact_type == "proof_obligation"
                    and attrs.get("research_material_key") == artifact.material_key
                    and attrs.get("research_artifact_type") == "proof_obligation"
                )
                continue
            entity_id = add_entity(
                con,
                ARTIFACT_ENTITY_TYPES[artifact.artifact_type],
                _artifact_title(artifact),
                body=(
                    f"{artifact.statement}\n\nReasoning: {artifact.reasoning_summary}\n\n"
                    f"Controller operation: {choice.operation}\n"
                    f"Model epistemic status: {artifact.epistemic_status}\n"
                    f"Related entity IDs: {artifact.related_entity_ids}\n"
                    f"Cited source IDs: {artifact.source_ids}"
                ),
                trust_state="quarantined",
                source_ids=artifact.source_ids,
                generated_by_llm=True,
            )
            if choice.idea is not None:
                set_attribute(con, entity_id, "research_ideation_call_id", str(choice.ideation_call_id))
                set_attribute(con, entity_id, "research_selected_idea", choice.idea.model_dump_json())
            if choice.operation == "develop":
                set_attribute(con, entity_id, "research_develop_provenance",
                              choice.develop_provenance)
            set_attribute(con, entity_id, "research_artifact_type", artifact.artifact_type)
            set_attribute(con, entity_id, "research_statement", artifact.statement)
            set_attribute(con, entity_id, "research_operation", choice.operation)
            set_attribute(con, entity_id, "research_material_key", artifact.material_key)
            set_attribute(
                con, entity_id, "model_epistemic_status", artifact.epistemic_status
            )
            set_attribute(
                con,
                entity_id,
                "related_entity_ids",
                json.dumps(artifact.related_entity_ids),
            )
            if choice.focus_obligation_id is not None:
                set_attribute(
                    con,
                    entity_id,
                    "research_focus_obligation_id",
                    str(choice.focus_obligation_id),
                )
            if artifact.branch_status is not None:
                set_attribute(
                    con, entity_id, "research_branch_status", artifact.branch_status
                )
            if artifact.artifact_type == "proof_obligation":
                set_attribute(con, entity_id, "is_proof_obligation", "true")
                set_attribute(con, entity_id, "research_obligation_state", "open")
            if artifact.artifact_type in {"lemma", "protocol_component", "proof_attempt"}:
                set_attribute(con, entity_id, "precise_candidate", "true")
            link_workstream_entity(con, workstream_id, entity_id, "created")
            artifact_ids.append(entity_id)
            accepted.append((artifact, entity_id))
            # Explicit structured scope is authoritative. Relevance references,
            # including contract/route ancestors, never create closure edges.
            for refuted_id in artifact.refutes_entity_ids:
                add_relation(con, entity_id, "REFUTES", refuted_id,
                             trust_state="quarantined", generated_by_llm=True)
            existing_keys.add(artifact.material_key)
            fingerprints.append(tokens)

        pending_attack_obligation_ids: set[int] = set()
        for artifact, entity_id in accepted:
            if artifact.artifact_type not in PROOF_ARTIFACT_TYPES:
                continue
            addressed_ids = {
                obligation_id
                for obligation_id in report.addressed_obligation_ids
                if obligation_id in artifact.related_entity_ids
            }
            attempted_ids = set(addressed_ids)
            if artifact.artifact_type == "proof_attempt":
                if (
                    choice.focus_obligation_id is not None
                    and choice.focus_obligation_id in artifact.related_entity_ids
                ):
                    attempted_ids.add(choice.focus_obligation_id)
                pending_attack_obligation_ids.update(attempted_ids)
            for obligation_id in sorted(attempted_ids):
                add_relation(
                    con,
                    entity_id,
                    "ATTEMPTS",
                    obligation_id,
                    trust_state="quarantined",
                    generated_by_llm=True,
                )
            attempted_by_entity[entity_id] = tuple(sorted(attempted_ids))
            if addressed_ids:
                set_attribute(
                    con,
                    entity_id,
                    "addresses_obligation_ids",
                    json.dumps(sorted(addressed_ids)),
                )

        for obligation_id in pending_attack_obligation_ids:
            set_attribute(
                con,
                obligation_id,
                "research_obligation_state",
                "candidate_pending_attack",
            )

        if choice.operation == "reframe":
            target = choice.target_entity_id
            audit_state = report.necessity_outcome
            if audit_state == "alternative_route_found":
                contract_ids = set(report.necessity_contract_entity_ids)
                audit_candidates = [entity_id for artifact, entity_id in accepted
                                    if artifact.artifact_type == "finding"
                                    and contract_ids <= set(artifact.related_entity_ids)]
                if not audit_candidates:
                    raise ModelOutputError("No non-duplicate alternative-route finding survived persistence.")
                candidate_id = audit_candidates[0]
                assert report.necessity_audit is not None
                replacements = sorted(entity_id for artifact, entity_id in accepted
                                      if artifact.artifact_type == "proof_obligation")
                if len(replacements) != len(report.necessity_audit.replacement_obligation_keys):
                    raise ModelOutputError("Every replacement premise must survive duplicate filtering.")
                set_attribute(con, candidate_id, "research_reframe_target_obligation_id", str(target))
                set_attribute(con, candidate_id, "research_bypass_replacement_obligation_ids", json.dumps(replacements))
                set_attribute(con, candidate_id, "research_necessity_audit", report.necessity_audit.model_dump_json())
                set_attribute(con, candidate_id, "research_necessity_contract_entity_ids",
                              json.dumps(report.necessity_contract_entity_ids))
                set_attribute(con, target, "research_reframe_candidate_id", str(candidate_id))
                set_attribute(con, target, "research_obligation_state", "reframe_pending_attack")
                audit_state = "bypass_candidate"
            set_attribute(con, target, "research_necessity_audit_state", audit_state)
            set_attribute(con, target, "research_necessity_audit_iteration_id", str(iteration_id))

        if choice.operation == "attack":
            # A bounded critical attack explicitly challenges its selected
            # candidate, independently of any additional contextual references.
            if report.attack_outcome == "critical_issue":
                for artifact, entity_id in accepted:
                    if (artifact.artifact_type in CRITICAL_ARTIFACT_TYPES
                            and choice.target_entity_id not in artifact.refutes_entity_ids):
                        add_relation(con, entity_id, "REFUTES", choice.target_entity_id,
                                     trust_state="quarantined", generated_by_llm=True)
            review_result = {
                "critical_issue": "issue_found",
                "no_critical_issue": "no_flaw_found",
                "inconclusive": "inconclusive",
            }[report.attack_outcome]
            candidate_attack_state = {
                "critical_issue": "challenged",
                "inconclusive": "inconclusive",
                "no_critical_issue": "survived_attack",
            }[report.attack_outcome]
            set_attribute(
                con,
                choice.target_entity_id,
                "research_attack_state",
                candidate_attack_state,
            )
            add_review(
                con,
                "counterexample_attempt",
                review_result,
                workstream_id=workstream_id,
                issues=(
                    f"Controller attack on entity #{choice.target_entity_id}: {report.summary} "
                    "This bounded result is not proof verification."
                ),
                provider=provider_name,
                model=model,
            )
            if _is_reframe_attack(context, choice):
                # Only the independently tested route can activate a reversible bypass.
                can_bypass = (report.attack_outcome == "no_critical_issue"
                               and not _candidate_has_unresolved_critical_issue(context, choice.target_entity_id))
                set_attribute(con, choice.focus_obligation_id, "research_obligation_state",
                              "bypassed" if can_bypass else "open")
                set_attribute(con, choice.focus_obligation_id, "research_necessity_audit_state",
                              "bypassed" if can_bypass else
                              "challenged" if report.attack_outcome == "critical_issue" else "inconclusive")
                if can_bypass:
                    routes = set(_stored_id_set(_attribute(context, choice.focus_obligation_id,
                                                        "research_bypass_candidate_ids")))
                    routes.add(choice.target_entity_id)
                    set_attribute(con, choice.focus_obligation_id, "research_bypass_candidate_ids",
                                  json.dumps(sorted(routes)))
                    set_attribute(con, choice.target_entity_id, "research_bypass_activated_iteration_id", str(iteration_id))
            elif choice.focus_obligation_id is not None:
                obligation_state = "challenged"
                if report.attack_outcome == "no_critical_issue":
                    if _candidate_can_complete_obligation_after_attack(
                        context,
                        choice.target_entity_id,
                        choice.focus_obligation_id,
                    ):
                        obligation_state = "resolved_candidate"
                        set_attribute(
                            con,
                            choice.focus_obligation_id,
                            "research_surviving_candidate_id",
                            str(choice.target_entity_id),
                        )
                    else:
                        obligation_state = "open"
                set_attribute(
                    con,
                    choice.focus_obligation_id,
                    "research_obligation_state",
                    obligation_state,
                )

        # Retain the accepted write receipt even if subsequent progress derivation
        # fails. Completion belongs to the later metrics transaction.
        con.execute(
            """
            UPDATE research_iterations
            SET artifact_ids_json=?,duplicate_count=?,attack_outcome=?,necessity_outcome=?,
                necessity_contract_entity_ids_json=?,necessity_audit_summary=? WHERE id=?
            """,
            (json.dumps(artifact_ids), duplicate_count, report.attack_outcome,
             report.necessity_outcome, json.dumps(report.necessity_contract_entity_ids),
             json.dumps({"summary": report.summary, "audit": report.necessity_audit.model_dump()}, ensure_ascii=False)
             if choice.operation == "reframe" else None, iteration_id),
        )
        receipt = dict(con.execute("SELECT * FROM research_iterations WHERE id=?", (iteration_id,)).fetchone())
        record_construction_routes(
            con, receipt, int(_primary_target(context, workstream_id)["id"]),
            reused_obligation_ids=tuple(reused_obligation_ids),
        )

    return PersistedStep(
        artifact_ids=tuple(artifact_ids),
        duplicate_count=duplicate_count,
        accepted_artifacts=tuple(AcceptedArtifactBrief(
            entity_id=entity_id, artifact_type=artifact.artifact_type,
            branch_status=artifact.branch_status,
            attempted_obligation_ids=attempted_by_entity.get(entity_id, ()),
        ) for artifact, entity_id in accepted),
    )


def build_progress_record(
    *,
    context_before: ResearchContext,
    context_after: ResearchContext,
    workstream_id: int,
    primary_id: int,
    choice: OperationChoice,
    report: ResearchStepReport,
    persisted: PersistedStep,
) -> ProgressRecord:
    """Pure classification of accepted writes against actual before/after graph state."""
    before_open = set(_open_obligation_ids(context_before, workstream_id, primary_id))
    after_open = set(_open_obligation_ids(context_after, workstream_id, primary_id))
    after_entities = {int(entity["id"]): entity for entity in context_after.entities}
    before_ids = {int(entity["id"]) for entity in context_before.entities}
    if tuple(artifact.entity_id for artifact in persisted.accepted_artifacts) != persisted.artifact_ids:
        raise TheoryError("Progress receipt does not match accepted artifact IDs.")
    events: list[ProgressEvent] = []
    if choice.operation == "attack":
        kind, attack_state = {
            "critical_issue": ("candidate_challenged", "challenged"),
            "no_critical_issue": ("candidate_survived_attack", "survived_attack"),
            "inconclusive": ("candidate_tested_inconclusive", "inconclusive"),
        }[report.attack_outcome]
        if _attribute(context_after, choice.target_entity_id, "research_attack_state") != attack_state:
            raise TheoryError("Progress attack outcome does not match persisted attack state.")
        focus = choice.focus_obligation_id
        events.append(ProgressEvent(
            kind=kind, entity_ids=(choice.target_entity_id,),
            obligation_ids=(focus,) if focus is not None else (),
        ))
        if (
            report.attack_outcome == "no_critical_issue" and focus in before_open
            and focus not in after_open
            and _attribute(context_after, focus, "research_obligation_state") == "resolved_candidate"
        ):
            events.append(ProgressEvent(
                kind="obligation_resolved", entity_ids=(choice.target_entity_id,),
                obligation_ids=(focus,),
            ))

    if choice.operation == "reframe":
        if not _attribute(context_after, choice.target_entity_id, "research_necessity_audit_state"):
            raise TheoryError("Necessity audit was not persisted.")
        events.append(ProgressEvent(kind="obligation_audited", entity_ids=(choice.target_entity_id,),
                                    obligation_ids=(choice.target_entity_id,)))
    if (_is_reframe_attack(context_before, choice)
            and choice.focus_obligation_id in before_open
            and choice.focus_obligation_id not in after_open
            and _attribute(context_after, choice.focus_obligation_id, "research_obligation_state") == "bypassed"):
        events.append(ProgressEvent(kind="obligation_bypassed", entity_ids=(choice.target_entity_id,),
                                    obligation_ids=(choice.focus_obligation_id,)))

    for obligation_id in sorted(after_open):
        before_log = _attribute(context_before, obligation_id, "research_bypass_reactivation_events")
        after_log = _attribute(context_after, obligation_id, "research_bypass_reactivation_events")
        if (after_log and before_log != after_log
                and _attribute(context_after, obligation_id, "research_necessity_audit_state") == "reactivated"):
            events.append(ProgressEvent(kind="obligation_reactivated", entity_ids=(obligation_id,),
                                        obligation_ids=(obligation_id,)))

    for artifact in persisted.accepted_artifacts:
        entity_id = artifact.entity_id
        entity = after_entities.get(entity_id)
        if entity is None or entity_id in before_ids:
            raise TheoryError("Progress artifact is not a newly persisted entity.")
        attrs = context_after.attributes.get(entity_id, {})
        if (attrs.get("research_artifact_type") != artifact.artifact_type
                or attrs.get("research_branch_status") != artifact.branch_status):
            raise TheoryError("Progress artifact receipt differs from persisted attributes.")
        attempted = tuple(sorted(set(artifact.attempted_obligation_ids) & before_open))
        persisted_attempts = {
            int(relation["target_entity_id"]) for relation in context_after.relations
            if relation["relation_type"] == "ATTEMPTS"
            and relation["status"] == "active"
            and int(relation["source_entity_id"]) == entity_id
        }
        if not set(attempted) <= persisted_attempts:
            raise TheoryError("Progress candidate is missing its persisted ATTEMPTS relation.")
        if artifact.branch_status in TERMINAL_BRANCH_STATES:
            events.append(ProgressEvent(kind="branch_closed", entity_ids=(entity_id,)))
        elif entity["entity_type"] in {"ProofAttempt", "Lemma"} and attempted:
            events.append(ProgressEvent(
                kind="candidate_created", entity_ids=(entity_id,), obligation_ids=attempted,
            ))
        elif artifact.artifact_type == "proof_obligation" and entity_id in after_open - before_open:
            events.append(ProgressEvent(
                kind="obligation_created", entity_ids=(entity_id,), obligation_ids=(entity_id,),
            ))
        else:
            events.append(ProgressEvent(kind="frontier_expanded", entity_ids=(entity_id,)))

    return ProgressRecord.from_events(
        tuple(events), open_obligations_before=len(before_open), open_obligations_after=len(after_open),
        accepted_artifact_count=len(persisted.artifact_ids), duplicate_count=persisted.duplicate_count,
    )


def _complete_iteration(
    iteration_id: int, workstream_id: int, choice: OperationChoice,
    report: ResearchStepReport, progress: ProgressRecord,
) -> None:
    fields = progress.persistence_fields()
    with connect() as con:
        cur = con.execute(
            f"UPDATE research_iterations SET status='completed',completed_at=?,"
            f"{','.join(f'{key}=?' for key in fields)} WHERE id=? AND status='running'",
            (utcnow(), *fields.values(), iteration_id),
        )
        if cur.rowcount != 1:
            raise TheoryError("Progress completion requires exactly one running iteration.")
        set_workstream_status(
            con, workstream_id, "active",
            summary=(
                f"Research iteration completed: {choice.operation}. "
                f"Progress: {progress.progress_level} / {progress.progress_class}. "
                f"Created {progress.accepted_artifact_count} quarantined artifact(s); "
                f"rejected {progress.duplicate_count} duplicate(s). {report.summary}"
            ),
        )


def _start_iteration(
    workstream_id: int, choice: OperationChoice, selection: ResearchSelection
) -> int:
    with connect() as con:
        number = int(
            con.execute(
                """
                SELECT COALESCE(MAX(iteration_number),0)+1
                FROM research_iterations WHERE workstream_id=?
                """,
                (workstream_id,),
            ).fetchone()[0]
        )
        cur = con.execute(
            """
            INSERT INTO research_iterations(
                project_id,workstream_id,iteration_number,operation,target_entity_id,
                rationale,consumed_entity_ids_json,status,created_at,
                selection_mode,legal_move_ids_json,selected_move_id,selection_rationale,
                strategy_provider,strategy_model,focus_obligation_id,develop_provenance
            ) VALUES(1,?,?,?,?,?,?,'running',?,?,?,?,?,?,?,?,?)
            """,
            (
                workstream_id,
                number,
                choice.operation,
                choice.target_entity_id,
                choice.rationale,
                json.dumps(choice.consumed_entity_ids),
                utcnow(),
                selection.selection_mode,
                json.dumps(selection.legal_move_ids),
                selection.move.move_id,
                selection.rationale,
                selection.strategy_provider,
                selection.strategy_model,
                choice.focus_obligation_id,
                choice.develop_provenance,
            ),
        )
        return int(cur.lastrowid)


def _record_iteration_error(
    iteration_id: int, workstream_id: int, error: Exception
) -> None:
    message = f"{type(error).__name__}: {error}"[:4000]
    with connect() as con:
        con.execute(
            """
            UPDATE research_iterations
            SET status='error',error_message=?,completed_at=? WHERE id=?
            """,
            (message, utcnow(), iteration_id),
        )
        set_workstream_status(
            con,
            workstream_id,
            "error",
            summary=f"Research controller execution error: {message}",
        )


def _finalize(
    *,
    workstream_id: int,
    iteration_id: int | None,
    status: str,
    stop_reason: str,
    detail: str,
) -> None:
    with connect() as con:
        if iteration_id is not None:
            con.execute(
                "UPDATE research_iterations SET stop_reason=? WHERE id=?",
                (stop_reason, iteration_id),
            )
        set_workstream_status(
            con,
            workstream_id,
            status,
            summary=f"Research controller stopped: {stop_reason}. {detail}",
        )


@contextmanager
def _research_controller_lock(workstream_id: int):
    """Prevent two local controller processes from owning one workstream."""
    lock_path = STATE_DIR / f"research-{workstream_id}.lock"
    try:
        handle = lock_path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise TheoryError(f"Cannot open research controller lock {lock_path}: {exc}") from exc
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TheoryError(
                f"Research controller for workstream #{workstream_id} is already active."
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _reconcile_stale_research(workstream_id: int) -> tuple[int, int]:
    """Close interrupted local rows without touching scientific state or telemetry."""
    completed_at = utcnow()
    with connect() as con:
        iterations = con.execute(
            """
            UPDATE research_iterations
            SET status='error',material_progress=0,error_message=?,completed_at=?
            WHERE workstream_id=? AND status='running'
            """,
            (INTERRUPTED_ITERATION_ERROR, completed_at, workstream_id),
        ).rowcount
        calls = con.execute(
            """
            UPDATE api_calls SET status='failed',error_message=?
            WHERE workstream_id=? AND status='started'
            """,
            (INTERRUPTED_API_CALL_ERROR, workstream_id),
        ).rowcount
    return iterations, calls


def research(
    workstream_id: int, provider_name: str = "auto", *, max_calls: int,
    strategy: Literal["auto", "off"] = "auto",
    max_cost_usd: float | None = None,
) -> ResearchOutcome:
    if not 1 <= max_calls <= MAX_CONTROLLER_CALLS:
        raise TheoryError(
            f"max_calls must be between 1 and {MAX_CONTROLLER_CALLS}."
        )
    if strategy not in {"auto", "off"}:
        raise ConfigurationError(f"Unknown research strategy: {strategy}")
    cfg = Config.load()
    invocation_cap = cfg.research_invocation_budget_usd if max_cost_usd is None else max_cost_usd
    if not isfinite(invocation_cap) or invocation_cap < 0:
        raise ConfigurationError("max_cost_usd must be finite and nonnegative.")
    if provider_name not in {"auto", "openai", "anthropic"}:
        raise ConfigurationError(f"Unknown provider override: {provider_name}")
    with _research_controller_lock(workstream_id):
        invocation_budget = InvocationBudget(invocation_cap)
        outcome = _run_research(
            workstream_id, provider_name, max_calls=max_calls, strategy=strategy, cfg=cfg,
            invocation_budget=invocation_budget,
        )
        return replace(
            outcome, invocation_spend_usd=invocation_budget.actual_spend_usd,
            invocation_budget_usd=invocation_cap,
        )


def _run_research(
    workstream_id: int, provider_name: str = "auto", *, max_calls: int,
    strategy: Literal["auto", "off"] = "auto", cfg: Config,
    invocation_budget: InvocationBudget,
) -> ResearchOutcome:
    """At most max_calls executions and max_calls planning calls (strategy + ideation).

    Ideation borrows one planning slot; selection falls back to the deterministic
    legal baseline when that shared allowance is exhausted. No retries or probes.
    """
    initial_context = for_workstream(workstream_id)
    workstream = initial_context.workstream
    if workstream is None:
        raise TheoryError(f"Workstream #{workstream_id} does not exist.")
    if workstream["workstream_type"] != "research":
        raise TheoryError(f"Workstream #{workstream_id} is not a research workstream.")
    if workstream["status"] != "active":
        raise TheoryError(
            f"Research workstream #{workstream_id} is {workstream['status']}, not active."
        )
    _reconcile_stale_research(workstream_id)
    primary = _primary_target(initial_context, workstream_id)

    calls_made = 0
    strategy_calls_made = 0
    ideation_calls_made = 0
    iteration_ids: list[int] = []
    artifact_ids: list[int] = []
    last_iteration_id: int | None = None
    current_run_consecutive_no_progress = 0
    providers: dict[str, Provider] = {}

    def invocation_budget_stop(purpose: str, conservative_cost: float) -> ResearchOutcome:
        _finalize(
            workstream_id=workstream_id, iteration_id=last_iteration_id,
            status="completed", stop_reason="invocation_budget_exhausted",
            detail=(f"The next {purpose} call cannot fit: "
                    f"${invocation_budget.actual_spend_usd:.4f} actual invocation spend + "
                    f"up to ${conservative_cost:.4f} > ${invocation_budget.cap_usd:.4f} invocation cap."),
        )
        return ResearchOutcome(
            workstream_id, calls_made, tuple(iteration_ids), tuple(artifact_ids),
            "invocation_budget_exhausted", "completed", strategy_calls_made, ideation_calls_made,
        )

    while calls_made < max_calls:
        reactivate_bypassed_obligations(workstream_id)
        full_context = for_workstream(workstream_id)
        history = _scientific_history(_history(workstream_id))
        if _all_branches_terminal(
            full_context, workstream_id
        ):
            _finalize(
                workstream_id=workstream_id,
                iteration_id=last_iteration_id,
                status="blocked",
                stop_reason="all_branches_blocked_or_refuted",
                detail="Every established construction route is inactive; no active obligation, candidate, or bypass work remains.",
            )
            return ResearchOutcome(
                workstream_id,
                calls_made,
                tuple(iteration_ids),
                tuple(artifact_ids),
                "all_branches_blocked_or_refuted",
                "blocked",
                strategy_calls_made,
                ideation_calls_made,
            )
        baseline_choice = choose_next_operation(full_context, workstream_id, primary, history)
        strategy_enabled = strategy == "auto" and provider_name == "auto"
        legal_moves = generate_legal_research_moves(
            full_context, workstream_id, primary, history, strategy_enabled=strategy_enabled,
        )
        ideation_call_id = None
        # Reserve both a generator and selector slot. A one-call run cannot ideate.
        if strategy_enabled and not ideation_calls_made and max_calls - strategy_calls_made >= 2:
            open_ids = _open_obligation_ids(full_context, workstream_id, int(primary["id"]))
            ideation_history = _active_obligation_history(
                full_context, workstream_id, int(primary["id"]), history,
            )
            all_obligations = set(_obligation_ids(full_context, workstream_id, int(primary["id"])))
            ideation_history = tuple({
                **row, "focus_obligation_id": _iteration_focus_obligation_id(full_context, row, all_obligations),
            } for row in ideation_history)
            previous_ideas = previous_ideations(workstream_id)
            trigger = choose_ideation_trigger(
                full_context, ideation_history, previous_ideas,
                open_obligation_ids=open_ids,
            )
            if trigger is not None and trigger.focus_obligation_id is None:
                if any(move.focus_obligation_id in open_ids
                       and move.operation in {"attack", "prove", "synthesize", "develop"}
                       for move in legal_moves):
                    trigger = None
                else:
                    current_ids = {int(e["id"]) for e in full_context.entities
                                   if e["status"] == "active" and e["trust_state"] != "contradicted"
                                   and _construction_entity_is_active(full_context, int(e["id"]))}
                    retained = tuple(i for i in trigger.entity_ids if i in current_ids)
                    if trigger.entity_ids and not retained:
                        trigger = None
                    else:
                        trigger = replace(trigger, entity_ids=retained)
                        if trigger.key in {p.get("trigger_key") for p in previous_ideas}:
                            trigger = None
            root_develop = next((move for move in legal_moves
                                 if move.operation == "develop" and move.target_entity_id == int(primary["id"])
                                 and move.focus_obligation_id is None), None)
            if trigger is not None and (trigger.focus_obligation_id is not None or root_develop is not None):
                model = cfg.research_ideation_model
                spec = get_model_spec(model)
                contract = _problem_contract(full_context, workstream_id)
                ideation_context = (
                    _obligation_ideation_context(full_context, workstream_id, int(primary["id"]), trigger.focus_obligation_id)
                    if trigger.focus_obligation_id is not None else
                    _top_level_ideation_context(full_context, workstream_id, int(primary["id"]), trigger.entity_ids)
                )
                ideation_prompt = build_ideation_prompt(ideation_context, contract, trigger)
                estimated_cost = budget_guard(
                    cfg, model=model, prompt=ideation_prompt,
                    max_output_tokens=IDEATION_MAX_OUTPUT_TOKENS,
                    purpose="research:ideate", response_model=IdeaBatch,
                )
                if not invocation_budget.can_fit(estimated_cost):
                    return invocation_budget_stop("research:ideate", estimated_cost)
                call_ids: list[int] = []
                try:
                    if spec.provider not in providers:
                        providers[spec.provider] = get_provider(spec.provider)

                    def validate_ideation(text: str) -> None:
                        validate_ideas(parse_json_model(text, IdeaBatch), ideation_context,
                                       {item.id for item in contract},
                                       focus_obligation_id=trigger.focus_obligation_id)

                    ideation_result = call_model(
                        run_id=None, workstream_id=workstream_id,
                        provider=providers[spec.provider], provider_name=spec.provider,
                        model=model, purpose="research:ideate", prompt=ideation_prompt,
                        max_output_tokens=IDEATION_MAX_OUTPUT_TOKENS,
                        estimated_max_cost_usd=estimated_cost, response_model=IdeaBatch,
                        effort="high", validate_response=validate_ideation,
                        planning_metadata=trigger.metadata(history), on_started=call_ids.append,
                        invocation_budget=invocation_budget,
                        context_scope=ideation_context.context_scope,
                    )
                    ideation_calls_made += 1
                    ideation_call_id = call_ids[0]
                    batch = parse_json_model(ideation_result.text, IdeaBatch)
                    legal_moves += tuple(LegalResearchMove(
                        "develop", trigger.focus_obligation_id or root_develop.target_entity_id,
                        focus_obligation_id=trigger.focus_obligation_id,
                        open_obligation_ids=open_ids,
                        rationale="Explore this provisional alternative under the supplied contract; retain its main risk and expose missing premises.",
                        idea=idea, ideation_call_id=ideation_call_id,
                    ) for idea in batch.ideas)
                except Exception as exc:
                    with connect() as con:
                        set_workstream_status(con, workstream_id, "error",
                            summary=f"Research ideation error: {type(exc).__name__}: {exc}"[:4000])
                    raise
        selection = select_research_move(
            baseline_choice=baseline_choice, legal_moves=legal_moves,
            strategy_enabled=strategy_enabled and strategy_calls_made + ideation_calls_made < max_calls,
        )
        if selection is None:
            model = _strategist_model(cfg)
            state = build_research_state(
                full_context, workstream_id=workstream_id, primary=primary,
                history=history, legal_moves=legal_moves,
            )
            strategy_prompt = build_strategist_sections(state).as_prompt_content()
            strategy_cost = budget_guard(
                cfg, model=model, prompt=strategy_prompt,
                max_output_tokens=STRATEGIST_MAX_OUTPUT_TOKENS, purpose="research:strategy",
                response_model=StrategistDecision,
            )
            if not invocation_budget.can_fit(strategy_cost):
                return invocation_budget_stop("research:strategy", strategy_cost)
            try:
                if "openai" not in providers:
                    providers["openai"] = get_provider("openai")

                def validate_strategy(text: str) -> None:
                    validate_strategist_decision(parse_json_model(text, StrategistDecision), legal_moves)

                result = call_model(
                    run_id=None, workstream_id=workstream_id,
                    provider=providers["openai"], provider_name="openai", model=model,
                    purpose="research:strategy", prompt=strategy_prompt,
                    max_output_tokens=STRATEGIST_MAX_OUTPUT_TOKENS,
                    estimated_max_cost_usd=strategy_cost,
                    response_model=StrategistDecision, effort="medium",
                    validate_response=validate_strategy,
                    invocation_budget=invocation_budget,
                )
                strategy_calls_made += 1
                selection = select_research_move(
                    baseline_choice=baseline_choice, legal_moves=legal_moves,
                    strategy_enabled=True,
                    decision=parse_json_model(result.text, StrategistDecision),
                    strategy_model=model,
                )
            except Exception as exc:
                with connect() as con:
                    set_workstream_status(
                        con, workstream_id, "error",
                        summary=f"Research strategy error: {type(exc).__name__}: {exc}"[:4000],
                    )
                raise
        assert selection is not None
        choice = selection.move.to_operation_choice()
        route = choose_model_route(choice, cfg, provider_override=provider_name)
        model_context = focus_research_context(
            full_context,
            workstream_id=workstream_id,
            primary_entity_id=int(primary["id"]),
            target_entity_id=choice.target_entity_id,
            focus_obligation_id=choice.focus_obligation_id,
            consumed_entity_ids=choice.consumed_entity_ids,
            additional_entity_ids=tuple(use.entity_id for use in choice.idea.exploits) if choice.idea else (),
            operation=choice.operation,
            continuation_route_ids=tuple(sorted(_constructive_continuation_streak(full_context, history)[0]))
            if choice.continue_construction else (),
        )
        response_model = _execution_response_model(choice.operation, route.provider)
        prompt = _research_prompt_sections(
            model_context, primary, choice,
            attack_response_format="flat" if response_model is FlatAttackReport else "variant",
        ).as_prompt_content()
        estimated_max_cost = budget_guard(
            cfg,
            model=route.model,
            prompt=prompt,
            max_output_tokens=route.max_output_tokens,
            purpose=f"research:{choice.operation}",
            response_model=response_model,
        )
        if not invocation_budget.can_fit(estimated_max_cost):
            return invocation_budget_stop(f"research:{choice.operation}", estimated_max_cost)
        if route.provider not in providers:
            try:
                providers[route.provider] = get_provider(route.provider)
            except Exception as exc:
                with connect() as con:
                    set_workstream_status(
                        con,
                        workstream_id,
                        "error",
                        summary=(
                            f"Research controller provider setup error: "
                            f"{type(exc).__name__}: {exc}"
                        )[:4000],
                    )
                raise
        iteration_id = _start_iteration(workstream_id, choice, selection)
        if ideation_call_id is not None:
            record_idea_selection(ideation_call_id, choice.idea.idea_id if choice.idea else None, iteration_id)
        last_iteration_id = iteration_id
        iteration_ids.append(iteration_id)
        try:
            result = call_model(
                run_id=None,
                workstream_id=workstream_id,
                provider=providers[route.provider],
                provider_name=route.provider,
                model=route.model,
                purpose=f"research:{choice.operation}",
                prompt=prompt,
                max_output_tokens=route.max_output_tokens,
                estimated_max_cost_usd=estimated_max_cost,
                response_model=response_model,
                effort=route.effort,
                validate_response=(lambda text: _validate_step_report(
                    _parse_execution_report(text, response_model), model_context, choice,
                )) if choice.operation == "attack" else None,
                invocation_budget=invocation_budget,
                context_scope=model_context.context_scope,
            )
            calls_made += 1
            report = _parse_execution_report(result.text, response_model)
            _validate_step_report(report, model_context, choice)
            persisted = _persist_step(
                iteration_id=iteration_id,
                workstream_id=workstream_id,
                provider_name=route.provider,
                model=route.model,
                choice=choice,
                context=full_context,
                report=report,
            )
            reactivate_bypassed_obligations(workstream_id)
            full_context_after = for_workstream(workstream_id)
            progress = build_progress_record(
                context_before=full_context, context_after=full_context_after,
                workstream_id=workstream_id, primary_id=int(primary["id"]),
                choice=choice, report=report, persisted=persisted,
            )
            _complete_iteration(iteration_id, workstream_id, choice, report, progress)
        except Exception as exc:
            _record_iteration_error(iteration_id, workstream_id, exc)
            raise

        artifact_ids.extend(persisted.artifact_ids)
        if progress.material_progress:
            current_run_consecutive_no_progress = 0
        else:
            current_run_consecutive_no_progress += 1
        if report.human_judgment_required:
            _finalize(
                workstream_id=workstream_id,
                iteration_id=iteration_id,
                status="blocked",
                stop_reason="human_judgment_required",
                detail=report.human_judgment_reason or "Scientific judgment is required.",
            )
            return ResearchOutcome(
                workstream_id,
                calls_made,
                tuple(iteration_ids),
                tuple(artifact_ids),
                "human_judgment_required",
                "blocked",
                strategy_calls_made,
                ideation_calls_made,
            )
        if (
            choice.operation == "attack"
            and report.attack_outcome == "no_critical_issue"
            and choice.focus_obligation_id is not None
            and _all_obligations_have_completed_candidates(
                full_context_after,
                _history(workstream_id),
                workstream_id,
                int(primary["id"]),
            )
            and not _open_obligation_ids(
                full_context_after, workstream_id, int(primary["id"])
            )
        ):
            _finalize(
                workstream_id=workstream_id,
                iteration_id=iteration_id,
                status="completed",
                stop_reason="candidate_survived_attack",
                detail=(
                    "Every graph-recorded proof obligation has an explicitly linked candidate "
                    "that survived its completed bounded attack without an unresolved issue on "
                    "that same candidate; this is not proof verification."
                ),
            )
            return ResearchOutcome(
                workstream_id,
                calls_made,
                tuple(iteration_ids),
                tuple(artifact_ids),
                "candidate_survived_attack",
                "completed",
                strategy_calls_made,
                ideation_calls_made,
            )
        if _all_branches_terminal(
            full_context_after, workstream_id
        ):
            _finalize(
                workstream_id=workstream_id,
                iteration_id=iteration_id,
                status="blocked",
                stop_reason="all_branches_blocked_or_refuted",
                detail="Every established construction route is inactive; no active obligation, candidate, or bypass work remains after this iteration.",
            )
            return ResearchOutcome(
                workstream_id,
                calls_made,
                tuple(iteration_ids),
                tuple(artifact_ids),
                "all_branches_blocked_or_refuted",
                "blocked",
                strategy_calls_made,
                ideation_calls_made,
            )
        if current_run_consecutive_no_progress >= 2:
            _finalize(
                workstream_id=workstream_id,
                iteration_id=iteration_id,
                status="blocked",
                stop_reason="stagnation",
                detail="Two consecutive iterations produced only duplicate or empty output.",
            )
            return ResearchOutcome(
                workstream_id,
                calls_made,
                tuple(iteration_ids),
                tuple(artifact_ids),
                "stagnation",
                "blocked",
                strategy_calls_made,
                ideation_calls_made,
            )

    final_context = for_workstream(workstream_id)
    remaining_obligations = _open_obligation_ids(
        final_context, workstream_id, int(primary["id"])
    )
    detail = f"The configured limit of {max_calls} execution call(s) was reached."
    if remaining_obligations:
        count = len(remaining_obligations)
        noun = "proof obligation" if count == 1 else "proof obligations"
        verb = "remains" if count == 1 else "remain"
        detail += f" {count} {noun} {verb} open."
    _finalize(
        workstream_id=workstream_id,
        iteration_id=last_iteration_id,
        status="completed",
        stop_reason="max_calls_exhausted",
        detail=detail,
    )
    return ResearchOutcome(
        workstream_id,
        calls_made,
        tuple(iteration_ids),
        tuple(artifact_ids),
        "max_calls_exhausted",
        "completed",
        strategy_calls_made,
        ideation_calls_made,
    )
