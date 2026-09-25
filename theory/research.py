from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Config
from .db import connect, utcnow
from .errors import ModelOutputError, TheoryError
from .graph import (
    add_entity,
    add_relation,
    add_review,
    link_workstream_entity,
    set_attribute,
    set_workstream_status,
)
from .jsonutil import parse_json_model
from .model_calls import budget_guard, call_model
from .providers import get_model_spec, get_provider
from .research_context import ResearchContext, for_workstream


RESEARCH_MAX_OUTPUT_TOKENS = 12_000
MAX_CONTROLLER_CALLS = 20
OPERATIONS = ("develop", "attack", "synthesize", "prove")
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


class ResearchArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    artifact_type: Literal[
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
    statement: str = Field(min_length=1, max_length=2_000)
    reasoning_summary: str = Field(min_length=1, max_length=5_000)
    material_key: str = Field(pattern=r"^[a-z0-9][a-z0-9_]{2,79}$")
    epistemic_status: Literal["inference", "speculation", "unresolved"]
    related_entity_ids: list[int] = Field(min_length=1, max_length=24)
    source_ids: list[int] = Field(default_factory=list, max_length=20)
    branch_status: Literal[
        "promising", "blocked", "failed", "refuted", "unresolved"
    ] | None

    @field_validator("statement", "reasoning_summary")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        return _strip_nonempty(value)

    @field_validator("related_entity_ids", "source_ids")
    @classmethod
    def positive_unique_ids(cls, values: list[int]) -> list[int]:
        return _positive_unique_ids(values)

    @model_validator(mode="after")
    def validate_branch_status(self) -> "ResearchArtifact":
        if self.branch_status in {"failed", "refuted"} and self.artifact_type != "failed_approach":
            raise ValueError("failed/refuted branches must be failed_approach artifacts")
        if self.branch_status == "blocked" and self.artifact_type not in {
            "obstruction",
            "failed_approach",
        }:
            raise ValueError("blocked branches must be obstruction or failed_approach artifacts")
        return self


class ResearchStepReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    operation: Literal["develop", "attack", "synthesize", "prove"]
    target_entity_id: int = Field(gt=0)
    summary: str = Field(min_length=1, max_length=4_000)
    artifacts: list[ResearchArtifact] = Field(default_factory=list, max_length=10)
    consumed_entity_ids: list[int] = Field(default_factory=list, max_length=8)
    addressed_obligation_ids: list[int] = Field(default_factory=list, max_length=12)
    attack_outcome: Literal[
        "not_applicable", "critical_issue", "no_critical_issue", "inconclusive"
    ]
    could_not_determine: list[str] = Field(default_factory=list, max_length=12)
    human_judgment_required: bool
    human_judgment_reason: str | None

    @field_validator("summary")
    @classmethod
    def strip_summary(cls, value: str) -> str:
        return _strip_nonempty(value)

    @field_validator("consumed_entity_ids", "addressed_obligation_ids")
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


@dataclass(frozen=True)
class OperationChoice:
    operation: str
    target_entity_id: int
    rationale: str
    consumed_entity_ids: tuple[int, ...] = ()
    open_obligation_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PersistedStep:
    artifact_ids: tuple[int, ...]
    duplicate_count: int
    material_progress: bool


@dataclass(frozen=True)
class ResearchOutcome:
    workstream_id: int
    calls_made: int
    iteration_ids: tuple[int, ...]
    artifact_ids: tuple[int, ...]
    stop_reason: str
    final_status: str


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


def _completed_for_target(history: tuple[dict, ...], operation: str, entity_id: int) -> bool:
    return any(
        row["status"] == "completed"
        and row["operation"] == operation
        and int(row["target_entity_id"]) == entity_id
        for row in history
    )


def _attribute(context: ResearchContext, entity_id: int, key: str) -> str | None:
    return context.attributes.get(entity_id, {}).get(key)


def _is_precise_candidate(context: ResearchContext, entity: dict) -> bool:
    entity_id = int(entity["id"])
    if entity["status"] != "active" or entity["trust_state"] == "contradicted":
        return False
    if entity["entity_type"] in PRECISE_ENTITY_TYPES:
        return True
    if entity["entity_type"] not in {"Conjecture", "Technique"}:
        return False
    if (_attribute(context, entity_id, "precise_candidate") or "").casefold() == "true":
        return True
    return _attribute(context, entity_id, "develop_item_type") == "protocol_component"


def _obligation_ids(
    context: ResearchContext, workstream_id: int, primary_entity_id: int
) -> tuple[int, ...]:
    linked = _linked_ids(context, workstream_id)
    blocked_by = {
        int(link["entity_id"])
        for link in context.workstream_links
        if int(link["workstream_id"]) == workstream_id
        and link["role"] == "blocked_by"
    }
    obligations: list[int] = []
    for entity in context.entities:
        entity_id = int(entity["id"])
        if (
            entity_id not in linked
            or entity_id == primary_entity_id
            or entity["status"] != "active"
        ):
            continue
        attrs = context.attributes.get(entity_id, {})
        is_open_obligation = entity["entity_type"] == "OpenQuestion" and (
            attrs.get("research_artifact_type") == "proof_obligation"
            or attrs.get("develop_item_type") == "proof_obligation"
            or attrs.get("is_proof_obligation", "").casefold() == "true"
            or entity_id in blocked_by
        )
        is_blocked_obligation = entity["entity_type"] == "Obstruction" and (
            entity_id in blocked_by
            or attrs.get("is_proof_obligation", "").casefold() == "true"
            or attrs.get("research_branch_status") == "blocked"
            or attrs.get("develop_branch_status") == "blocked"
        )
        if is_open_obligation or is_blocked_obligation:
            obligations.append(entity_id)
    return tuple(sorted(obligations))


def _addressed_obligation_ids(context: ResearchContext) -> set[int]:
    addressed = {
        int(relation["target_entity_id"])
        for relation in context.relations
        if relation["relation_type"] == "ATTEMPTS"
    }
    for entity_id, attrs in context.attributes.items():
        raw = attrs.get("addresses_obligation_ids")
        if raw is None:
            continue
        try:
            addressed.update(int(value) for value in json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return addressed


def _branch_states(
    context: ResearchContext, workstream_id: int
) -> tuple[str, ...]:
    states: list[str] = []
    for entity_id in _linked_ids(context, workstream_id):
        attrs = context.attributes.get(entity_id, {})
        state = attrs.get("research_branch_status") or attrs.get("develop_branch_status")
        if state in LIVE_BRANCH_STATES | TERMINAL_BRANCH_STATES:
            states.append(state)
    return tuple(states)


def _all_branches_terminal(context: ResearchContext, workstream_id: int) -> bool:
    states = _branch_states(context, workstream_id)
    return bool(states) and not any(state in LIVE_BRANCH_STATES for state in states)


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
        and entity["entity_type"] in SYNTHESIS_INPUT_TYPES
    ]
    candidates.sort(
        key=lambda entity: (
            entity["entity_type"] not in {"Lemma", "Theorem", "ProofAttempt", "Technique"},
            -int(entity["id"]),
        )
    )
    return tuple(int(entity["id"]) for entity in candidates[:4])


def choose_next_operation(
    context: ResearchContext,
    workstream_id: int,
    primary: dict,
    history: tuple[dict, ...],
) -> OperationChoice:
    """Choose the next bounded operation deterministically from current graph state."""
    primary_id = int(primary["id"])
    linked = _linked_ids(context, workstream_id)
    entity_by_id = {int(entity["id"]): entity for entity in context.entities}
    obligations = _obligation_ids(context, workstream_id, primary_id)
    addressed = _addressed_obligation_ids(context)
    open_obligations = tuple(entity_id for entity_id in obligations if entity_id not in addressed)
    precise = [
        entity_by_id[entity_id]
        for entity_id in linked
        if entity_id in entity_by_id and _is_precise_candidate(context, entity_by_id[entity_id])
    ]
    precise.sort(
        key=lambda entity: (
            {"ProofAttempt": 0, "Lemma": 1, "Theorem": 2, "Technique": 3}.get(
                entity["entity_type"], 4
            ),
            -int(entity["id"]),
        )
    )

    if open_obligations:
        obligation_id = open_obligations[0]
        consumed = _relevant_synthesis_inputs(context, workstream_id, obligation_id)
        if len(consumed) >= 2 and not _completed_for_target(
            history, "synthesize", obligation_id
        ):
            return OperationChoice(
                operation="synthesize",
                target_entity_id=obligation_id,
                consumed_entity_ids=consumed,
                open_obligation_ids=open_obligations,
                rationale=(
                    f"Open proof obligation #{obligation_id} has {len(consumed)} linked, "
                    "non-terminal artifacts that can be combined in one bounded synthesis."
                ),
            )
        if precise:
            candidate_id = int(precise[0]["id"])
            return OperationChoice(
                operation="prove",
                target_entity_id=candidate_id,
                open_obligation_ids=open_obligations,
                rationale=(
                    f"Precise candidate #{candidate_id} exists while proof obligation "
                    f"#{obligation_id} remains open; a rigorous proof attempt is the next "
                    "material step."
                ),
            )
        return OperationChoice(
            operation="develop",
            target_entity_id=primary_id,
            open_obligation_ids=open_obligations,
            rationale=(
                f"Proof obligation #{obligation_id} is open, but the graph lacks either two "
                "usable synthesis inputs or a precise candidate statement."
            ),
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

    unproved = [
        entity
        for entity in precise
        if entity["entity_type"] != "ProofAttempt"
        and not _completed_for_target(history, "prove", int(entity["id"]))
        and not (
            int(entity["id"]) == primary_id
            and primary["entity_type"] in PRECISE_ENTITY_TYPES
        )
    ]
    if unproved:
        target_id = int(unproved[0]["id"])
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


def _operation_instructions(choice: OperationChoice) -> str:
    if choice.operation == "develop":
        return (
            "Derive a substantive consequence, lemma, protocol component, parameter analysis, "
            "or proof obligation. Explore a genuinely new branch. Preserve a failed branch as "
            "failed_approach or obstruction rather than hiding it."
        )
    if choice.operation == "attack":
        return (
            "Attack this concrete candidate for counterexamples, invalid steps, boundary cases, "
            "or hidden assumptions. Set attack_outcome precisely. An empty no_critical_issue "
            "result means only that this pass found no critical issue, never verification."
        )
    if choice.operation == "synthesize":
        consumed = ", ".join(f"#{value}" for value in choice.consumed_entity_ids)
        return (
            f"Consume all of these existing candidate artifacts: {consumed}. Combine them to "
            f"attempt to close the specific obligation #{choice.target_entity_id}. If the "
            "combination fails, emit a failed_approach or obstruction with the exact failure."
        )
    return (
        "Turn this precise candidate statement into a rigorous stepwise proof attempt. Expose "
        "every new obligation. Any addressed obligation must appear in the proof artifact's "
        "related_entity_ids. If the proof fails, persist the exact failed approach or blocker."
    )


def _research_prompt(
    context: ResearchContext, primary: dict, choice: OperationChoice
) -> str:
    payload = context.as_model_payload()
    decision = {
        "operation": choice.operation,
        "target_entity_id": choice.target_entity_id,
        "primary_entity_id": int(primary["id"]),
        "rationale": choice.rationale,
        "required_consumed_entity_ids": choice.consumed_entity_ids,
        "currently_open_obligation_ids": choice.open_obligation_ids,
    }
    return f'''You are executing one bounded operation in a human-directed theoretical-research workbench.

Execute only the selected operation. Do not choose another operation and do not make a second
pass. {_operation_instructions(choice)}

EPISTEMIC AND WRITE RULES
- Use only the linked GRAPH CONTEXT below; do not use chat history, retrieval, or outside facts.
- Sourced means source-backed, not mathematically verified.
- Inferred objects are provisional; speculative objects are hypotheses.
- Contradicted objects are counterevidence/history.
- NEVER use quarantined objects as facts. You may analyze them only as candidate artifacts.
- Do not claim novelty, correctness, verification, or a completed proof.
- New output may use only inference, speculation, or unresolved as epistemic_status.
- Every artifact needs a stable lowercase material_key naming its mathematical content.
- Do not restate an existing entity or existing material_key. Rephrasing is not progress.
- Every artifact must reference the selected target in related_entity_ids.
- addressed_obligation_ids means a concrete candidate argument was produced; it does not mean
  the obligation was human-verified or mathematically resolved.
- Set human_judgment_required only when a scientific choice cannot responsibly be made from
  this graph state.

CONTROLLER DECISION
{json.dumps(decision, indent=2, sort_keys=True)}

GRAPH CONTEXT
{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}

Return ONLY strict JSON with exactly this shape:
{{
  "operation": "{choice.operation}",
  "target_entity_id": {choice.target_entity_id},
  "summary": "technical result of this one bounded operation",
  "artifacts": [
    {{
      "artifact_type": "consequence|lemma|protocol_component|parameter_analysis|proof_obligation|open_question|proof_attempt|synthesis|counterexample|obstruction|failed_approach|finding",
      "statement": "precise substantive research object",
      "reasoning_summary": "derivation, argument, calculation, or exact failure point",
      "material_key": "stable_lowercase_concept_key",
      "epistemic_status": "inference|speculation|unresolved",
      "related_entity_ids": [{choice.target_entity_id}],
      "source_ids": [],
      "branch_status": null
    }}
  ],
  "consumed_entity_ids": {json.dumps(list(choice.consumed_entity_ids))},
  "addressed_obligation_ids": [],
  "attack_outcome": "not_applicable|critical_issue|no_critical_issue|inconclusive",
  "could_not_determine": [],
  "human_judgment_required": false,
  "human_judgment_reason": null
}}'''


def _validate_step_report(
    report: ResearchStepReport,
    context: ResearchContext,
    choice: OperationChoice,
) -> None:
    if report.operation != choice.operation:
        raise ModelOutputError(
            f"Research output chose {report.operation!r}, expected {choice.operation!r}."
        )
    if report.target_entity_id != choice.target_entity_id:
        raise ModelOutputError(
            f"Research output targeted entity #{report.target_entity_id}, expected "
            f"#{choice.target_entity_id}."
        )
    allowed_entity_ids = {int(entity["id"]) for entity in context.entities}
    allowed_source_ids = {int(source["id"]) for source in context.sources}
    obligation_ids = set(choice.open_obligation_ids)
    unknown_addressed = set(report.addressed_obligation_ids) - obligation_ids
    if unknown_addressed:
        raise ModelOutputError(
            "Research output addressed unknown or non-open obligation IDs: "
            + ", ".join(str(value) for value in sorted(unknown_addressed))
        )
    if tuple(report.consumed_entity_ids) != choice.consumed_entity_ids:
        raise ModelOutputError(
            "Research output did not echo the controller's exact consumed_entity_ids."
        )
    for index, artifact in enumerate(report.artifacts, start=1):
        unknown_entities = set(artifact.related_entity_ids) - allowed_entity_ids
        if unknown_entities:
            raise ModelOutputError(
                f"Research artifact {index} referenced unknown/out-of-context entity IDs: "
                + ", ".join(str(value) for value in sorted(unknown_entities))
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
        required_refs = set(choice.consumed_entity_ids) | {choice.target_entity_id}
        if not report.artifacts:
            raise ModelOutputError("Synthesize must persist an attempted result or failure.")
        for index, artifact in enumerate(report.artifacts, start=1):
            if not required_refs <= set(artifact.related_entity_ids):
                raise ModelOutputError(
                    f"Synthesis artifact {index} did not reference every consumed artifact "
                    "and the target obligation."
                )
        if choice.target_entity_id not in report.addressed_obligation_ids and not any(
            artifact.artifact_type in {"failed_approach", "obstruction"}
            for artifact in report.artifacts
        ):
            raise ModelOutputError(
                "An unsuccessful synthesis must persist a failed approach or obstruction."
            )

    if choice.operation == "prove" and not report.artifacts:
        raise ModelOutputError("Prove must persist a proof attempt, obligation, or failure.")
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
    duplicate_count = 0
    with connect() as con:
        for artifact in report.artifacts:
            tokens = _normalized_tokens(artifact.statement)
            if artifact.material_key in existing_keys or _is_lexical_duplicate(
                tokens, fingerprints
            ):
                duplicate_count += 1
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
            set_attribute(con, entity_id, "research_artifact_type", artifact.artifact_type)
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
            if artifact.branch_status is not None:
                set_attribute(
                    con, entity_id, "research_branch_status", artifact.branch_status
                )
            if artifact.artifact_type == "proof_obligation":
                set_attribute(con, entity_id, "is_proof_obligation", "true")
            if artifact.artifact_type in {"lemma", "protocol_component", "proof_attempt"}:
                set_attribute(con, entity_id, "precise_candidate", "true")
            link_workstream_entity(con, workstream_id, entity_id, "created")
            artifact_ids.append(entity_id)
            accepted.append((artifact, entity_id))
            existing_keys.add(artifact.material_key)
            fingerprints.append(tokens)

        for obligation_id in report.addressed_obligation_ids:
            supporting_ids = [
                entity_id
                for artifact, entity_id in accepted
                if artifact.artifact_type in PROOF_ARTIFACT_TYPES
                and obligation_id in artifact.related_entity_ids
            ]
            for entity_id in supporting_ids:
                add_relation(
                    con,
                    entity_id,
                    "ATTEMPTS",
                    obligation_id,
                    trust_state="quarantined",
                    generated_by_llm=True,
                )
                set_attribute(
                    con,
                    entity_id,
                    "addresses_obligation_ids",
                    json.dumps(report.addressed_obligation_ids),
                )

        if choice.operation == "attack":
            review_result = {
                "critical_issue": "issue_found",
                "no_critical_issue": "no_flaw_found",
                "inconclusive": "inconclusive",
            }[report.attack_outcome]
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

        material_progress = bool(artifact_ids)
        con.execute(
            """
            UPDATE research_iterations
            SET status='completed',material_progress=?,artifact_ids_json=?,duplicate_count=?,
                attack_outcome=?,completed_at=?
            WHERE id=?
            """,
            (
                int(material_progress),
                json.dumps(artifact_ids),
                duplicate_count,
                report.attack_outcome,
                utcnow(),
                iteration_id,
            ),
        )
        set_workstream_status(
            con,
            workstream_id,
            "active",
            summary=(
                f"Research iteration completed: {choice.operation}. "
                f"Created {len(artifact_ids)} quarantined artifact(s); "
                f"rejected {duplicate_count} duplicate(s). {report.summary}"
            ),
        )

    return PersistedStep(
        artifact_ids=tuple(artifact_ids),
        duplicate_count=duplicate_count,
        material_progress=material_progress,
    )


def _start_iteration(workstream_id: int, choice: OperationChoice) -> int:
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
                rationale,status,created_at
            ) VALUES(1,?,?,?,?,?,'running',?)
            """,
            (
                workstream_id,
                number,
                choice.operation,
                choice.target_entity_id,
                choice.rationale,
                utcnow(),
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


def _consecutive_no_progress(workstream_id: int) -> int:
    with connect() as con:
        rows = con.execute(
            """
            SELECT material_progress FROM research_iterations
            WHERE workstream_id=? AND status='completed'
            ORDER BY iteration_number DESC LIMIT 2
            """,
            (workstream_id,),
        ).fetchall()
    count = 0
    for row in rows:
        if row["material_progress"]:
            break
        count += 1
    return count


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


def research(
    workstream_id: int, provider_name: str, *, max_calls: int
) -> ResearchOutcome:
    """Run a bounded, deterministic controller with one model call per iteration."""
    if not 1 <= max_calls <= MAX_CONTROLLER_CALLS:
        raise TheoryError(
            f"max_calls must be between 1 and {MAX_CONTROLLER_CALLS}."
        )
    cfg = Config.load()
    models = {"openai": cfg.openai_model, "anthropic": cfg.anthropic_model}
    if provider_name not in models:
        raise TheoryError(f"Unknown provider: {provider_name}")
    model = models[provider_name]
    get_model_spec(model, provider_name)

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
    primary = _primary_target(initial_context, workstream_id)

    calls_made = 0
    iteration_ids: list[int] = []
    artifact_ids: list[int] = []
    last_iteration_id: int | None = None
    provider = None

    while calls_made < max_calls:
        context = for_workstream(workstream_id)
        history = _history(workstream_id)
        if _all_branches_terminal(context, workstream_id):
            _finalize(
                workstream_id=workstream_id,
                iteration_id=last_iteration_id,
                status="blocked",
                stop_reason="all_branches_blocked_or_refuted",
                detail="No promising or unresolved branch remains in the linked graph state.",
            )
            return ResearchOutcome(
                workstream_id,
                calls_made,
                tuple(iteration_ids),
                tuple(artifact_ids),
                "all_branches_blocked_or_refuted",
                "blocked",
            )
        if _consecutive_no_progress(workstream_id) >= 2:
            _finalize(
                workstream_id=workstream_id,
                iteration_id=last_iteration_id,
                status="blocked",
                stop_reason="stagnation",
                detail="Two consecutive completed iterations created no substantive new object.",
            )
            return ResearchOutcome(
                workstream_id,
                calls_made,
                tuple(iteration_ids),
                tuple(artifact_ids),
                "stagnation",
                "blocked",
            )

        choice = choose_next_operation(context, workstream_id, primary, history)
        prompt = _research_prompt(context, primary, choice)
        estimated_max_cost = budget_guard(
            cfg,
            model=model,
            prompt=prompt,
            max_output_tokens=RESEARCH_MAX_OUTPUT_TOKENS,
            purpose=f"research:{choice.operation}",
        )
        if provider is None:
            try:
                provider = get_provider(provider_name)
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
        iteration_id = _start_iteration(workstream_id, choice)
        last_iteration_id = iteration_id
        iteration_ids.append(iteration_id)
        try:
            result = call_model(
                run_id=None,
                workstream_id=workstream_id,
                provider=provider,
                provider_name=provider_name,
                model=model,
                purpose=f"research:{choice.operation}",
                prompt=prompt,
                max_output_tokens=RESEARCH_MAX_OUTPUT_TOKENS,
                estimated_max_cost_usd=estimated_max_cost,
            )
            calls_made += 1
            report = parse_json_model(result.text, ResearchStepReport)
            _validate_step_report(report, context, choice)
            persisted = _persist_step(
                iteration_id=iteration_id,
                workstream_id=workstream_id,
                provider_name=provider_name,
                model=model,
                choice=choice,
                context=context,
                report=report,
            )
        except Exception as exc:
            _record_iteration_error(iteration_id, workstream_id, exc)
            raise

        artifact_ids.extend(persisted.artifact_ids)
        context_after = for_workstream(workstream_id)
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
            )
        if (
            choice.operation == "attack"
            and report.attack_outcome == "no_critical_issue"
            and not (
                set(
                    _obligation_ids(
                        context_after, workstream_id, int(primary["id"])
                    )
                )
                - _addressed_obligation_ids(context_after)
            )
        ):
            _finalize(
                workstream_id=workstream_id,
                iteration_id=iteration_id,
                status="completed",
                stop_reason="candidate_survived_attack",
                detail=(
                    "All graph-recorded proof obligations were addressed and the subsequent "
                    "bounded attack found no critical issue; this is not proof verification."
                ),
            )
            return ResearchOutcome(
                workstream_id,
                calls_made,
                tuple(iteration_ids),
                tuple(artifact_ids),
                "candidate_survived_attack",
                "completed",
            )
        if _all_branches_terminal(context_after, workstream_id):
            _finalize(
                workstream_id=workstream_id,
                iteration_id=iteration_id,
                status="blocked",
                stop_reason="all_branches_blocked_or_refuted",
                detail="No promising or unresolved branch remains after this iteration.",
            )
            return ResearchOutcome(
                workstream_id,
                calls_made,
                tuple(iteration_ids),
                tuple(artifact_ids),
                "all_branches_blocked_or_refuted",
                "blocked",
            )
        if _consecutive_no_progress(workstream_id) >= 2:
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
            )

    _finalize(
        workstream_id=workstream_id,
        iteration_id=last_iteration_id,
        status="completed",
        stop_reason="max_calls_exhausted",
        detail=f"The configured limit of {max_calls} model call(s) was reached.",
    )
    return ResearchOutcome(
        workstream_id,
        calls_made,
        tuple(iteration_ids),
        tuple(artifact_ids),
        "max_calls_exhausted",
        "completed",
    )
