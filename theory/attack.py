from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import Config
from .db import connect
from .errors import ModelOutputError, TheoryError
from .graph import (
    add_entity,
    add_review,
    link_workstream_entity,
    set_attribute,
    set_workstream_status,
)
from .jsonutil import parse_json_model
from .model_calls import budget_guard, call_model
from .providers import get_model_spec, get_provider
from .research_context import ResearchContext, for_workstream


ATTACK_MAX_OUTPUT_TOKENS = 8_000
PRIMARY_TARGET_TYPES = {"Conjecture", "ResearchIdea", "OpenQuestion", "Theorem"}
CONCRETE_CANDIDATE_TYPES = {
    "counterexample",
    "obstruction",
    "hidden_assumption",
    "boundary_case",
    "conflict",
    "ambiguity",
    "failed_strategy",
}
ARTIFACT_TYPES = {
    "counterexample": "Counterexample",
    "obstruction": "Obstruction",
    "hidden_assumption": "Finding",
    "boundary_case": "Finding",
    "conflict": "Finding",
    "ambiguity": "Finding",
    "failed_strategy": "FailedApproach",
    "open_question": "OpenQuestion",
}


class AttackCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_type: Literal[
        "counterexample",
        "obstruction",
        "hidden_assumption",
        "boundary_case",
        "conflict",
        "ambiguity",
        "failed_strategy",
        "open_question",
    ]
    statement: str = Field(min_length=1, max_length=2_000)
    epistemic_status: Literal["inference", "speculation", "unresolved"]
    related_entity_ids: list[int] = Field(min_length=1, max_length=20)
    source_ids: list[int] = Field(default_factory=list, max_length=20)
    reasoning_summary: str = Field(min_length=1, max_length=4_000)

    @field_validator("statement", "reasoning_summary")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text cannot be blank")
        return stripped

    @field_validator("related_entity_ids", "source_ids")
    @classmethod
    def positive_unique_ids(cls, values: list[int]) -> list[int]:
        if any(value <= 0 for value in values):
            raise ValueError("IDs must be positive")
        if len(values) != len(set(values)):
            raise ValueError("IDs must not be duplicated")
        return values


class AttackReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target_entity_id: int = Field(gt=0)
    summary: str = Field(min_length=1, max_length=4_000)
    candidates: list[AttackCandidate] = Field(default_factory=list, max_length=12)
    could_not_determine: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("summary")
    @classmethod
    def strip_summary(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("summary cannot be blank")
        return stripped

    @field_validator("could_not_determine")
    @classmethod
    def clean_unresolved(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("could_not_determine entries cannot be blank")
        return cleaned


@dataclass(frozen=True)
class AttackOutcome:
    workstream_id: int
    target_entity_id: int
    artifact_ids: tuple[int, ...]
    review_id: int
    review_result: str


def _primary_target(context: ResearchContext, workstream_id: int) -> dict:
    input_ids = {
        int(link["entity_id"])
        for link in context.workstream_links
        if int(link["workstream_id"]) == workstream_id and link["role"] == "input"
    }
    if not input_ids:
        raise TheoryError(f"Attack workstream #{workstream_id} has no input entity.")
    candidates = [
        entity
        for entity in context.entities
        if int(entity["id"]) in input_ids and entity["entity_type"] in PRIMARY_TARGET_TYPES
    ]
    if not candidates:
        allowed = ", ".join(sorted(PRIMARY_TARGET_TYPES))
        raise TheoryError(
            f"Attack workstream #{workstream_id} has no eligible primary target; "
            f"expected one input of type: {allowed}."
        )
    if len(candidates) > 1:
        ids = ", ".join(f"#{entity['id']}" for entity in candidates)
        raise TheoryError(
            f"Attack workstream #{workstream_id} has ambiguous primary targets: {ids}. "
            "Keep exactly one theorem-like input target."
        )
    return candidates[0]


def _attack_prompt(context: ResearchContext, target: dict) -> str:
    payload = context.as_model_payload()
    payload["primary_target"] = target
    return f'''You are performing one adversarial theoretical-research attack.

Your job is not to encourage the idea and not to produce a proof. Search aggressively for
reasons the primary target may be false, trivial, subsumed by an existing result,
inconsistent with its assumptions, blocked by a represented obstruction, true only under a
narrower model, cosmetically improved, or underspecified. Look for counterexamples, minimal
failing cases, boundary regimes, hidden assumptions, conflicting graph results, lower-bound
obstacles, proof-strategy failure points, and missing definitions. State what you could not
determine.

EPISTEMIC RULES
- Sourced means source-backed, not mathematically verified.
- Inferred objects are provisional conclusions.
- Speculative objects are hypotheses.
- Contradicted objects are counterevidence or history.
- NEVER use quarantined objects as facts or assumptions.
- Retired relations are absent and must not be reconstructed.
- Do not claim novelty.
- Do not claim the target is false without a concrete counterargument/counterexample or
  relevant source-backed graph evidence.
- Separate graph-supported observations, inference, speculation, and unresolved questions.

Use only entity IDs and source IDs present in GRAPH CONTEXT. Every candidate must include the
primary target ID in related_entity_ids. A source ID is a citation to existing context, not
permission to label new output sourced. Newly generated output may use only inference,
speculation, or unresolved as epistemic_status.

GRAPH CONTEXT
{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}

Return ONLY strict JSON with exactly this shape:
{{
  "target_entity_id": {target['id']},
  "summary": "short technical summary",
  "candidates": [
    {{
      "candidate_type": "counterexample|obstruction|hidden_assumption|boundary_case|conflict|ambiguity|failed_strategy|open_question",
      "statement": "precise candidate problem or question",
      "epistemic_status": "inference|speculation|unresolved",
      "related_entity_ids": [{target['id']}],
      "source_ids": [],
      "reasoning_summary": "concise technical reasoning"
    }}
  ],
  "could_not_determine": ["specific unresolved point"]
}}'''


def _validate_report_references(
    report: AttackReport, context: ResearchContext, target_entity_id: int
) -> None:
    if report.target_entity_id != target_entity_id:
        raise ModelOutputError(
            f"Attack output targeted entity #{report.target_entity_id}, expected "
            f"#{target_entity_id}."
        )
    allowed_entity_ids = {int(entity["id"]) for entity in context.entities}
    allowed_source_ids = {int(source["id"]) for source in context.sources}
    for index, candidate in enumerate(report.candidates, start=1):
        unknown_entities = set(candidate.related_entity_ids) - allowed_entity_ids
        if unknown_entities:
            raise ModelOutputError(
                f"Attack candidate {index} referenced unknown/out-of-context entity IDs: "
                + ", ".join(str(value) for value in sorted(unknown_entities))
            )
        if target_entity_id not in candidate.related_entity_ids:
            raise ModelOutputError(
                f"Attack candidate {index} did not reference primary target "
                f"#{target_entity_id}."
            )
        unknown_sources = set(candidate.source_ids) - allowed_source_ids
        if unknown_sources:
            raise ModelOutputError(
                f"Attack candidate {index} referenced unknown/out-of-context source IDs: "
                + ", ".join(str(value) for value in sorted(unknown_sources))
            )


def _review_result(report: AttackReport) -> str:
    if any(
        candidate.candidate_type in CONCRETE_CANDIDATE_TYPES
        for candidate in report.candidates
    ):
        return "issue_found"
    if report.candidates or report.could_not_determine:
        return "inconclusive"
    return "no_flaw_found"


def _persist_attack(
    *,
    workstream_id: int,
    target_entity_id: int,
    provider_name: str,
    model: str,
    report: AttackReport,
) -> AttackOutcome:
    review_result = _review_result(report)
    artifact_ids: list[int] = []
    with connect() as con:
        for candidate in report.candidates:
            entity_type = ARTIFACT_TYPES[candidate.candidate_type]
            title_prefix = candidate.candidate_type.replace("_", " ").title()
            title = f"{title_prefix}: {candidate.statement}"[:240]
            body = (
                f"{candidate.statement}\n\nReasoning: {candidate.reasoning_summary}\n\n"
                f"Model epistemic status: {candidate.epistemic_status}\n"
                f"Related entity IDs: {candidate.related_entity_ids}\n"
                f"Cited source IDs: {candidate.source_ids}"
            )
            entity_id = add_entity(
                con,
                entity_type,
                title,
                body=body,
                trust_state="quarantined",
                source_ids=candidate.source_ids,
                generated_by_llm=True,
            )
            set_attribute(con, entity_id, "attack_candidate_type", candidate.candidate_type)
            set_attribute(
                con, entity_id, "model_epistemic_status", candidate.epistemic_status
            )
            set_attribute(
                con,
                entity_id,
                "related_entity_ids",
                json.dumps(candidate.related_entity_ids),
            )
            link_workstream_entity(con, workstream_id, entity_id, "created")
            artifact_ids.append(entity_id)

        unresolved = (
            "; ".join(report.could_not_determine)
            if report.could_not_determine
            else "none reported"
        )
        issues = (
            f"{report.summary}\nCreated quarantined candidate entities: {artifact_ids or 'none'}. "
            f"Could not determine: {unresolved}"
        )
        review_id = add_review(
            con,
            "counterexample_attempt",
            review_result,
            workstream_id=workstream_id,
            issues=issues,
            provider=provider_name,
            model=model,
        )
        summary = (
            f"Attack completed. Review result: {review_result}. "
            f"Created {len(artifact_ids)} quarantined candidate artifact(s). "
            f"{report.summary} Could not determine: {unresolved}"
        )
        set_workstream_status(con, workstream_id, "completed", summary=summary)

    return AttackOutcome(
        workstream_id=workstream_id,
        target_entity_id=target_entity_id,
        artifact_ids=tuple(artifact_ids),
        review_id=review_id,
        review_result=review_result,
    )


def _mark_workstream_error(workstream_id: int, error: Exception) -> None:
    with connect() as con:
        set_workstream_status(
            con,
            workstream_id,
            "error",
            summary=f"Attack execution error: {type(error).__name__}: {error}"[:4000],
        )


def attack(workstream_id: int, provider_name: str) -> AttackOutcome:
    """Run exactly one provider call over one selected attack-workstream context."""
    cfg = Config.load()
    models = {"openai": cfg.openai_model, "anthropic": cfg.anthropic_model}
    if provider_name not in models:
        raise TheoryError(f"Unknown provider: {provider_name}")
    model = models[provider_name]
    get_model_spec(model, provider_name)

    context = for_workstream(workstream_id)
    workstream = context.workstream
    if workstream is None:
        raise TheoryError(f"Workstream #{workstream_id} does not exist.")
    if workstream["workstream_type"] != "attack":
        raise TheoryError(f"Workstream #{workstream_id} is not an attack workstream.")
    if workstream["status"] != "active":
        raise TheoryError(
            f"Attack workstream #{workstream_id} is {workstream['status']}, not active."
        )
    target = _primary_target(context, workstream_id)
    target_entity_id = int(target["id"])
    prompt = _attack_prompt(context, target)
    estimated_max_cost = budget_guard(
        cfg,
        model=model,
        prompt=prompt,
        max_output_tokens=ATTACK_MAX_OUTPUT_TOKENS,
        purpose="attack",
    )

    try:
        provider = get_provider(provider_name)
        result = call_model(
            run_id=None,
            workstream_id=workstream_id,
            provider=provider,
            provider_name=provider_name,
            model=model,
            purpose="attack",
            prompt=prompt,
            max_output_tokens=ATTACK_MAX_OUTPUT_TOKENS,
            estimated_max_cost_usd=estimated_max_cost,
        )
        report = parse_json_model(result.text, AttackReport)
        _validate_report_references(report, context, target_entity_id)
        return _persist_attack(
            workstream_id=workstream_id,
            target_entity_id=target_entity_id,
            provider_name=provider_name,
            model=model,
            report=report,
        )
    except Exception as exc:
        _mark_workstream_error(workstream_id, exc)
        raise
