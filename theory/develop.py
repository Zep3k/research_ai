from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Config
from .db import connect
from .errors import ModelOutputError, TheoryError
from .graph import add_entity, link_workstream_entity, set_attribute, set_workstream_status
from .jsonutil import parse_json_model
from .model_calls import budget_guard, call_model
from .providers import get_model_spec, get_provider
from .research_context import ResearchContext, for_workstream


DEVELOP_MAX_OUTPUT_TOKENS = 16_000
PRIMARY_TARGET_TYPES = {
    "Conjecture",
    "Finding",
    "Lemma",
    "OpenQuestion",
    "ProofAttempt",
    "ResearchIdea",
    "Technique",
    "Theorem",
}
DEVELOPMENT_ARTIFACT_TYPES = {
    "consequence": "Finding",
    "intermediate_lemma": "Lemma",
    "protocol_component": "Technique",
    "parameter_analysis": "Finding",
    "proof_obligation": "OpenQuestion",
    "open_question": "OpenQuestion",
}
BRANCH_ARTIFACT_TYPES = {
    "promising": "Technique",
    "blocked": "Obstruction",
    "failed": "FailedApproach",
    "unresolved": "Technique",
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


class DevelopmentItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    item_type: Literal[
        "consequence",
        "intermediate_lemma",
        "protocol_component",
        "parameter_analysis",
        "proof_obligation",
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
        return _strip_nonempty(value)

    @field_validator("related_entity_ids", "source_ids")
    @classmethod
    def positive_unique_ids(cls, values: list[int]) -> list[int]:
        return _positive_unique_ids(values)


class DevelopmentBranch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=160)
    status: Literal["promising", "blocked", "failed", "unresolved"]
    objective: str = Field(min_length=1, max_length=2_000)
    technical_plan: str = Field(min_length=1, max_length=5_000)
    proof_obligations: list[str] = Field(default_factory=list, max_length=10)
    epistemic_status: Literal["inference", "speculation", "unresolved"]
    related_entity_ids: list[int] = Field(min_length=1, max_length=20)
    source_ids: list[int] = Field(default_factory=list, max_length=20)

    @field_validator("name", "objective", "technical_plan")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        return _strip_nonempty(value)

    @field_validator("proof_obligations")
    @classmethod
    def clean_proof_obligations(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("proof obligations cannot be blank")
        return cleaned

    @field_validator("related_entity_ids", "source_ids")
    @classmethod
    def positive_unique_ids(cls, values: list[int]) -> list[int]:
        return _positive_unique_ids(values)


class DevelopmentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target_entity_id: int = Field(gt=0)
    summary: str = Field(min_length=1, max_length=4_000)
    developments: list[DevelopmentItem] = Field(min_length=4, max_length=12)
    branches: list[DevelopmentBranch] = Field(min_length=2, max_length=4)
    could_not_determine: list[str] = Field(default_factory=list, max_length=12)

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

    @model_validator(mode="after")
    def require_technical_coverage(self) -> "DevelopmentReport":
        kinds = {item.item_type for item in self.developments}
        required = {"consequence", "parameter_analysis", "proof_obligation"}
        missing = required - kinds
        if missing:
            raise ValueError(
                "developments are missing required item types: "
                + ", ".join(sorted(missing))
            )
        if not kinds & {"intermediate_lemma", "protocol_component"}:
            raise ValueError(
                "developments require an intermediate_lemma or protocol_component"
            )
        branch_names = [branch.name.casefold() for branch in self.branches]
        if len(branch_names) != len(set(branch_names)):
            raise ValueError("development branch names must be distinct")
        return self


@dataclass(frozen=True)
class DevelopOutcome:
    workstream_id: int
    target_entity_id: int
    development_artifact_ids: tuple[int, ...]
    branch_artifact_ids: tuple[int, ...]

    @property
    def artifact_ids(self) -> tuple[int, ...]:
        return self.development_artifact_ids + self.branch_artifact_ids


def _primary_target(context: ResearchContext, workstream_id: int) -> dict:
    input_ids = {
        int(link["entity_id"])
        for link in context.workstream_links
        if int(link["workstream_id"]) == workstream_id and link["role"] == "input"
    }
    if not input_ids:
        raise TheoryError(f"Develop workstream #{workstream_id} has no input entity.")
    candidates = [
        entity
        for entity in context.entities
        if int(entity["id"]) in input_ids and entity["entity_type"] in PRIMARY_TARGET_TYPES
    ]
    if not candidates:
        allowed = ", ".join(sorted(PRIMARY_TARGET_TYPES))
        raise TheoryError(
            f"Develop workstream #{workstream_id} has no eligible primary target; "
            f"expected one input of type: {allowed}."
        )
    if len(candidates) > 1:
        ids = ", ".join(f"#{entity['id']}" for entity in candidates)
        raise TheoryError(
            f"Develop workstream #{workstream_id} has ambiguous primary targets: {ids}. "
            "Keep exactly one eligible input target."
        )
    return candidates[0]


def _develop_prompt(context: ResearchContext, target: dict) -> str:
    payload = context.as_model_payload()
    payload["primary_target"] = target
    return f'''You are performing one constructive theoretical-research development pass.

Push the primary target forward technically. Do not merely encourage or criticize it. Derive
consequences, propose useful intermediate lemmas or protocol/algorithm components, perform
explicit parameter or counting analysis, identify proof obligations, and explore 2 to 4
materially different branches. Preserve an informative branch even when it appears blocked or
failed. State what cannot be determined from the graph context.

EPISTEMIC RULES
- Sourced means source-backed, not mathematically verified.
- Inferred objects are provisional conclusions.
- Speculative objects are hypotheses.
- Contradicted objects are counterevidence or history.
- NEVER use quarantined objects as facts or assumptions.
- Retired relations are absent and must not be reconstructed.
- Do not claim novelty, proof, correctness, or verification.
- Do not silently strengthen definitions, assumptions, or models.
- Separate graph-supported observations, inference, speculation, and unresolved questions.

Use only entity IDs and source IDs present in GRAPH CONTEXT. Every development item and branch
must include the primary target ID in related_entity_ids. A source ID cites existing context; it
does not make new output sourced. Newly generated output may use only inference, speculation,
or unresolved as epistemic_status. Mark a branch failed only when you can state the exact
technical failure point; otherwise use blocked or unresolved.

GRAPH CONTEXT
{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}

Return 4 to 12 non-redundant development items. They must include a consequence, a parameter
analysis, a proof obligation, and at least one intermediate lemma or protocol component. Return
2 to 4 branches with distinct names and materially different technical plans.

Return ONLY strict JSON with exactly this shape:
{{
  "target_entity_id": {target['id']},
  "summary": "short technical account of progress and limits",
  "developments": [
    {{
      "item_type": "consequence|intermediate_lemma|protocol_component|parameter_analysis|proof_obligation|open_question",
      "statement": "precise technical statement",
      "epistemic_status": "inference|speculation|unresolved",
      "related_entity_ids": [{target['id']}],
      "source_ids": [],
      "reasoning_summary": "concise derivation, calculation, or rationale"
    }}
  ],
  "branches": [
    {{
      "name": "short distinct branch name",
      "status": "promising|blocked|failed|unresolved",
      "objective": "what this branch tries to establish or construct",
      "technical_plan": "concrete technical path, including the failure point if failed",
      "proof_obligations": ["specific obligation"],
      "epistemic_status": "inference|speculation|unresolved",
      "related_entity_ids": [{target['id']}],
      "source_ids": []
    }}
  ],
  "could_not_determine": ["specific unresolved point"]
}}'''


def _validate_references(
    *,
    label: str,
    index: int,
    related_entity_ids: list[int],
    source_ids: list[int],
    target_entity_id: int,
    allowed_entity_ids: set[int],
    allowed_source_ids: set[int],
) -> None:
    unknown_entities = set(related_entity_ids) - allowed_entity_ids
    if unknown_entities:
        raise ModelOutputError(
            f"Develop {label} {index} referenced unknown/out-of-context entity IDs: "
            + ", ".join(str(value) for value in sorted(unknown_entities))
        )
    if target_entity_id not in related_entity_ids:
        raise ModelOutputError(
            f"Develop {label} {index} did not reference primary target "
            f"#{target_entity_id}."
        )
    unknown_sources = set(source_ids) - allowed_source_ids
    if unknown_sources:
        raise ModelOutputError(
            f"Develop {label} {index} referenced unknown/out-of-context source IDs: "
            + ", ".join(str(value) for value in sorted(unknown_sources))
        )


def _validate_report_references(
    report: DevelopmentReport, context: ResearchContext, target_entity_id: int
) -> None:
    if report.target_entity_id != target_entity_id:
        raise ModelOutputError(
            f"Develop output targeted entity #{report.target_entity_id}, expected "
            f"#{target_entity_id}."
        )
    allowed_entity_ids = {int(entity["id"]) for entity in context.entities}
    allowed_source_ids = {int(source["id"]) for source in context.sources}
    for index, item in enumerate(report.developments, start=1):
        _validate_references(
            label="item",
            index=index,
            related_entity_ids=item.related_entity_ids,
            source_ids=item.source_ids,
            target_entity_id=target_entity_id,
            allowed_entity_ids=allowed_entity_ids,
            allowed_source_ids=allowed_source_ids,
        )
    for index, branch in enumerate(report.branches, start=1):
        _validate_references(
            label="branch",
            index=index,
            related_entity_ids=branch.related_entity_ids,
            source_ids=branch.source_ids,
            target_entity_id=target_entity_id,
            allowed_entity_ids=allowed_entity_ids,
            allowed_source_ids=allowed_source_ids,
        )


def _persist_development(
    *, workstream_id: int, target_entity_id: int, report: DevelopmentReport
) -> DevelopOutcome:
    development_ids: list[int] = []
    branch_ids: list[int] = []
    with connect() as con:
        for item in report.developments:
            entity_type = DEVELOPMENT_ARTIFACT_TYPES[item.item_type]
            title_prefix = item.item_type.replace("_", " ").title()
            title = f"{title_prefix}: {item.statement}"[:240]
            body = (
                f"{item.statement}\n\nReasoning: {item.reasoning_summary}\n\n"
                f"Model epistemic status: {item.epistemic_status}\n"
                f"Related entity IDs: {item.related_entity_ids}\n"
                f"Cited source IDs: {item.source_ids}"
            )
            entity_id = add_entity(
                con,
                entity_type,
                title,
                body=body,
                trust_state="quarantined",
                source_ids=item.source_ids,
                generated_by_llm=True,
            )
            set_attribute(con, entity_id, "develop_item_type", item.item_type)
            set_attribute(con, entity_id, "model_epistemic_status", item.epistemic_status)
            set_attribute(
                con,
                entity_id,
                "related_entity_ids",
                json.dumps(item.related_entity_ids),
            )
            link_workstream_entity(con, workstream_id, entity_id, "created")
            development_ids.append(entity_id)

        for branch in report.branches:
            entity_type = BRANCH_ARTIFACT_TYPES[branch.status]
            title = f"Development branch ({branch.status}): {branch.name}"[:240]
            obligations = (
                "\n".join(f"- {obligation}" for obligation in branch.proof_obligations)
                or "- none reported"
            )
            body = (
                f"Objective: {branch.objective}\n\nTechnical plan: {branch.technical_plan}\n\n"
                f"Proof obligations:\n{obligations}\n\n"
                f"Branch status: {branch.status}\n"
                f"Model epistemic status: {branch.epistemic_status}\n"
                f"Related entity IDs: {branch.related_entity_ids}\n"
                f"Cited source IDs: {branch.source_ids}"
            )
            entity_id = add_entity(
                con,
                entity_type,
                title,
                body=body,
                trust_state="quarantined",
                source_ids=branch.source_ids,
                generated_by_llm=True,
            )
            set_attribute(con, entity_id, "develop_branch_name", branch.name)
            set_attribute(con, entity_id, "develop_branch_status", branch.status)
            set_attribute(con, entity_id, "model_epistemic_status", branch.epistemic_status)
            set_attribute(
                con,
                entity_id,
                "related_entity_ids",
                json.dumps(branch.related_entity_ids),
            )
            link_workstream_entity(con, workstream_id, entity_id, "created")
            branch_ids.append(entity_id)

        branch_counts = {
            status: sum(branch.status == status for branch in report.branches)
            for status in BRANCH_ARTIFACT_TYPES
        }
        unresolved = (
            "; ".join(report.could_not_determine)
            if report.could_not_determine
            else "none reported"
        )
        status_summary = ", ".join(
            f"{status}={count}" for status, count in branch_counts.items() if count
        )
        summary = (
            f"Development completed. Created {len(development_ids)} technical artifact(s) "
            f"and {len(branch_ids)} branch artifact(s) ({status_summary}). "
            f"{report.summary} Could not determine: {unresolved}"
        )
        set_workstream_status(con, workstream_id, "completed", summary=summary)

    return DevelopOutcome(
        workstream_id=workstream_id,
        target_entity_id=target_entity_id,
        development_artifact_ids=tuple(development_ids),
        branch_artifact_ids=tuple(branch_ids),
    )


def _mark_workstream_error(workstream_id: int, error: Exception) -> None:
    with connect() as con:
        set_workstream_status(
            con,
            workstream_id,
            "error",
            summary=f"Develop execution error: {type(error).__name__}: {error}"[:4000],
        )


def develop(workstream_id: int, provider_name: str) -> DevelopOutcome:
    """Run exactly one provider call over one selected develop-workstream context."""
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
    if workstream["workstream_type"] != "develop":
        raise TheoryError(f"Workstream #{workstream_id} is not a develop workstream.")
    if workstream["status"] != "active":
        raise TheoryError(
            f"Develop workstream #{workstream_id} is {workstream['status']}, not active."
        )
    target = _primary_target(context, workstream_id)
    target_entity_id = int(target["id"])
    prompt = _develop_prompt(context, target)
    estimated_max_cost = budget_guard(
        cfg,
        model=model,
        prompt=prompt,
        max_output_tokens=DEVELOP_MAX_OUTPUT_TOKENS,
        purpose="develop",
    )

    try:
        provider = get_provider(provider_name)
        result = call_model(
            run_id=None,
            workstream_id=workstream_id,
            provider=provider,
            provider_name=provider_name,
            model=model,
            purpose="develop",
            prompt=prompt,
            max_output_tokens=DEVELOP_MAX_OUTPUT_TOKENS,
            estimated_max_cost_usd=estimated_max_cost,
        )
        report = parse_json_model(result.text, DevelopmentReport)
        _validate_report_references(report, context, target_entity_id)
        return _persist_development(
            workstream_id=workstream_id,
            target_entity_id=target_entity_id,
            report=report,
        )
    except Exception as exc:
        _mark_workstream_error(workstream_id, exc)
        raise
