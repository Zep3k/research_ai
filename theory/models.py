from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StructuredModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Formalization(StructuredModel):
    precise_question: str
    assumptions_to_pin_down: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    possible_variants: list[str] = Field(default_factory=list)
    immediate_failure_modes: list[str] = Field(default_factory=list)


class LiteratureHit(StructuredModel):
    openalex_id: str
    title: str
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    cited_by_count: int = 0
    abstract: str | None = None


class RetrievedSource(LiteratureHit):
    source_id: str
    retrieved_for_queries: list[str] = Field(default_factory=list)


class LiteratureSearch(StructuredModel):
    query: str
    status: Literal["ok", "failed"]
    openalex_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class LiteratureBundle(StructuredModel):
    sources: list[RetrievedSource] = Field(default_factory=list)
    searches: list[LiteratureSearch] = Field(default_factory=list)


class Finding(StructuredModel):
    statement: str
    epistemic_status: Literal["sourced", "inference", "speculation", "unresolved"]
    source_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sourced_findings_need_sources(self) -> "Finding":
        if self.epistemic_status == "sourced" and not self.source_ids:
            raise ValueError("sourced findings must cite at least one retrieved source_id")
        return self


class ResearchReport(StructuredModel):
    precise_question: str
    nearest_results: list[Finding] = Field(default_factory=list)
    reasons_to_continue: list[Finding] = Field(default_factory=list)
    reasons_to_stop_or_reframe: list[Finding] = Field(default_factory=list)
    hidden_assumptions: list[Finding] = Field(default_factory=list)
    counterexample_targets: list[Finding] = Field(default_factory=list)
    smallest_decisive_subproblems: list[Finding] = Field(default_factory=list)
    kill_conditions: list[Finding] = Field(default_factory=list)
    next_high_information_actions: list[Finding] = Field(default_factory=list)
    epistemic_notes: list[Finding] = Field(default_factory=list)


class ModelResult(StructuredModel):
    text: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
