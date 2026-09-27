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
    input_tokens: int = Field(default=0, ge=0, strict=True)
    uncached_input_tokens: int = Field(default=0, ge=0, strict=True)
    cache_read_input_tokens: int = Field(default=0, ge=0, strict=True)
    cache_write_input_tokens: int = Field(default=0, ge=0, strict=True)
    cache_write_5m_input_tokens: int = Field(default=0, ge=0, strict=True)
    cache_write_1h_input_tokens: int = Field(default=0, ge=0, strict=True)
    output_tokens: int = Field(default=0, ge=0, strict=True)
    reasoning_tokens: int | None = Field(default=None, ge=0, strict=True)
    uncached_input_cost_usd: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    cache_read_cost_usd: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    cache_write_cost_usd: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    output_cost_usd: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    cost_usd: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    response_status: str = "completed"
    incomplete_reason: str | None = None

    @model_validator(mode="after")
    def validate_accounting(self) -> "ModelResult":
        if self.input_tokens != (
            self.uncached_input_tokens + self.cache_read_input_tokens + self.cache_write_input_tokens
        ):
            raise ValueError("input_tokens must equal uncached + cache-read + cache-write")
        if self.cache_write_input_tokens != (
            self.cache_write_5m_input_tokens + self.cache_write_1h_input_tokens
        ):
            raise ValueError("cache-write tokens must equal the two accounting buckets")
        if self.cost_usd != (
            self.uncached_input_cost_usd + self.cache_read_cost_usd
            + self.cache_write_cost_usd + self.output_cost_usd
        ):
            raise ValueError("cost_usd must equal the four cost components")
        return self
