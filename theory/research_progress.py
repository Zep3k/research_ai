"""Internal deterministic progress telemetry, never an execution-model output schema."""
from __future__ import annotations

import json
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


ProgressEventKind = Literal[
    "obligation_resolved", "obligation_bypassed", "obligation_retracted", "candidate_challenged", "branch_closed",
    "candidate_survived_attack", "candidate_tested_inconclusive",
    "obligation_audited", "candidate_created", "obligation_created", "obligation_reactivated", "frontier_expanded",
    "duplicate_only", "no_progress",
]
ProgressLevel = Literal["closure", "validation", "construction", "exploration", "none"]
# This order is descriptive precedence, not a strategic ranking or reward.
PROGRESS_PRECEDENCE = get_args(ProgressEventKind)
EVENT_LEVEL: dict[str, ProgressLevel] = {
    "obligation_resolved": "closure",
    "obligation_retracted": "closure",  # Historical telemetry remains readable.
    "obligation_bypassed": "closure",
    "obligation_reactivated": "exploration",
    "obligation_audited": "validation",
    "candidate_challenged": "closure",
    "branch_closed": "closure",
    "candidate_survived_attack": "validation",
    "candidate_tested_inconclusive": "validation",
    "candidate_created": "construction",
    "obligation_created": "exploration",
    "frontier_expanded": "exploration",
    "duplicate_only": "none",
    "no_progress": "none",
}
TEST_EVENTS = {"candidate_challenged", "candidate_survived_attack", "candidate_tested_inconclusive"}


class ProgressEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    kind: ProgressEventKind
    entity_ids: tuple[int, ...] = ()
    obligation_ids: tuple[int, ...] = ()

    @field_validator("entity_ids", "obligation_ids")
    @classmethod
    def canonical_ids(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value <= 0 for value in values) or tuple(sorted(set(values))) != values:
            raise ValueError("Progress IDs must be positive, unique and sorted")
        return values

    @model_validator(mode="after")
    def valid_subjects(self) -> "ProgressEvent":
        if self.kind in {"duplicate_only", "no_progress"}:
            if self.entity_ids or self.obligation_ids:
                raise ValueError("Empty-result events cannot have subjects")
        elif not self.entity_ids:
            raise ValueError("Progress events require an entity subject")
        if self.kind in {"obligation_resolved", "obligation_retracted", "obligation_bypassed", "obligation_reactivated", "obligation_audited", "candidate_created", "obligation_created"} and not self.obligation_ids:
            raise ValueError("Obligation progress requires an obligation subject")
        return self


def event_count(events: tuple[ProgressEvent, ...], kinds: set[str], *, obligations: bool = False) -> int:
    return len({
        entity_id for event in events if event.kind in kinds
        for entity_id in (event.obligation_ids if obligations else event.entity_ids)
    })


class ProgressRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    progress_class: ProgressEventKind
    progress_level: ProgressLevel
    events: tuple[ProgressEvent, ...] = Field(min_length=1)
    material_progress: bool
    resolution_progress: bool
    open_obligations_before: int = Field(ge=0)
    open_obligations_after: int = Field(ge=0)
    resolved_obligation_count: int = Field(ge=0)
    new_obligation_count: int = Field(ge=0)
    candidate_created_count: int = Field(ge=0)
    candidate_tested_count: int = Field(ge=0)
    closed_branch_count: int = Field(ge=0)
    accepted_artifact_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)

    @model_validator(mode="after")
    def consistent_record(self) -> "ProgressRecord":
        kinds = {event.kind for event in self.events}
        primary = next(kind for kind in PROGRESS_PRECEDENCE if kind in kinds)
        if self.progress_class != primary or self.progress_level != EVENT_LEVEL[primary]:
            raise ValueError("Progress class/level must follow event precedence")
        if self.material_progress != (self.progress_level != "none"):
            raise ValueError("material_progress must project progress_level")
        if self.resolution_progress != (self.open_obligations_after < self.open_obligations_before):
            raise ValueError("resolution_progress must reflect actual open-obligation counts")
        expected_counts = {
            "resolved_obligation_count": event_count(self.events, {"obligation_resolved"}, obligations=True),
            "new_obligation_count": event_count(self.events, {"obligation_created"}, obligations=True),
            "candidate_created_count": event_count(self.events, {"candidate_created"}),
            "candidate_tested_count": event_count(self.events, TEST_EVENTS),
            "closed_branch_count": event_count(self.events, {"branch_closed"}),
            "accepted_artifact_count": event_count(self.events, {
                "candidate_created", "obligation_created", "branch_closed", "frontier_expanded",
            }),
        }
        if any(getattr(self, key) != value for key, value in expected_counts.items()):
            raise ValueError("Progress counters must match their persisted event subjects")
        if kinds & {"duplicate_only", "no_progress"}:
            expected = "duplicate_only" if self.duplicate_count else "no_progress"
            if len(self.events) != 1 or primary != expected or self.accepted_artifact_count:
                raise ValueError("Empty-result events must be exclusive and match duplicate count")
        return self

    @classmethod
    def from_events(
        cls, events: tuple[ProgressEvent, ...], *, open_obligations_before: int,
        open_obligations_after: int, accepted_artifact_count: int, duplicate_count: int,
    ) -> "ProgressRecord":
        ordered = tuple(sorted(events, key=lambda event: (
            PROGRESS_PRECEDENCE.index(event.kind), event.entity_ids, event.obligation_ids,
        )))
        if not ordered:
            ordered = (ProgressEvent(kind="duplicate_only" if duplicate_count else "no_progress"),)
        primary = ordered[0].kind
        return cls(
            progress_class=primary, progress_level=EVENT_LEVEL[primary], events=ordered,
            material_progress=EVENT_LEVEL[primary] != "none",
            resolution_progress=open_obligations_after < open_obligations_before,
            open_obligations_before=open_obligations_before, open_obligations_after=open_obligations_after,
            resolved_obligation_count=event_count(ordered, {"obligation_resolved"}, obligations=True),
            new_obligation_count=event_count(ordered, {"obligation_created"}, obligations=True),
            candidate_created_count=event_count(ordered, {"candidate_created"}),
            candidate_tested_count=event_count(ordered, TEST_EVENTS),
            closed_branch_count=event_count(ordered, {"branch_closed"}),
            accepted_artifact_count=accepted_artifact_count, duplicate_count=duplicate_count,
        )

    def persistence_fields(self) -> dict:
        fields = self.model_dump(exclude={"events"})
        fields["material_progress"] = int(self.material_progress)
        fields["resolution_progress"] = int(self.resolution_progress)
        fields["progress_events_json"] = json.dumps([event.model_dump() for event in self.events])
        return fields


def progress_event_kinds(raw: str | None) -> tuple[ProgressEventKind, ...] | None:
    """Compact stored telemetry without inferring anything for legacy rows."""
    if raw is None:
        return None
    events = TypeAdapter(tuple[ProgressEvent, ...]).validate_json(raw)
    return tuple(dict.fromkeys(event.kind for event in events))
