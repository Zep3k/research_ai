from __future__ import annotations

import sqlite3
from enum import StrEnum
from typing import Iterable, TypeVar

from .errors import TheoryError, TrustError


class EntityType(StrEnum):
    PAPER = "Paper"
    THEOREM = "Theorem"
    DEFINITION = "Definition"
    ASSUMPTION = "Assumption"
    MODEL = "Model"
    TECHNIQUE = "Technique"
    RESEARCH_IDEA = "ResearchIdea"
    CONJECTURE = "Conjecture"
    OPEN_QUESTION = "OpenQuestion"
    PROOF_ATTEMPT = "ProofAttempt"
    LEMMA = "Lemma"
    COUNTEREXAMPLE = "Counterexample"
    OBSTRUCTION = "Obstruction"
    FAILED_APPROACH = "FailedApproach"
    FINDING = "Finding"


class RelationType(StrEnum):
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    EXTENDS = "EXTENDS"
    IMPROVES = "IMPROVES"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    BLOCKS = "BLOCKS"
    ATTEMPTS = "ATTEMPTS"
    FAILS_AT = "FAILS_AT"
    SOURCED_FROM = "SOURCED_FROM"


class TrustState(StrEnum):
    UNVERIFIED = "unverified"
    SOURCED = "sourced"
    INFERRED = "inferred"
    SPECULATIVE = "speculative"
    CONTRADICTED = "contradicted"
    QUARANTINED = "quarantined"


class EntityStatus(StrEnum):
    ACTIVE = "active"
    ABANDONED = "abandoned"
    RESOLVED = "resolved"


class RelationStatus(StrEnum):
    ACTIVE = "active"
    RETIRED = "retired"


class WorkstreamType(StrEnum):
    LITERATURE = "literature"
    EXPLORE = "explore"
    ATTACK = "attack"
    PROOF = "proof"


class WorkstreamStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    BLOCKED = "blocked"


class WorkstreamRole(StrEnum):
    INPUT = "input"
    CREATED = "created"
    MODIFIED = "modified"
    EVIDENCE = "evidence"
    BLOCKED_BY = "blocked_by"


class ReviewType(StrEnum):
    LITERATURE_CHECK = "literature_check"
    COUNTEREXAMPLE_ATTEMPT = "counterexample_attempt"
    INDEPENDENT_PROOF_ATTEMPT = "independent_proof_attempt"
    PROOF_CRITIQUE = "proof_critique"
    SOURCE_VERIFICATION = "source_verification"


class ReviewResult(StrEnum):
    NO_FLAW_FOUND = "no_flaw_found"
    ISSUE_FOUND = "issue_found"
    INCONCLUSIVE = "inconclusive"


EnumT = TypeVar("EnumT", bound=StrEnum)


def parse_enum(enum_type: type[EnumT], value: str | EnumT, label: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    normalized = value.strip().replace("-", "_").replace(" ", "_").casefold()
    for item in enum_type:
        candidates = {
            item.value.casefold(),
            item.name.casefold(),
            item.value.replace("_", "").casefold(),
        }
        if normalized in candidates or normalized.replace("_", "") in candidates:
            return item
    allowed = ", ".join(item.value for item in enum_type)
    raise TheoryError(f"Invalid {label} {value!r}. Expected one of: {allowed}.")


def require_entity(
    con: sqlite3.Connection, entity_id: int, project_id: int = 1
) -> sqlite3.Row:
    row = con.execute(
        "SELECT * FROM entities WHERE id=? AND project_id=?", (entity_id, project_id)
    ).fetchone()
    if row is None:
        raise TheoryError(f"Entity #{entity_id} does not exist in project #{project_id}.")
    return row


def require_source(
    con: sqlite3.Connection, source_id: int, project_id: int = 1
) -> sqlite3.Row:
    row = con.execute(
        "SELECT * FROM sources WHERE id=? AND project_id=?", (source_id, project_id)
    ).fetchone()
    if row is None:
        raise TheoryError(f"Source #{source_id} does not exist in project #{project_id}.")
    return row


def validate_entity_write(
    con: sqlite3.Connection,
    *,
    entity_type: str | EntityType,
    trust_state: str | TrustState,
    source_ids: Iterable[int] = (),
    project_id: int = 1,
    generated_by_llm: bool = False,
) -> tuple[EntityType, TrustState, tuple[int, ...]]:
    parsed_type = parse_enum(EntityType, entity_type, "entity type")
    parsed_trust = parse_enum(TrustState, trust_state, "trust state")
    sources = tuple(dict.fromkeys(source_ids))
    for source_id in sources:
        require_source(con, source_id, project_id)
    if parsed_trust is TrustState.SOURCED and not sources:
        origin = "LLM-generated " if generated_by_llm else ""
        raise TrustError(f"A {origin}sourced entity requires at least one persisted source.")
    return parsed_type, parsed_trust, sources


def validate_relation_write(
    con: sqlite3.Connection,
    *,
    source_entity_id: int,
    relation_type: str | RelationType,
    target_entity_id: int,
    evidence_source_id: int | None,
    trust_state: str | TrustState,
    project_id: int = 1,
    generated_by_llm: bool = False,
) -> tuple[RelationType, TrustState]:
    parsed_type = parse_enum(RelationType, relation_type, "relation type")
    parsed_trust = parse_enum(TrustState, trust_state, "trust state")
    require_entity(con, source_entity_id, project_id)
    require_entity(con, target_entity_id, project_id)
    if source_entity_id == target_entity_id:
        raise TheoryError("A relation cannot point an entity to itself.")
    if evidence_source_id is not None:
        require_source(con, evidence_source_id, project_id)
    if parsed_trust is TrustState.SOURCED and evidence_source_id is None:
        origin = "LLM-generated " if generated_by_llm else ""
        raise TrustError(f"A {origin}sourced relation requires an evidence source.")
    return parsed_type, parsed_trust


_ALLOWED_TRANSITIONS: dict[TrustState, set[TrustState]] = {
    TrustState.UNVERIFIED: {
        TrustState.SOURCED,
        TrustState.INFERRED,
        TrustState.SPECULATIVE,
        TrustState.CONTRADICTED,
        TrustState.QUARANTINED,
    },
    TrustState.SOURCED: {
        TrustState.UNVERIFIED,
        TrustState.INFERRED,
        TrustState.CONTRADICTED,
        TrustState.QUARANTINED,
    },
    TrustState.INFERRED: {
        TrustState.UNVERIFIED,
        TrustState.SOURCED,
        TrustState.SPECULATIVE,
        TrustState.CONTRADICTED,
        TrustState.QUARANTINED,
    },
    TrustState.SPECULATIVE: {
        TrustState.UNVERIFIED,
        TrustState.SOURCED,
        TrustState.INFERRED,
        TrustState.CONTRADICTED,
        TrustState.QUARANTINED,
    },
    TrustState.CONTRADICTED: {TrustState.UNVERIFIED, TrustState.QUARANTINED},
    TrustState.QUARANTINED: {TrustState.UNVERIFIED},
}


def validate_trust_transition(
    current: str | TrustState,
    desired: str | TrustState,
    *,
    has_provenance: bool,
) -> TrustState:
    old = parse_enum(TrustState, current, "current trust state")
    new = parse_enum(TrustState, desired, "trust state")
    if old == new:
        return new
    if new not in _ALLOWED_TRANSITIONS[old]:
        raise TrustError(
            f"Trust transition {old.value!r} -> {new.value!r} is not allowed; "
            "move through 'unverified' for explicit re-evaluation."
        )
    if new is TrustState.SOURCED and not has_provenance:
        raise TrustError("Trust state 'sourced' requires persisted provenance.")
    return new
