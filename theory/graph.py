from __future__ import annotations

import sqlite3
import re
from dataclasses import dataclass
from typing import Iterable

from .db import utcnow
from .errors import TheoryError
from .trust import (
    EntityStatus,
    EntityType,
    RelationStatus,
    RelationType,
    ReviewResult,
    ReviewType,
    TrustState,
    WorkstreamRole,
    WorkstreamStatus,
    WorkstreamType,
    parse_enum,
    require_entity,
    require_source,
    validate_entity_write,
    validate_relation_write,
    validate_trust_transition,
)


THEOREM_LIKE_TYPES = {
    EntityType.THEOREM.value,
    EntityType.LEMMA.value,
    EntityType.CONJECTURE.value,
    EntityType.OPEN_QUESTION.value,
}


@dataclass(frozen=True)
class AttributeDelta:
    unchanged: tuple[tuple[str, str], ...]
    changed: tuple[tuple[str, str, str], ...]
    only_a: tuple[tuple[str, str], ...]
    only_b: tuple[tuple[str, str], ...]


def normalize_attribute_key(key: str) -> str:
    """Apply syntax-only normalization without imposing a domain ontology."""
    normalized = re.sub(r"[\s-]+", "_", key.strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def add_entity(
    con: sqlite3.Connection,
    entity_type: str | EntityType,
    title: str,
    *,
    body: str = "",
    status: str | EntityStatus = EntityStatus.ACTIVE,
    trust_state: str | TrustState = TrustState.UNVERIFIED,
    confidence: float = 0.0,
    source_ids: Iterable[int] = (),
    project_id: int = 1,
    generated_by_llm: bool = False,
) -> int:
    title = title.strip()
    if not title:
        raise TheoryError("Entity title cannot be empty.")
    if not 0.0 <= confidence <= 1.0:
        raise TheoryError("Entity confidence must be between 0 and 1.")
    parsed_type, parsed_trust, sources = validate_entity_write(
        con,
        entity_type=entity_type,
        trust_state=trust_state,
        source_ids=source_ids,
        project_id=project_id,
        generated_by_llm=generated_by_llm,
    )
    parsed_status = parse_enum(EntityStatus, status, "entity status")
    now = utcnow()
    # A sourced entity is staged as unverified so its provenance can be attached
    # before the database trigger permits the trust-state promotion.
    inserted_trust = (
        TrustState.UNVERIFIED if parsed_trust is TrustState.SOURCED else parsed_trust
    )
    cur = con.execute(
        """
        INSERT INTO entities(
            project_id,entity_type,title,body,status,trust_state,confidence,
            created_at,updated_at
        ) VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            project_id,
            parsed_type.value,
            title,
            body,
            parsed_status.value,
            inserted_trust.value,
            confidence,
            now,
            now,
        ),
    )
    entity_id = int(cur.lastrowid)
    for source_id in sources:
        con.execute(
            "INSERT INTO entity_sources(project_id,entity_id,source_id,created_at) VALUES(?,?,?,?)",
            (project_id, entity_id, source_id, now),
        )
    if parsed_trust is TrustState.SOURCED:
        con.execute(
            "UPDATE entities SET trust_state=?,updated_at=? WHERE id=?",
            (parsed_trust.value, now, entity_id),
        )
    return entity_id


def set_entity_trust(
    con: sqlite3.Connection,
    entity_id: int,
    trust_state: str | TrustState,
    *,
    project_id: int = 1,
) -> None:
    entity = require_entity(con, entity_id, project_id)
    has_identifiable_provenance = con.execute(
        """
        SELECT 1
        FROM entity_sources es
        JOIN sources s ON s.id=es.source_id
        WHERE es.entity_id=?
          AND (
              s.paper_entity_id IS NOT NULL OR
              length(trim(COALESCE(s.external_url, ''))) > 0
          )
        LIMIT 1
        """,
        (entity_id,),
    ).fetchone() is not None
    desired = validate_trust_transition(
        entity["trust_state"],
        trust_state,
        has_identifiable_provenance=has_identifiable_provenance,
    )
    con.execute(
        "UPDATE entities SET trust_state=?,updated_at=? WHERE id=? AND project_id=?",
        (desired.value, utcnow(), entity_id, project_id),
    )


def set_attribute(
    con: sqlite3.Connection,
    entity_id: int,
    key: str,
    value: str,
    *,
    project_id: int = 1,
) -> None:
    require_entity(con, entity_id, project_id)
    key = normalize_attribute_key(key)
    if not key:
        raise TheoryError("Attribute key cannot be empty.")
    now = utcnow()
    con.execute(
        """
        INSERT INTO entity_attributes(
            project_id,entity_id,key,value,created_at,updated_at
        ) VALUES(?,?,?,?,?,?)
        ON CONFLICT(entity_id,key) DO UPDATE SET
            value=excluded.value,updated_at=excluded.updated_at
        """,
        (project_id, entity_id, key, value, now, now),
    )


def list_attributes(con: sqlite3.Connection, entity_id: int) -> dict[str, str]:
    require_entity(con, entity_id)
    rows = con.execute(
        "SELECT key,value FROM entity_attributes WHERE entity_id=? ORDER BY key", (entity_id,)
    ).fetchall()
    return {row["key"]: row["value"] for row in rows}


def add_relation(
    con: sqlite3.Connection,
    source_entity_id: int,
    relation_type: str | RelationType,
    target_entity_id: int,
    *,
    evidence_source_id: int | None = None,
    status: str | RelationStatus = RelationStatus.ACTIVE,
    trust_state: str | TrustState = TrustState.UNVERIFIED,
    confidence: float = 0.0,
    project_id: int = 1,
    generated_by_llm: bool = False,
) -> int:
    if not 0.0 <= confidence <= 1.0:
        raise TheoryError("Relation confidence must be between 0 and 1.")
    parsed_type, parsed_trust = validate_relation_write(
        con,
        source_entity_id=source_entity_id,
        relation_type=relation_type,
        target_entity_id=target_entity_id,
        evidence_source_id=evidence_source_id,
        trust_state=trust_state,
        project_id=project_id,
        generated_by_llm=generated_by_llm,
    )
    parsed_status = parse_enum(RelationStatus, status, "relation status")
    cur = con.execute(
        """
        INSERT INTO relations(
            project_id,source_entity_id,relation_type,target_entity_id,
            evidence_source_id,status,trust_state,confidence,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            project_id,
            source_entity_id,
            parsed_type.value,
            target_entity_id,
            evidence_source_id,
            parsed_status.value,
            parsed_trust.value,
            confidence,
            utcnow(),
        ),
    )
    return int(cur.lastrowid)


def add_source(
    con: sqlite3.Connection,
    entity_id: int,
    *,
    paper_entity_id: int | None = None,
    source_type: str = "paper_locator",
    page: int | None = None,
    section: str | None = None,
    theorem: str | None = None,
    excerpt: str | None = None,
    external_url: str | None = None,
    project_id: int = 1,
) -> int:
    require_entity(con, entity_id, project_id)
    source_type = source_type.strip()
    if not source_type:
        raise TheoryError("Source type cannot be empty.")
    if page is not None and page <= 0:
        raise TheoryError("Source page must be positive.")
    if paper_entity_id is not None:
        paper = require_entity(con, paper_entity_id, project_id)
        if paper["entity_type"] != EntityType.PAPER.value:
            raise TheoryError(f"Entity #{paper_entity_id} is not a Paper.")
    if not any(
        value is not None
        for value in (paper_entity_id, page, section, theorem, excerpt, external_url)
    ):
        raise TheoryError("A source needs a paper or at least one concrete locator.")
    cur = con.execute(
        """
        INSERT INTO sources(
            project_id,paper_entity_id,source_type,page,section,theorem,
            excerpt,external_url,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            project_id,
            paper_entity_id,
            source_type,
            page,
            section,
            theorem,
            excerpt,
            external_url,
            utcnow(),
        ),
    )
    source_id = int(cur.lastrowid)
    attach_source(con, entity_id, source_id, project_id=project_id)
    return source_id


def attach_source(
    con: sqlite3.Connection,
    entity_id: int,
    source_id: int,
    *,
    project_id: int = 1,
) -> None:
    require_entity(con, entity_id, project_id)
    require_source(con, source_id, project_id)
    con.execute(
        """
        INSERT OR IGNORE INTO entity_sources(project_id,entity_id,source_id,created_at)
        VALUES(?,?,?,?)
        """,
        (project_id, entity_id, source_id, utcnow()),
    )


def create_workstream(
    con: sqlite3.Connection,
    workstream_type: str | WorkstreamType,
    goal: str,
    *,
    status: str | WorkstreamStatus = WorkstreamStatus.ACTIVE,
    summary: str = "",
    project_id: int = 1,
) -> int:
    goal = goal.strip()
    if not goal:
        raise TheoryError("Workstream goal cannot be empty.")
    parsed_type = parse_enum(WorkstreamType, workstream_type, "workstream type")
    parsed_status = parse_enum(WorkstreamStatus, status, "workstream status")
    if parsed_status is WorkstreamStatus.LEGACY_FAILED:
        raise TheoryError("legacy_failed is reserved for migrated V0.2 workstreams.")
    now = utcnow()
    cur = con.execute(
        """
        INSERT INTO workstreams(
            project_id,workstream_type,goal,status,summary,created_at,updated_at
        ) VALUES(?,?,?,?,?,?,?)
        """,
        (project_id, parsed_type.value, goal, parsed_status.value, summary, now, now),
    )
    return int(cur.lastrowid)


def set_workstream_status(
    con: sqlite3.Connection,
    workstream_id: int,
    status: str | WorkstreamStatus,
    *,
    summary: str | None = None,
) -> None:
    existing = con.execute("SELECT 1 FROM workstreams WHERE id=?", (workstream_id,)).fetchone()
    if existing is None:
        raise TheoryError(f"Workstream #{workstream_id} does not exist.")
    parsed = parse_enum(WorkstreamStatus, status, "workstream status")
    if parsed is WorkstreamStatus.LEGACY_FAILED:
        raise TheoryError("legacy_failed is reserved for migrated V0.2 workstreams.")
    if summary is None:
        con.execute(
            "UPDATE workstreams SET status=?,updated_at=? WHERE id=?",
            (parsed.value, utcnow(), workstream_id),
        )
    else:
        con.execute(
            "UPDATE workstreams SET status=?,summary=?,updated_at=? WHERE id=?",
            (parsed.value, summary, utcnow(), workstream_id),
        )


def link_workstream_entity(
    con: sqlite3.Connection,
    workstream_id: int,
    entity_id: int,
    role: str | WorkstreamRole,
    *,
    project_id: int = 1,
) -> None:
    workstream = con.execute(
        "SELECT * FROM workstreams WHERE id=? AND project_id=?", (workstream_id, project_id)
    ).fetchone()
    if workstream is None:
        raise TheoryError(f"Workstream #{workstream_id} does not exist.")
    require_entity(con, entity_id, project_id)
    parsed_role = parse_enum(WorkstreamRole, role, "workstream entity role")
    con.execute(
        """
        INSERT OR IGNORE INTO workstream_entities(
            project_id,workstream_id,entity_id,role,created_at
        ) VALUES(?,?,?,?,?)
        """,
        (project_id, workstream_id, entity_id, parsed_role.value, utcnow()),
    )


def add_review(
    con: sqlite3.Connection,
    review_type: str | ReviewType,
    result: str | ReviewResult,
    *,
    target_entity_id: int | None = None,
    workstream_id: int | None = None,
    issues: str = "",
    provider: str | None = None,
    model: str | None = None,
    run_id: int | None = None,
    project_id: int = 1,
) -> int:
    if (target_entity_id is None) == (workstream_id is None):
        raise TheoryError("A review must target exactly one entity or workstream.")
    if target_entity_id is not None:
        require_entity(con, target_entity_id, project_id)
    if workstream_id is not None:
        row = con.execute(
            "SELECT 1 FROM workstreams WHERE id=? AND project_id=?",
            (workstream_id, project_id),
        ).fetchone()
        if row is None:
            raise TheoryError(f"Workstream #{workstream_id} does not exist.")
    parsed_type = parse_enum(ReviewType, review_type, "review type")
    parsed_result = parse_enum(ReviewResult, result, "review result")
    cur = con.execute(
        """
        INSERT INTO reviews(
            project_id,target_entity_id,workstream_id,review_type,result,issues,
            provider,model,run_id,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (
            project_id,
            target_entity_id,
            workstream_id,
            parsed_type.value,
            parsed_result.value,
            issues,
            provider,
            model,
            run_id,
            utcnow(),
        ),
    )
    return int(cur.lastrowid)


def compare_attributes(
    con: sqlite3.Connection, entity_a_id: int, entity_b_id: int
) -> AttributeDelta:
    entity_a = require_entity(con, entity_a_id)
    entity_b = require_entity(con, entity_b_id)
    for label, entity in (("A", entity_a), ("B", entity_b)):
        if entity["entity_type"] not in THEOREM_LIKE_TYPES:
            raise TheoryError(
                f"Entity {label} #{entity['id']} is {entity['entity_type']}, not theorem-like."
            )
    attrs_a = list_attributes(con, entity_a_id)
    attrs_b = list_attributes(con, entity_b_id)
    unchanged: list[tuple[str, str]] = []
    changed: list[tuple[str, str, str]] = []
    only_a: list[tuple[str, str]] = []
    only_b: list[tuple[str, str]] = []
    for key in sorted(attrs_a.keys() | attrs_b.keys()):
        if key in attrs_a and key in attrs_b:
            if attrs_a[key] == attrs_b[key]:
                unchanged.append((key, attrs_a[key]))
            else:
                changed.append((key, attrs_a[key], attrs_b[key]))
        elif key in attrs_a:
            only_a.append((key, attrs_a[key]))
        else:
            only_b.append((key, attrs_b[key]))
    return AttributeDelta(tuple(unchanged), tuple(changed), tuple(only_a), tuple(only_b))
