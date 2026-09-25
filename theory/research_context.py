from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .db import connect
from .errors import TheoryError
from .trust import require_entity


@dataclass(frozen=True)
class ResearchContext:
    target_entity: dict[str, Any] | None = None
    workstream: dict[str, Any] | None = None
    entities: tuple[dict[str, Any], ...] = ()
    relations: tuple[dict[str, Any], ...] = ()
    attributes: dict[int, dict[str, str]] = field(default_factory=dict)
    sources: tuple[dict[str, Any], ...] = ()
    workstream_links: tuple[dict[str, Any], ...] = ()
    selections: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rows_as_dicts(rows: Iterable[sqlite3.Row]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(row) for row in rows)


def _build_context(
    con: sqlite3.Connection,
    base_entity_ids: set[int],
    *,
    target_entity_id: int | None = None,
    workstream: sqlite3.Row | None = None,
) -> ResearchContext:
    if not base_entity_ids:
        return ResearchContext(workstream=dict(workstream) if workstream else None)

    placeholders = ",".join("?" for _ in base_entity_ids)
    base_params = tuple(sorted(base_entity_ids))
    relation_rows = con.execute(
        f"""
        SELECT * FROM relations
        WHERE source_entity_id IN ({placeholders}) OR target_entity_id IN ({placeholders})
        ORDER BY id
        """,
        base_params + base_params,
    ).fetchall()
    entity_ids = set(base_entity_ids)
    for relation in relation_rows:
        entity_ids.add(int(relation["source_entity_id"]))
        entity_ids.add(int(relation["target_entity_id"]))

    entity_placeholders = ",".join("?" for _ in entity_ids)
    entity_params = tuple(sorted(entity_ids))
    entity_rows = con.execute(
        f"SELECT * FROM entities WHERE id IN ({entity_placeholders}) ORDER BY id",
        entity_params,
    ).fetchall()
    attribute_rows = con.execute(
        f"""
        SELECT entity_id,key,value FROM entity_attributes
        WHERE entity_id IN ({entity_placeholders}) ORDER BY entity_id,key
        """,
        entity_params,
    ).fetchall()
    source_rows = con.execute(
        f"""
        SELECT es.entity_id,s.* FROM entity_sources es
        JOIN sources s ON s.id=es.source_id
        WHERE es.entity_id IN ({entity_placeholders})
        ORDER BY es.entity_id,s.id
        """,
        entity_params,
    ).fetchall()
    link_rows = con.execute(
        f"""
        SELECT * FROM workstream_entities
        WHERE entity_id IN ({entity_placeholders})
        ORDER BY workstream_id,entity_id,role
        """,
        entity_params,
    ).fetchall()

    attributes: dict[int, dict[str, str]] = {}
    for row in attribute_rows:
        attributes.setdefault(int(row["entity_id"]), {})[row["key"]] = row["value"]

    direct_ids = entity_ids - base_entity_ids
    type_by_id = {int(row["id"]): row["entity_type"] for row in entity_rows}
    sourced_ids = {
        int(row["id"]) for row in entity_rows if row["trust_state"] == "sourced"
    }
    blocker_ids: set[int] = set()
    for relation in relation_rows:
        if relation["relation_type"] != "BLOCKS":
            continue
        if int(relation["target_entity_id"]) in base_entity_ids:
            blocker_ids.add(int(relation["source_entity_id"]))

    selections = {
        "assumptions": tuple(sorted(i for i in direct_ids if type_by_id.get(i) == "Assumption")),
        "nearest_theorems": tuple(
            sorted(i for i in direct_ids if type_by_id.get(i) in {"Theorem", "Lemma"})
        ),
        "proof_attempts": tuple(
            sorted(i for i in direct_ids if type_by_id.get(i) == "ProofAttempt")
        ),
        "counterexamples": tuple(
            sorted(i for i in direct_ids if type_by_id.get(i) == "Counterexample")
        ),
        "blockers": tuple(sorted(blocker_ids)),
        "source_backed_findings": tuple(
            sorted(i for i in direct_ids & sourced_ids if type_by_id.get(i) == "Finding")
        ),
    }
    target = next(
        (dict(row) for row in entity_rows if int(row["id"]) == target_entity_id), None
    )
    return ResearchContext(
        target_entity=target,
        workstream=dict(workstream) if workstream else None,
        entities=_rows_as_dicts(entity_rows),
        relations=_rows_as_dicts(relation_rows),
        attributes=attributes,
        sources=_rows_as_dicts(source_rows),
        workstream_links=_rows_as_dicts(link_rows),
        selections=selections,
    )


def for_entity(entity_id: int) -> ResearchContext:
    """Build a deterministic one-hop graph neighborhood for an entity."""
    with connect() as con:
        require_entity(con, entity_id)
        return _build_context(con, {entity_id}, target_entity_id=entity_id)


def for_workstream(workstream_id: int) -> ResearchContext:
    """Build context from durable workstream inputs/artifacts and their neighbors."""
    with connect() as con:
        workstream = con.execute(
            "SELECT * FROM workstreams WHERE id=?", (workstream_id,)
        ).fetchone()
        if workstream is None:
            raise TheoryError(f"Workstream #{workstream_id} does not exist.")
        rows = con.execute(
            "SELECT entity_id FROM workstream_entities WHERE workstream_id=? ORDER BY entity_id",
            (workstream_id,),
        ).fetchall()
        return _build_context(
            con, {int(row["entity_id"]) for row in rows}, workstream=workstream
        )
