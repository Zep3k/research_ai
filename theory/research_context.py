from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .db import connect
from .errors import TheoryError
from .trust import TrustState, require_entity, source_has_identifiable_origin


EPISTEMIC_GUIDANCE = {
    TrustState.SOURCED.value: (
        "SOURCED / SOURCE-BACKED EVIDENCE (NOT THEOREM-VERIFIED)",
        "A persisted locator names an origin claimed as evidence. Do not treat this as "
        "mathematical verification.",
    ),
    TrustState.INFERRED.value: (
        "INFERRED",
        "Treat as a provisional conclusion derived from evidence or reasoning.",
    ),
    TrustState.SPECULATIVE.value: (
        "SPECULATIVE",
        "Treat only as a hypothesis or attack direction.",
    ),
    TrustState.UNVERIFIED.value: (
        "UNVERIFIED",
        "Do not assume true; it has not passed a provenance or review gate.",
    ),
    TrustState.CONTRADICTED.value: (
        "CONTRADICTED / COUNTEREVIDENCE",
        "Treat as counterevidence or research history, not as an active assumption.",
    ),
    TrustState.QUARANTINED.value: (
        "QUARANTINED — DO NOT ASSUME TRUE",
        "Never use as a fact or assumption unless a human explicitly reconsiders it.",
    ),
}


@dataclass(frozen=True)
class EpistemicGroup:
    label: str
    model_instruction: str
    entities: tuple[dict[str, Any], ...] = ()
    relations: tuple[dict[str, Any], ...] = ()


def _empty_epistemic() -> dict[str, EpistemicGroup]:
    return {
        state: EpistemicGroup(label=label, model_instruction=instruction)
        for state, (label, instruction) in EPISTEMIC_GUIDANCE.items()
    }


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
    epistemic: dict[str, EpistemicGroup] = field(default_factory=_empty_epistemic)
    context_scope: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_model_payload(self) -> dict[str, Any]:
        """Serialize context with epistemic policy and partitions foregrounded."""
        return {
            "epistemic_policy": [
                "Sourced means source-backed, not theorem-verified.",
                "Inferred objects are provisional conclusions.",
                "Speculative objects are hypotheses.",
                "Contradicted objects are counterevidence or history.",
                "Quarantined objects must never be used as assumptions unless explicitly reconsidered.",
            ],
            "workstream": self.workstream,
            "target_entity": self.target_entity,
            # Execution routing provenance stays in the graph for audit, not model context.
            "attributes_by_entity_id": {
                entity_id: {key: value for key, value in attrs.items()
                            if key != "research_idea_origin"}
                for entity_id, attrs in self.attributes.items()
            },
            "sources": self.sources,
            "workstream_links": self.workstream_links,
            "selections": self.selections,
            "active_relations": self.relations,
            "epistemic_partitions": {
                state: asdict(group) for state, group in self.epistemic.items()
            },
            **({"context_scope": self.context_scope} if self.context_scope is not None else {}),
        }


def focus_research_context(
    full_context: ResearchContext,
    *,
    workstream_id: int,
    primary_entity_id: int,
    target_entity_id: int,
    focus_obligation_id: int | None = None,
    consumed_entity_ids: tuple[int, ...] = (),
    additional_entity_ids: tuple[int, ...] = (),
) -> ResearchContext:
    """Pure local view plus recorded dependency ancestry; omission is not a judgment."""
    anchors = {target_entity_id, *consumed_entity_ids, *additional_entity_ids}
    if focus_obligation_id is not None:
        anchors.add(focus_obligation_id)
    mandatory = anchors | {primary_entity_id} | {
        int(link["entity_id"]) for link in full_context.workstream_links
        if int(link["workstream_id"]) == workstream_id and link["role"] == "input"
    }
    available = {int(entity["id"]) for entity in full_context.entities}
    missing = mandatory - available
    if missing:
        raise TheoryError(f"Mandatory research context entities are absent: {sorted(missing)}")

    included = set(mandatory)
    active_relations = tuple(
        relation for relation in full_context.relations if relation["status"] == "active"
    )
    for relation in active_relations:
        endpoints = {int(relation["source_entity_id"]), int(relation["target_entity_id"])}
        if endpoints & anchors:
            included.update(endpoints)

    references_by_id: dict[int, set[int]] = {}
    for entity_id, attrs in full_context.attributes.items():
        references: set[int] = set()
        for key in ("related_entity_ids", "addresses_obligation_ids", "research_related_obligation_ids"):
            try:
                values = json.loads(attrs.get(key, "[]"))
            except (TypeError, ValueError):
                continue
            if isinstance(values, list):
                references.update(value for value in values if type(value) is int and value > 0)
        try:
            references.add(int(attrs["research_focus_obligation_id"]))
        except (KeyError, TypeError, ValueError):
            pass
        references_by_id[entity_id] = references
        if entity_id in anchors:
            included.update(references)
        if references & anchors:
            included.add(entity_id)

    # Follow explicit forward references of selected components, including their
    # recorded ancestors. Do not recursively expand reverse neighbors or traverse
    # contract inputs into unrelated branches. Finite visited sets handle cycles.
    inputs = {primary_entity_id} | {
        int(link["entity_id"]) for link in full_context.workstream_links
        if int(link["workstream_id"]) == workstream_id and link["role"] == "input"
    }
    for relation in active_relations:
        if relation["relation_type"] in {"DEPENDS_ON", "USES"}:
            references_by_id.setdefault(int(relation["source_entity_id"]), set()).add(
                int(relation["target_entity_id"])
            )
    pending = sorted(anchors - inputs)
    visited: set[int] = set()
    while pending:
        entity_id = pending.pop()
        if entity_id in visited or entity_id not in available:
            continue
        visited.add(entity_id)
        dependencies = references_by_id.get(entity_id, set()) & available
        included.update(dependencies)
        pending.extend(sorted(dependencies - visited - inputs))

    included &= available
    entities = tuple(entity for entity in full_context.entities if int(entity["id"]) in included)
    relations = tuple(
        relation for relation in active_relations
        if int(relation["source_entity_id"]) in included
        and int(relation["target_entity_id"]) in included
    )
    epistemic = {
        state: EpistemicGroup(
            label=label, model_instruction=instruction,
            entities=tuple(entity for entity in entities if entity["trust_state"] == state),
            relations=tuple(relation for relation in relations if relation["trust_state"] == state),
        )
        for state, (label, instruction) in EPISTEMIC_GUIDANCE.items()
    }
    return ResearchContext(
        target_entity=next(entity for entity in entities if int(entity["id"]) == target_entity_id),
        workstream=full_context.workstream,
        entities=entities, relations=relations,
        attributes={entity_id: attrs for entity_id, attrs in full_context.attributes.items()
                    if entity_id in included},
        sources=tuple(source for source in full_context.sources if int(source["entity_id"]) in included),
        workstream_links=tuple(link for link in full_context.workstream_links if int(link["entity_id"]) in included),
        selections={key: tuple(entity_id for entity_id in ids if entity_id in included)
                    for key, ids in full_context.selections.items()},
        epistemic=epistemic,
        context_scope={
            "mode": "focused_research_operation",
            "primary_entity_id": primary_entity_id,
            "target_entity_id": target_entity_id,
            "focus_obligation_id": focus_obligation_id,
            "consumed_entity_ids": list(consumed_entity_ids),
            "included_entity_ids": [int(entity["id"]) for entity in entities],
            "full_workstream_entity_count": len(full_context.entities),
            "focused_entity_count": len(entities),
        },
    )


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
        WHERE status='active'
          AND (source_entity_id IN ({placeholders}) OR target_entity_id IN ({placeholders}))
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
    raw_source_rows = con.execute(
        f"""
        SELECT es.entity_id,s.*,paper.title AS paper_title
        FROM entity_sources es
        JOIN sources s ON s.id=es.source_id
        LEFT JOIN entities paper ON paper.id=s.paper_entity_id
        WHERE es.entity_id IN ({entity_placeholders})
        ORDER BY es.entity_id,s.id
        """,
        entity_params,
    ).fetchall()
    source_rows = []
    for row in raw_source_rows:
        source = dict(row)
        source["identifiable_origin"] = source_has_identifiable_origin(row)
        source_rows.append(source)
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
    epistemic = _empty_epistemic()
    entity_dicts = _rows_as_dicts(entity_rows)
    relation_dicts = _rows_as_dicts(relation_rows)
    for state, group in tuple(epistemic.items()):
        epistemic[state] = EpistemicGroup(
            label=group.label,
            model_instruction=group.model_instruction,
            entities=tuple(row for row in entity_dicts if row["trust_state"] == state),
            relations=tuple(row for row in relation_dicts if row["trust_state"] == state),
        )
    return ResearchContext(
        target_entity=target,
        workstream=dict(workstream) if workstream else None,
        entities=entity_dicts,
        relations=relation_dicts,
        attributes=attributes,
        sources=tuple(source_rows),
        workstream_links=_rows_as_dicts(link_rows),
        selections=selections,
        epistemic=epistemic,
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
