from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .db import connect
from .errors import TheoryError
from .research_negative_memory import select_negative_memory
from .research_routes import ROUTE_IDS, live_construction_route_ids, persisted_id_set, route_entity_is_live
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
                            if key not in {"research_idea_origin", "research_develop_provenance"}}
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
    operation: str | None = None,
    continuation_route_ids: tuple[int, ...] = (),
    forward_dependencies_only: bool = False,
    history: tuple[dict, ...] = (),
) -> ResearchContext:
    """Select exact graph objects by persisted dependencies, never text ranking.

    Inputs are mandatory text, not expansion anchors. Continuation and synthesis
    traverse only forward dependencies; local proof work also receives direct
    evidence and counterevidence. Omission does not change scientific state.
    """
    inputs = {primary_entity_id} | {
        int(link["entity_id"]) for link in full_context.workstream_links
        if int(link["workstream_id"]) == workstream_id and link["role"] == "input"
    }
    selected = {target_entity_id, *consumed_entity_ids, *additional_entity_ids}
    if focus_obligation_id is not None:
        selected.add(focus_obligation_id)
    mandatory = selected | inputs
    available = {int(entity["id"]) for entity in full_context.entities}
    missing = mandatory - available
    if missing:
        raise TheoryError(f"Mandatory research context entities are absent: {sorted(missing)}")

    live_routes = live_construction_route_ids(full_context)
    owners = {i: persisted_id_set(attrs.get(ROUTE_IDS))
              for i, attrs in full_context.attributes.items()}
    # Explicitly selected historical work retains its own dependency ancestry.
    selected_routes = frozenset().union(*(
        owners.get(i, frozenset()) for i in selected - inputs
        if not owners.get(i, frozenset()) & live_routes
    ))

    def route_allowed(entity_id: int) -> bool:
        roots = owners.get(entity_id, frozenset())
        return not roots or bool(roots & live_routes)

    continuation_artifacts: tuple[int, ...] = ()
    if continuation_route_ids:
        linked = {int(link["entity_id"]) for link in full_context.workstream_links
                  if int(link["workstream_id"]) == workstream_id and link["role"] != "input"}
        current_routes = live_routes & set(continuation_route_ids)
        continuation_artifacts = tuple(sorted((
            i for i in linked - inputs
            if owners.get(i, frozenset()) & current_routes and route_entity_is_live(full_context, i)
        ), reverse=True)[:4])
    anchors = (selected - inputs) | set(continuation_artifacts)
    included = mandatory | set(continuation_artifacts)
    active_relations = tuple(
        relation for relation in full_context.relations if relation["status"] == "active"
    )
    references_by_id: dict[int, set[int]] = {}
    for entity_id, attrs in full_context.attributes.items():
        references: set[int] = set()
        for key in ("related_entity_ids", "addresses_obligation_ids", "research_related_obligation_ids",
                    "research_bypass_replacement_obligation_ids"):
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
    forward_only = (forward_dependencies_only or bool(continuation_route_ids)
                    or operation in {"synthesize", "attack"} or bool(consumed_entity_ids))
    if not forward_only:
        # Direct evidence around a non-contract target/focus is available for
        # testing. Reverse neighborhoods are never recursively expanded.
        for entity_id, references in references_by_id.items():
            if references & anchors and route_allowed(entity_id):
                included.add(entity_id)
        for relation in active_relations:
            source, target = int(relation["source_entity_id"]), int(relation["target_entity_id"])
            if relation["relation_type"] in {"DEPENDS_ON", "USES"}:
                continue  # Forward dependency closure below handles these.
            endpoints = {source, target}
            if endpoints & anchors:
                included.update(i for i in endpoints if route_allowed(i))

    # Follow forward references from anchors and direct evidence, stopping at
    # contract inputs. Finite visited sets preserve full chains and handle cycles.
    explicit_dependencies: dict[int, set[int]] = {}
    for relation in active_relations:
        if relation["relation_type"] in {"DEPENDS_ON", "USES"}:
            source, target = int(relation["source_entity_id"]), int(relation["target_entity_id"])
            explicit_dependencies.setdefault(source, set()).add(target)
            references_by_id.setdefault(source, set()).add(target)

    def expand_dependencies(seeds: set[int], *, explicit_only: bool = False) -> set[int]:
        pending = sorted(seeds - inputs)
        visited: set[int] = set()
        while pending:
            entity_id = pending.pop()
            if entity_id in visited or entity_id not in available:
                continue
            visited.add(entity_id)
            references = (explicit_dependencies if explicit_only else references_by_id).get(entity_id, set())
            dependencies = {i for i in references & available
                            if i in mandatory or route_allowed(i)
                            or owners.get(i, frozenset()) & selected_routes
                            # A directed premise remains available for an attack
                            # even when that cited premise is historical/refuted.
                            or operation == "attack" and i in explicit_dependencies.get(entity_id, set())}
            included.update(dependencies)
            pending.extend(sorted(dependencies - visited - inputs))
        return visited

    if operation == "attack":
        negative_ids = {int(e["id"]) for e in full_context.entities
                        if e["entity_type"] in {"Obstruction", "Counterexample", "FailedApproach"}
                        and e["status"] == "active"}
        # An input target still cites directed premises. Keep those premises
        # without turning the contract into a generic neighborhood anchor.
        if target_entity_id in inputs:
            anchors.update(explicit_dependencies.get(target_entity_id, set()) & available)
            included.update(anchors)
        # Owning roots are structural context, not sibling-expansion anchors.
        route_roots = live_routes & frozenset().union(*(owners.get(i, frozenset()) for i in anchors))
        included.update(route_roots)
        premises = set(anchors)
        for relation in active_relations:
            source, target = int(relation["source_entity_id"]), int(relation["target_entity_id"])
            if (relation["relation_type"] == "SUPPORTS" and target in anchors
                    and source not in negative_ids and route_allowed(source)
                    and route_entity_is_live(full_context, source)):
                included.add(source)
                premises.add(source)
        premises.update(expand_dependencies(premises))
        seen_evidence: set[int] = set()
        while True:
            historical_premises = {i for i in premises - inputs
                                   if owners.get(i, frozenset()) and not route_allowed(i)}
            evidence = {i for i in negative_ids if (
                i in mandatory or route_allowed(i) or owners.get(i, frozenset()) & selected_routes
                or references_by_id.get(i, set()) & historical_premises
            ) and references_by_id.get(i, set()) & (premises - inputs)}
            for relation in active_relations:
                source, target = int(relation["source_entity_id"]), int(relation["target_entity_id"])
                negative = (source if relation["relation_type"] in {"REFUTES", "BLOCKS", "CONTRADICTS"}
                            and target in premises - inputs else target
                            if relation["relation_type"] == "FAILS_AT" and source in premises - inputs else None)
                if negative in negative_ids and (
                    negative in mandatory or route_allowed(negative)
                    or owners.get(negative, frozenset()) & selected_routes
                    or (source if negative == target else target) in historical_premises
                ):
                    evidence.add(negative)
            new_evidence = evidence - seen_evidence
            if not new_evidence:
                break
            seen_evidence.update(new_evidence)
            included.update(new_evidence)
            # Relevance references on failure evidence must not pull siblings.
            expanded = expand_dependencies(new_evidence, explicit_only=True)
            premises.update(expanded - new_evidence)
    else:
        expand_dependencies(included)

    memory_enabled = operation in {"develop", "synthesize", "prove", "reframe"}
    negative_memory_ids = select_negative_memory(
        full_context, workstream_id=workstream_id, primary_entity_id=primary_entity_id,
        target_entity_id=target_entity_id, focus_obligation_id=focus_obligation_id,
        operation=operation,
        consumed_entity_ids=consumed_entity_ids, additional_entity_ids=additional_entity_ids,
        continuation_route_ids=continuation_route_ids, history=history,
    ) if memory_enabled else ()
    # Lessons are exact records, not graph-expansion seeds for historical branches.
    included.update(negative_memory_ids)
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
        selections={**{key: tuple(entity_id for entity_id in ids if entity_id in included)
                       for key, ids in full_context.selections.items()},
                    **({"negative_memory_ids": negative_memory_ids} if memory_enabled else {})},
        epistemic=epistemic,
        context_scope={
            "mode": "focused_research_operation",
            "primary_entity_id": primary_entity_id,
            "target_entity_id": target_entity_id,
            "focus_obligation_id": focus_obligation_id,
            "consumed_entity_ids": list(consumed_entity_ids),
            "expansion_anchor_ids": sorted(anchors),
            "continuation_artifact_ids": list(continuation_artifacts),
            "included_entity_ids": [int(entity["id"]) for entity in entities],
            "full_workstream_entity_count": len(full_context.entities),
            "focused_entity_count": len(entities),
            **({"negative_memory_ids": list(negative_memory_ids)} if memory_enabled else {}),
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
