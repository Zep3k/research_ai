"""Persist construction ancestry from controller write receipts.

Route identity is an entity ID, never a model judgment or a timestamp ranking.
The same recorder is used for new steps and migration of historical receipts.
"""
from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .research_context import ResearchContext


ROUTE_IDS = "research_construction_route_ids"
STARTED_AT = "research_construction_started_iteration_id"
SUPERSEDED_BY = "research_construction_superseded_by_route_ids"
CONSTRUCTION_TYPES = {"protocol_component", "lemma", "proof_attempt", "synthesis"}


def _ids(raw: str | None) -> set[int]:
    values = json.loads(raw or "[]")
    return {i for i in values if type(i) is int and i > 0} if isinstance(values, list) else set()


def record_construction_routes(
    con: sqlite3.Connection, iteration: dict, primary_id: int, *,
    reused_obligation_ids: tuple[int, ...] = (),
) -> None:
    """Attach accepted outputs and record an explicit top-level route departure.

    An ordinary step after an attack is not evidence of a distinct construction.
    Frontier/idea selection plus accepted construction is such evidence. A
    continuation inherits its preceding receipt even though its target is still
    the contract input. Reframes retain their independent bypass ownership.
    """
    from .graph import set_attribute

    ws = iteration["workstream_id"]
    linked = {r[0] for r in con.execute(
        "SELECT entity_id FROM workstream_entities WHERE workstream_id=?", (ws,),
    )}
    attrs: dict[int, dict[str, str]] = {}
    for r in con.execute(
        "SELECT a.entity_id,a.key,a.value FROM entity_attributes a WHERE EXISTS "
        "(SELECT 1 FROM workstream_entities l WHERE l.entity_id=a.entity_id AND l.workstream_id=?)", (ws,),
    ):
        attrs.setdefault(r["entity_id"], {})[r["key"]] = r["value"]
    accepted = sorted(_ids(iteration.get("artifact_ids_json")) & linked)
    if not accepted and not reused_obligation_ids:
        return
    target = iteration["target_entity_id"]
    top_level = (iteration["operation"] == "develop" and target == primary_id
                 and iteration.get("focus_obligation_id") is None)
    continuation = (iteration.get("selected_move_id") or "").endswith(":continue")
    prior = con.execute(
        "SELECT * FROM research_iterations WHERE workstream_id=? AND iteration_number<? "
        "AND status='completed' ORDER BY iteration_number DESC LIMIT 1",
        (ws, iteration["iteration_number"]),
    ).fetchone()

    def receipt_roots(row: dict) -> set[int]:
        anchors = _ids(row.get("artifact_ids_json")) | _ids(row.get("consumed_entity_ids_json"))
        anchors.update({row["target_entity_id"], row.get("focus_obligation_id")})
        return set().union(*(_ids(attrs.get(i, {}).get(ROUTE_IDS)) for i in anchors))

    inherited: set[int] = set()
    if top_level and continuation and prior and prior["operation"] == "develop" and prior["target_entity_id"] == target:
        for i in _ids(prior["artifact_ids_json"]):
            inherited.update(_ids(attrs.get(i, {}).get(ROUTE_IDS)))
    elif not top_level:
        anchors = {target, iteration.get("focus_obligation_id")}
        anchors.update(_ids(iteration.get("consumed_entity_ids_json")))
        for i in anchors:
            inherited.update(_ids(attrs.get(i, {}).get(ROUTE_IDS)))

    # These accepted artifact types cannot originate with a terminal branch
    # status. Their current status may have changed since the write receipt.
    construction = [i for i in accepted
                    if attrs.get(i, {}).get("research_artifact_type") in CONSTRUCTION_TYPES]
    roots = inherited
    if top_level and not roots:
        # An obligation-only initial decomposition has ancestry, but cannot
        # supersede an existing substantive construction.
        root = (construction or [i for i in accepted
                                 if attrs.get(i, {}).get("research_artifact_type") == "proof_obligation"])
        if root:
            roots = {root[0]}
            set_attribute(con, root[0], STARTED_AT, str(iteration["id"]))
            explicit_departure = iteration.get("develop_provenance") in {"frontier", "idea"}
            if not continuation and construction and explicit_departure:
                # Follow the controller's persisted execution path. Independent
                # live roots it was not working on are not superseded.
                departed: set[int] = set()
                for previous in con.execute(
                    "SELECT * FROM research_iterations WHERE workstream_id=? AND iteration_number<? "
                    "AND status='completed' ORDER BY iteration_number DESC",
                    (ws, iteration["iteration_number"]),
                ):
                    departed = receipt_roots(dict(previous))
                    if departed:
                        break
                for older in departed:
                    properties = attrs.get(older, {})
                    if (STARTED_AT in properties and int(properties[STARTED_AT]) < iteration["id"]
                            and older not in roots and SUPERSEDED_BY not in properties):
                        set_attribute(con, older, SUPERSEDED_BY, json.dumps(sorted(roots)))
                        set_attribute(con, older, "research_construction_superseded_iteration_id", str(iteration["id"]))

    for i in accepted:
        if ROUTE_IDS in attrs.get(i, {}):
            continue  # Preserve explicit ownership, including global [].
        owners = set(roots)
        for dependency in _ids(attrs.get(i, {}).get("related_entity_ids")):
            owners.update(_ids(attrs.get(dependency, {}).get(ROUTE_IDS)))
        set_attribute(con, i, ROUTE_IDS, json.dumps(sorted(owners)))
    for i in reused_obligation_ids:
        # Explicit route-neutral obligations stay global when reused locally.
        existing = _ids(attrs.get(i, {}).get(ROUTE_IDS))
        if existing:
            set_attribute(con, i, ROUTE_IDS, json.dumps(sorted(existing | roots)))


def backfill_construction_routes(con: sqlite3.Connection) -> None:
    """Reconstruct only controller receipts with an unambiguous contract root."""
    primary_types = {"Conjecture", "Finding", "Lemma", "OpenQuestion", "ProofAttempt",
                     "ResearchIdea", "Technique", "Theorem"}
    for ws in con.execute("SELECT id FROM workstreams WHERE workstream_type='research'").fetchall():
        inputs = [r["entity_id"] for r in con.execute(
            "SELECT l.entity_id,e.entity_type FROM workstream_entities l "
            "JOIN entities e ON e.id=l.entity_id WHERE l.workstream_id=? AND l.role='input'", (ws["id"],),
        ) if r["entity_type"] in primary_types]
        if len(inputs) != 1:
            continue
        for row in con.execute(
            "SELECT * FROM research_iterations WHERE workstream_id=? AND status='completed' "
            "ORDER BY iteration_number", (ws["id"],),
        ).fetchall():
            record_construction_routes(con, dict(row), inputs[0])


def _attribute(context: ResearchContext, entity_id: int, key: str) -> str | None:
    return context.attributes.get(entity_id, {}).get(key)


def _has_terminal_branch_state(context: ResearchContext, entity_id: int) -> bool:
    attrs = context.attributes.get(entity_id, {})
    state = attrs.get("research_branch_status") or attrs.get("develop_branch_status")
    return state in {"blocked", "failed", "refuted"}


def persisted_id_set(raw: object) -> frozenset[int]:
    if not isinstance(raw, str):
        return frozenset()
    try:
        values = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return frozenset()
    if not isinstance(values, list):
        return frozenset()
    try:
        return frozenset(int(value) for value in values)
    except (TypeError, ValueError):
        return frozenset()


def live_construction_route_ids(context: ResearchContext) -> frozenset[int]:
    """Persisted route roots survive until challenged, terminal or superseded.

    An unfinished component or testable/surviving root remains live across
    controller calls. Scheduling order and strategist rankings play no role.
    """
    return frozenset(
        entity_id for entity_id, attrs in context.attributes.items()
        if STARTED_AT in attrs and not persisted_id_set(attrs.get(SUPERSEDED_BY))
        and route_entity_is_live(context, entity_id)
    )


def committed_construction_route_ids(
    context: ResearchContext, workstream_id: int, open_obligation_ids: tuple[int, ...],
) -> frozenset[int]:
    """Live routes owning both persisted construction and an active open premise.

    The caller supplies the graph's active obligations. Neither history nor
    relevance references can create ownership or commitment.
    """
    linked = {int(link["entity_id"]) for link in context.workstream_links
              if int(link["workstream_id"]) == workstream_id}
    construction_owners = frozenset().union(*(
        persisted_id_set(attrs.get(ROUTE_IDS)) for entity_id, attrs in context.attributes.items()
        if entity_id in linked and attrs.get("research_artifact_type") in CONSTRUCTION_TYPES
    ))
    obligation_owners = frozenset().union(*(
        persisted_id_set(context.attributes.get(i, {}).get(ROUTE_IDS))
        for i in open_obligation_ids if i in linked
    ))
    return live_construction_route_ids(context) & construction_owners & obligation_owners


def entity_has_closing_relation(context: ResearchContext, entity_id: int) -> bool:
    """Only directed active edges establish refutation scope, never relevance."""
    return any(
        r["status"] == "active" and (
            (r["relation_type"] in {"CONTRADICTS", "REFUTES", "BLOCKS"}
             and int(r["target_entity_id"]) == entity_id)
            or (r["relation_type"] == "FAILS_AT" and int(r["source_entity_id"]) == entity_id)
        )
        for r in context.relations
    )


def route_entity_is_live(context: ResearchContext, entity_id: int) -> bool:
    entity = next((e for e in context.entities if int(e["id"]) == entity_id), None)
    return bool(
        entity and entity["status"] == "active" and entity["trust_state"] != "contradicted"
        and not _has_terminal_branch_state(context, entity_id)
        and not persisted_id_set(_attribute(context, entity_id, SUPERSEDED_BY))
        and _attribute(context, entity_id, "research_obligation_state") != "blocked"
        and _attribute(context, entity_id, "research_attack_state") != "challenged"
        and not entity_has_closing_relation(context, entity_id)
    )
