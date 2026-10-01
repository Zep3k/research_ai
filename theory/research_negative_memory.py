"""Bounded views of persisted failures; no scientific state or model judgments."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .research_routes import ROUTE_IDS, STARTED_AT, live_construction_route_ids, persisted_id_set

if TYPE_CHECKING:
    from .research_context import ResearchContext


MAX_NEGATIVE_MEMORY_LESSONS = 6
NEGATIVE_TYPES = {"Obstruction", "Counterexample", "FailedApproach"}


def select_negative_memory(
    context: ResearchContext, *, workstream_id: int, primary_entity_id: int,
    target_entity_id: int, focus_obligation_id: int | None = None,
    operation: str = "develop",
    consumed_entity_ids: tuple[int, ...] = (), additional_entity_ids: tuple[int, ...] = (),
    continuation_route_ids: tuple[int, ...] = (), history: tuple[dict, ...] = (),
) -> tuple[int, ...]:
    """Rank direct scope, live owners, reframe provenance, then root-escape history.

    Equal priorities use descending entity IDs (persisted creation order).
    Contract relevance alone is not branch scope. A root escape may remember
    live routes and the last route touched by a completed controller receipt.
    """
    inputs = {primary_entity_id} | {
        int(link["entity_id"]) for link in context.workstream_links
        if int(link["workstream_id"]) == workstream_id and link["role"] == "input"
    }
    linked = {int(link["entity_id"]) for link in context.workstream_links
              if int(link["workstream_id"]) == workstream_id}
    owners = {i: persisted_id_set(attrs.get(ROUTE_IDS)) for i, attrs in context.attributes.items()}
    live = live_construction_route_ids(context) & linked
    selected = {target_entity_id, *consumed_entity_ids, *additional_entity_ids}
    if focus_obligation_id is not None:
        selected.add(focus_obligation_id)
    branch = (frozenset().union(*(owners.get(i, frozenset()) for i in selected - inputs))
              | set(continuation_route_ids)) & live
    root_escape = (operation == "develop" and target_entity_id == primary_entity_id
                   and focus_obligation_id is None and not continuation_route_ids)
    if root_escape:
        branch = frozenset()  # Root ideation scaffolding does not turn an escape into local work.
    scope = {target_entity_id, focus_obligation_id} - inputs - {None}
    closure_scope = {target_entity_id, focus_obligation_id} - {None}
    completed = sorted((row for row in history if row.get("status") == "completed"
                        and row.get("workstream_id", workstream_id) == workstream_id),
                       key=lambda row: (row.get("iteration_number", 0), row.get("id", 0)), reverse=True)
    reframe: dict[int, tuple[set[int], frozenset[int]]] = {}
    recent_routes: frozenset[int] = frozenset()
    recent_artifacts: frozenset[int] = frozenset()
    for row in completed:
        artifacts = persisted_id_set(row.get("artifact_ids_json"))
        anchors = {row.get("target_entity_id"), row.get("focus_obligation_id")}
        anchors.update(persisted_id_set(row.get("consumed_entity_ids_json")))
        routes = frozenset().union(*(owners.get(i, frozenset()) for i in anchors | set(artifacts)))
        if not recent_routes and routes:
            recent_routes = routes
        if row is completed[0]:
            recent_artifacts = artifacts
        if row.get("operation") == "reframe":
            for entity_id in artifacts:
                reframe.setdefault(entity_id, (anchors - inputs - {None}, routes))
    if not recent_routes:
        roots = [(int(attrs[STARTED_AT]), i) for i, attrs in context.attributes.items()
                 if STARTED_AT in attrs and i in linked]
        if roots:
            recent_routes = frozenset({max(roots)[1]})
    escape_routes = live | recent_routes
    direct_edges: set[int] = set()
    for relation in context.relations:
        if relation["status"] != "active":
            continue
        source, target = int(relation["source_entity_id"]), int(relation["target_entity_id"])
        if relation["relation_type"] in {"REFUTES", "BLOCKS", "CONTRADICTS"} and target in closure_scope:
            direct_edges.add(source)
        elif relation["relation_type"] == "FAILS_AT" and source in closure_scope:
            direct_edges.add(target)

    ranked: list[tuple[int, int]] = []
    for entity in context.entities:
        entity_id = int(entity["id"])
        if entity["entity_type"] not in NEGATIVE_TYPES and not (
            entity["entity_type"] == "Finding" and entity_id in reframe
        ):
            continue
        attrs = context.attributes.get(entity_id, {})
        references = persisted_id_set(attrs.get("related_entity_ids"))
        for key in ("research_focus_obligation_id", "research_reframe_target_obligation_id"):
            if attrs.get(key):
                references |= frozenset({int(attrs[key])})
        audit_scope, audit_routes = reframe.get(entity_id, (set(), frozenset()))
        if entity_id in direct_edges or references & scope:
            priority = 1
        elif owners.get(entity_id, frozenset()) & branch:
            priority = 2
        elif entity_id in reframe and (audit_scope & scope or audit_routes & branch):
            priority = 3
        elif root_escape and (owners.get(entity_id, frozenset()) & escape_routes
                             or audit_routes & escape_routes or entity_id in recent_artifacts
                             or entity_id in additional_entity_ids):
            priority = 4
        else:
            continue
        ranked.append((priority, entity_id))
    return tuple(entity_id for _, entity_id in sorted(ranked, key=lambda item: (item[0], -item[1]))
                 [:MAX_NEGATIVE_MEMORY_LESSONS])


def negative_memory_section(context: ResearchContext) -> str:
    """Foreground exact records; scoped evidence never becomes a prohibition."""
    ids = getattr(context, "selections", {}).get("negative_memory_ids", ())
    if not ids:
        return ""
    by_id = {int(entity["id"]): entity for entity in context.entities}
    scope_keys = {
        ROUTE_IDS, "related_entity_ids", "research_focus_obligation_id",
        "research_reframe_target_obligation_id", "research_branch_status",
        "research_operation", "research_iteration_id", "research_necessity_audit",
    }
    # Full bodies/audits are verbatim. The normal graph retains all attributes;
    # this foreground section repeats only the persisted scientific scope.
    lessons = [{"entity": by_id[i], "attributes": {
        key: value for key, value in context.attributes.get(i, {}).items() if key in scope_keys
    }} for i in ids]
    return (
        "NEGATIVE RESEARCH MEMORY\n"
        "These are prior failures/constraints relevant to this route, supplied as scoped evidence. "
        "Do not treat them as universal impossibility results. Do not simply recreate a failed "
        "or over-strong mechanism. If a new construction reuses one, explicitly identify what "
        "premise or mechanism has materially changed. Preserve their recorded scope and trust state.\n"
        + json.dumps(lessons, sort_keys=True, ensure_ascii=False)
    )
