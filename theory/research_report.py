"""Deterministic, read-only projection of persisted research state.

No scheduling, reconciliation, migrations, model invocation, or trust promotion.
IDs link every scientific item back to the graph; receipt text is never promoted
into an accepted artifact. Report models contain only frozen models and tuples.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import json
from math import fsum
import re
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .db import SCHEMA_VERSION
from .errors import TheoryError
from .paths import DB_PATH
from .research import (
    INACTIVE_OBLIGATION_STATES, ROOT_TARGET_TYPES, TERMINAL_BRANCH_STATES, _open_obligation_ids,
    _obligation_route_activity, _persisted_open_obligation_ids, _route_entity_is_live,
    bypass_route_is_live, live_construction_route_ids,
)
from .research_routes import ROUTE_IDS
from .research_context import ResearchContext, _build_context


class ReportModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class SourceReport(ReportModel):
    id: int
    paper_entity_id: int | None
    paper_title: str | None
    source_type: str
    page: int | None
    section: str | None
    theorem: str | None
    excerpt: str | None
    external_url: str | None
    identifiable_origin: bool


class EntityReport(ReportModel):
    id: int
    entity_type: str
    title: str
    statement: str
    lifecycle_status: str
    trust_state: str
    link_roles: tuple[str, ...]
    model_artifact: bool
    model_epistemic_status: str | None
    branch_state: str | None
    bounded_attack_state: str | None
    attempted_obligation_ids: tuple[int, ...]
    sources: tuple[SourceReport, ...]


class RouteReport(ReportModel):
    candidate_id: int
    activated_iteration_id: int | None
    replacement_obligation_ids: tuple[int, ...]
    survives_under_recorded_graph: bool


class ReactivationReport(ReportModel):
    timestamp: str | None
    previous_state: str | None
    candidate_ids: tuple[int, ...]


class ObligationReport(ReportModel):
    entity_id: int
    recorded_state: str | None
    open_under_controller_rules: bool
    route_inactive_reason: str | None
    owning_bypass_candidate_ids: tuple[int, ...]
    owning_construction_route_ids: tuple[int, ...]
    live_owning_construction_route_ids: tuple[int, ...]
    parent_obligation_ids: tuple[int, ...]
    necessity_audit_state: str | None
    candidate_ids: tuple[int, ...]
    surviving_candidate_id: int | None
    routes: tuple[RouteReport, ...]
    reactivations: tuple[ReactivationReport, ...]


class ArtifactGroup(ReportModel):
    role: str
    entity_ids: tuple[int, ...]


class ProgressEventReport(ReportModel):
    kind: str
    entity_ids: tuple[int, ...]
    obligation_ids: tuple[int, ...]


class IterationReport(ReportModel):
    id: int
    number: int
    operation: str
    target_entity_id: int
    focus_obligation_id: int | None
    status: str
    rationale: str
    material_progress: bool
    progress_class: str | None
    progress_level: str | None
    resolution_progress: bool | None
    open_obligations_before: int | None
    open_obligations_after: int | None
    artifact_ids: tuple[int, ...]
    consumed_entity_ids: tuple[int, ...]
    events: tuple[ProgressEventReport, ...]
    bounded_attack_outcome: str
    necessity_outcome: str | None
    stop_reason: str | None
    error_message: str | None
    created_at: str
    completed_at: str | None


class ReviewReport(ReportModel):
    id: int
    target_entity_id: int | None
    review_type: str
    result: str
    issues: str
    provider: str | None
    model: str | None
    created_at: str


class UnresolvedItem(ReportModel):
    origin: Literal["graph", "review", "api_call", "workstream"]
    record_id: int
    kind: str
    text: str
    scope: Literal["current_stop", "recorded_unreconciled"]


class TokenTally(ReportModel):
    known_tokens: int
    unknown_calls: int


class UsageReport(ReportModel):
    calls: int
    completed_calls: int
    failed_calls: int
    pending_calls: int
    input_tokens: int
    output_tokens: int
    cache_read: TokenTally
    cache_write: TokenTally
    cache_write_5m: TokenTally
    cache_write_1h: TokenTally
    reasoning: TokenTally
    cost_usd: float
    pending_admission_usd: float
    unmetered_calls: int


class ModelUsage(ReportModel):
    provider: str
    model: str
    usage: UsageReport


class ExecutionReport(ReportModel):
    strategy_calls: int
    execution_calls: int
    other_calls: int
    total: UsageReport
    models: tuple[ModelUsage, ...]


class FrontierReport(ReportModel):
    description: str
    open_obligation_ids: tuple[int, ...]
    pending_attack_ids: tuple[int, ...]
    candidate_ids: tuple[int, ...]
    blocked_branch_ids: tuple[int, ...]


class ResearchReport(ReportModel):
    schema_version: int = 1
    workstream_id: int
    goal: str
    primary_goal_ids: tuple[int, ...]
    lifecycle_status: str
    summary: str
    stop_reason: str | None
    stop_reason_source: str | None
    last_recorded_stop_reason: str | None
    frontier: FrontierReport
    entities: tuple[EntityReport, ...]
    obligations: tuple[ObligationReport, ...]
    artifact_groups: tuple[ArtifactGroup, ...]
    latest_material_progress: IterationReport | None
    recent_iterations: tuple[IterationReport, ...]
    iteration_count: int
    reviews: tuple[ReviewReport, ...]
    unresolved_items: tuple[UnresolvedItem, ...]
    execution: ExecutionReport
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ReportSnapshot:
    context: ResearchContext
    iterations: tuple[dict, ...]
    reviews: tuple[dict, ...]
    calls: tuple[dict, ...]


def _read_snapshot(workstream_id: int) -> ReportSnapshot:
    if not DB_PATH.is_file():
        raise TheoryError('No theory workspace found. Run `theory init "Project name"` first.')
    # Never use db.connect(): even read callers of that helper run migrations.
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA query_only = ON")
        con.execute("BEGIN")  # All tables are read from the same SQLite snapshot.
        version = con.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise TheoryError(
                f"Research report requires schema {SCHEMA_VERSION}; found {version}. "
                "Migrate the workspace separately; report generation never migrates."
            )
        workstream = con.execute("SELECT * FROM workstreams WHERE id=?", (workstream_id,)).fetchone()
        if workstream is None:
            raise TheoryError(f"Workstream #{workstream_id} does not exist.")
        if workstream["workstream_type"] != "research":
            raise TheoryError(f"Workstream #{workstream_id} is not a research workstream.")
        linked = {row[0] for row in con.execute(
            "SELECT entity_id FROM workstream_entities WHERE workstream_id=?", (workstream_id,)
        )}
        context = _build_context(con, linked, workstream=workstream)
        iterations = tuple(dict(row) for row in con.execute(
            "SELECT * FROM research_iterations WHERE workstream_id=? ORDER BY iteration_number,id",
            (workstream_id,),
        ))
        # Include reviews of scoped graph objects, not only workstream-level reviews.
        ids = tuple(entity["id"] for entity in context.entities)
        reviews = tuple(dict(row) for row in con.execute(
            f"SELECT * FROM reviews WHERE workstream_id=? OR target_entity_id IN ({','.join('?' for _ in ids) or 'NULL'}) ORDER BY id",
            (workstream_id, *ids),
        ))
        calls = tuple(dict(row) for row in con.execute(
            "SELECT * FROM api_calls WHERE workstream_id=? ORDER BY id", (workstream_id,),
        ))
        return ReportSnapshot(context, iterations, reviews, calls)
    except sqlite3.Error as exc:
        raise TheoryError(f"Cannot read research report: {exc}") from exc
    finally:
        con.close()


def _json(raw: str | None, default):
    try:
        return json.loads(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _ids(raw) -> tuple[int, ...]:
    values = _json(raw, []) if isinstance(raw, str) else raw
    return tuple(sorted({i for i in values if type(i) is int and i > 0})) if isinstance(values, (list, tuple)) else ()


def _id(raw) -> int | None:
    try:
        value = int(raw)
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def _entities(snapshot: ReportSnapshot) -> tuple[EntityReport, ...]:
    context = snapshot.context
    workstream_id = context.workstream["id"]
    roles: dict[int, set[str]] = defaultdict(set)
    for link in context.workstream_links:
        if link["workstream_id"] == workstream_id:
            roles[link["entity_id"]].add(link["role"])
    attempts: dict[int, set[int]] = defaultdict(set)
    for relation in context.relations:
        if relation["status"] == "active" and relation["relation_type"] == "ATTEMPTS":
            attempts[relation["source_entity_id"]].add(relation["target_entity_id"])
    receipt_artifacts = {i for row in snapshot.iterations for i in _ids(row["artifact_ids_json"])}
    result = []
    for entity in context.entities:
        entity_id = entity["id"]
        attrs = context.attributes.get(entity_id, {})
        attempts[entity_id].update(_ids(attrs.get("addresses_obligation_ids")))
        result.append(EntityReport(
            id=entity_id, entity_type=entity["entity_type"], title=entity["title"],
            statement=attrs.get("research_statement") or entity["body"] or entity["title"],
            lifecycle_status=entity["status"], trust_state=entity["trust_state"],
            link_roles=tuple(sorted(roles[entity_id])),
            model_artifact=bool(entity_id in receipt_artifacts or attrs.get("research_artifact_type")
                                or attrs.get("develop_item_type") or attrs.get("model_epistemic_status")),
            model_epistemic_status=attrs.get("model_epistemic_status"),
            branch_state=attrs.get("research_branch_status") or attrs.get("develop_branch_status"),
            bounded_attack_state=attrs.get("research_attack_state"),
            attempted_obligation_ids=tuple(sorted(attempts[entity_id])),
            sources=tuple(SourceReport(**{key: source.get(key) for key in SourceReport.model_fields})
                          for source in sorted(context.sources, key=lambda source: source["id"]) if source["entity_id"] == entity_id),
        ))
    return tuple(sorted(result, key=lambda entity: entity.id))


def _obligations(context: ResearchContext, entities: tuple[EntityReport, ...], primary_ids: tuple[int, ...]) -> tuple[ObligationReport, ...]:
    primary_id = primary_ids[0] if len(primary_ids) == 1 else -1
    active, owners, parents = _obligation_route_activity(context, context.workstream["id"], primary_id)
    open_ids = set(active)
    persisted_open = set(_persisted_open_obligation_ids(context, context.workstream["id"], primary_id))
    live_roots = live_construction_route_ids(context)
    result = []
    for entity in entities:
        attrs = context.attributes.get(entity.id, {})
        is_obligation = (
            entity.entity_type in {"OpenQuestion", "Obstruction"}
            and (attrs.get("is_proof_obligation", "").casefold() == "true"
                 or (entity.entity_type == "OpenQuestion" and
                     (attrs.get("research_artifact_type") == "proof_obligation"
                      or attrs.get("develop_item_type") == "proof_obligation")))
        )
        if not entity.link_roles or entity.id in primary_ids or not is_obligation:
            continue
        routes = set(_ids(attrs.get("research_bypass_candidate_ids")))
        candidate = _id(attrs.get("research_reframe_candidate_id"))
        if candidate:
            routes.add(candidate)
        route_reports = []
        for route in sorted(routes):
            route_attrs = context.attributes.get(route, {})
            # Use the controller's pure liveness predicate, never its reconciliation writer.
            try:
                live = bypass_route_is_live(context, entity.id, route)
            except (TheoryError, ValueError, TypeError):
                live = False
            route_reports.append(RouteReport(
                candidate_id=route,
                activated_iteration_id=_id(route_attrs.get("research_bypass_activated_iteration_id")),
                replacement_obligation_ids=_ids(route_attrs.get("research_bypass_replacement_obligation_ids")),
                survives_under_recorded_graph=live,
            ))
        log = _json(attrs.get("research_bypass_reactivation_events"), [])
        result.append(ObligationReport(
            entity_id=entity.id, recorded_state=attrs.get("research_obligation_state"),
            open_under_controller_rules=entity.id in open_ids,
            route_inactive_reason=(
                "replacement_not_live" if entity.id in owners and not _route_entity_is_live(context, entity.id)
                else "no_live_owning_construction" if _ids(attrs.get(ROUTE_IDS))
                and not set(_ids(attrs.get(ROUTE_IDS))) & live_roots
                else "no_live_owning_bypass" if entity.id in owners
                else "inactive_parent_obligation"
            ) if entity.id in persisted_open - open_ids else None,
            owning_bypass_candidate_ids=owners.get(entity.id, ()),
            owning_construction_route_ids=_ids(attrs.get(ROUTE_IDS)),
            live_owning_construction_route_ids=tuple(sorted(set(_ids(attrs.get(ROUTE_IDS))) & live_roots)),
            parent_obligation_ids=parents.get(entity.id, ()),
            necessity_audit_state=attrs.get("research_necessity_audit_state"),
            candidate_ids=tuple(e.id for e in entities if entity.id in e.attempted_obligation_ids),
            surviving_candidate_id=_id(attrs.get("research_surviving_candidate_id")),
            routes=tuple(route_reports),
            reactivations=tuple(ReactivationReport(
                timestamp=event.get("timestamp"), previous_state=event.get("previous_state"),
                candidate_ids=_ids(event.get("candidate_ids")),
            ) for event in log if isinstance(event, dict)) if isinstance(log, list) else (),
        ))
    return tuple(result)


GROUP_ORDER = ("problem_contract", "proof_candidates", "mechanisms", "findings", "proof_obligations",
               "negative_results", "open_questions", "source_material", "other_artifacts", "related_context")


def _groups(entities: tuple[EntityReport, ...], obligations: tuple[ObligationReport, ...]) -> tuple[ArtifactGroup, ...]:
    obligation_ids = {o.entity_id for o in obligations}
    grouped: dict[str, list[int]] = defaultdict(list)
    for entity in entities:
        if not entity.link_roles:
            role = "related_context"
        elif "input" in entity.link_roles:
            role = "problem_contract"
        elif entity.id in obligation_ids:
            role = "proof_obligations"
        elif "evidence" in entity.link_roles and "created" not in entity.link_roles:
            role = "source_material"
        else:
            role = {"Theorem": "proof_candidates", "Lemma": "proof_candidates", "ProofAttempt": "proof_candidates",
                    "Conjecture": "proof_candidates", "Technique": "mechanisms", "Finding": "findings",
                    "Counterexample": "negative_results", "Obstruction": "negative_results", "FailedApproach": "negative_results",
                    "OpenQuestion": "open_questions", "Paper": "source_material"}.get(entity.entity_type, "other_artifacts")
        grouped[role].append(entity.id)
    return tuple(ArtifactGroup(role=role, entity_ids=tuple(grouped[role])) for role in GROUP_ORDER if grouped[role])


def _iteration(row: dict) -> IterationReport:
    events = _json(row["progress_events_json"], [])
    return IterationReport(
        id=row["id"], number=row["iteration_number"], operation=row["operation"],
        target_entity_id=row["target_entity_id"], focus_obligation_id=row["focus_obligation_id"],
        status=row["status"], rationale=row["selection_rationale"] or row["rationale"],
        material_progress=bool(row["material_progress"]), progress_class=row["progress_class"],
        progress_level=row["progress_level"], resolution_progress=(bool(row["resolution_progress"]) if row["resolution_progress"] is not None else None),
        open_obligations_before=row["open_obligations_before"], open_obligations_after=row["open_obligations_after"],
        artifact_ids=_ids(row["artifact_ids_json"]), consumed_entity_ids=_ids(row["consumed_entity_ids_json"]),
        events=tuple(ProgressEventReport(kind=event["kind"], entity_ids=_ids(event.get("entity_ids")),
                                         obligation_ids=_ids(event.get("obligation_ids")))
                     for event in events if isinstance(event, dict) and isinstance(event.get("kind"), str)) if isinstance(events, list) else (),
        bounded_attack_outcome=row["attack_outcome"], necessity_outcome=row["necessity_outcome"],
        stop_reason=row["stop_reason"], error_message=row["error_message"],
        created_at=row["created_at"], completed_at=row["completed_at"],
    )


def _usage(calls: tuple[dict, ...]) -> UsageReport:
    def tally(key: str) -> TokenTally:
        return TokenTally(known_tokens=sum(row[key] or 0 for row in calls),
                          unknown_calls=sum(row[key] is None for row in calls))
    return UsageReport(
        calls=len(calls), completed_calls=sum(row["status"] == "completed" for row in calls),
        failed_calls=sum(row["status"] == "failed" for row in calls),
        pending_calls=sum(row["status"] == "started" for row in calls),
        input_tokens=sum(row["input_tokens"] for row in calls), output_tokens=sum(row["output_tokens"] for row in calls),
        cache_read=tally("cache_read_input_tokens"), cache_write=tally("cache_write_input_tokens"),
        cache_write_5m=tally("cache_write_5m_input_tokens"), cache_write_1h=tally("cache_write_1h_input_tokens"),
        reasoning=tally("reasoning_tokens"), cost_usd=fsum(row["cost_usd"] for row in calls),
        pending_admission_usd=fsum(row["estimated_max_cost_usd"] for row in calls if row["status"] == "started"),
        unmetered_calls=sum(row["status"] in {"failed", "started"} and row["uncached_input_tokens"] is None
                            and row["input_tokens"] == row["output_tokens"] == 0 for row in calls),
    )


def _execution(calls: tuple[dict, ...]) -> ExecutionReport:
    strategy = sum(row["purpose"] == "research:strategy" for row in calls)
    execution = sum(row["purpose"] in {f"research:{op}" for op in ("develop", "prove", "attack", "synthesize", "reframe")} for row in calls)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in calls:
        groups[row["provider"], row["model"]].append(row)
    return ExecutionReport(strategy_calls=strategy, execution_calls=execution, other_calls=len(calls)-strategy-execution,
                           total=_usage(calls), models=tuple(ModelUsage(provider=provider, model=model, usage=_usage(tuple(rows)))
                           for (provider, model), rows in sorted(groups.items())))


def _unresolved(snapshot: ReportSnapshot, entities: tuple[EntityReport, ...], stop_reason: str | None) -> tuple[UnresolvedItem, ...]:
    result = []
    if stop_reason == "human_judgment_required":
        result.append(UnresolvedItem(origin="workstream", record_id=snapshot.context.workstream["id"],
                                     kind="human_judgment", text=snapshot.context.workstream["summary"], scope="current_stop"))
    for entity in entities:
        if (entity.entity_type == "OpenQuestion" and entity.lifecycle_status == "active"
                and snapshot.context.attributes.get(entity.id, {}).get("research_obligation_state") not in INACTIVE_OBLIGATION_STATES):
            result.append(UnresolvedItem(origin="graph", record_id=entity.id, kind="open_question",
                                         text=entity.statement, scope="recorded_unreconciled"))
    for review in snapshot.reviews:
        if review["result"] in {"issue_found", "inconclusive"} and review["issues"]:
            result.append(UnresolvedItem(origin="review", record_id=review["id"], kind=review["result"],
                                         text=review["issues"], scope="recorded_unreconciled"))
    for call in snapshot.calls:
        if call["purpose"] in {"research:strategy", "research:ideate"} or not call["purpose"].startswith("research:"):
            continue
        # Receipt text records what a model said, not what survived validation.
        payload = _json(call["response_text"], {})
        if not isinstance(payload, dict):
            continue
        if call["purpose"] == "research:attack" and isinstance(payload.get("report"), dict):
            payload = payload["report"]
        texts = payload.get("could_not_determine", [])
        if isinstance(texts, list):
            for text in texts:
                if isinstance(text, str) and text.strip():
                    result.append(UnresolvedItem(origin="api_call", record_id=call["id"], kind="model_uncertainty",
                                                 text=text, scope="recorded_unreconciled"))
        if payload.get("human_judgment_required") is True and isinstance(payload.get("human_judgment_reason"), str):
            result.append(UnresolvedItem(origin="api_call", record_id=call["id"], kind="model_human_judgment_request",
                                         text=payload["human_judgment_reason"], scope="recorded_unreconciled"))
    return tuple(result)


def build_research_report(workstream_id: int) -> ResearchReport:
    return report_from_snapshot(_read_snapshot(workstream_id))


def report_from_snapshot(snapshot: ReportSnapshot) -> ResearchReport:
    """Pure aggregation, also usable for reconstructed evaluation snapshots."""
    snapshot = replace(snapshot,
                       iterations=tuple(sorted(snapshot.iterations, key=lambda row: (row["iteration_number"], row["id"]))),
                       reviews=tuple(sorted(snapshot.reviews, key=lambda row: row["id"])),
                       calls=tuple(sorted(snapshot.calls, key=lambda row: row["id"])))
    context = snapshot.context
    workstream = context.workstream
    entities = _entities(snapshot)
    primary_ids = tuple(entity.id for entity in entities if "input" in entity.link_roles and entity.entity_type in ROOT_TARGET_TYPES)
    obligations = _obligations(context, entities, primary_ids)
    iterations = tuple(_iteration(row) for row in sorted(snapshot.iterations, key=lambda row: (row["iteration_number"], row["id"])))
    last_stop = next((iteration.stop_reason for iteration in reversed(iterations) if iteration.stop_reason), None)
    stop_reason = None
    stop_source = None
    if workstream["status"] != "active":
        if iterations and iterations[-1].stop_reason:
            stop_reason, stop_source = iterations[-1].stop_reason, f"iteration:{iterations[-1].id}"
        else:
            match = re.match(r"Research controller stopped: ([a-z_]+)\.", workstream["summary"])
            if match:
                stop_reason, stop_source = match.group(1), "workstream_summary"
    closed = tuple(entity.id for entity in entities if entity.link_roles and
                   (entity.branch_state in TERMINAL_BRANCH_STATES
                    or context.attributes.get(entity.id, {}).get("research_obligation_state") == "blocked"
                    or entity.lifecycle_status == "abandoned"
                    or entity.trust_state == "contradicted"))
    candidates = tuple(entity.id for entity in entities if entity.link_roles and entity.lifecycle_status == "active"
                       and entity.id not in closed and entity.bounded_attack_state != "challenged" and
                       (context.attributes.get(entity.id, {}).get("precise_candidate", "").casefold() == "true"
                        or (entity.entity_type in {"Lemma", "Theorem", "ProofAttempt", "Conjecture"}
                            and bool(set(entity.link_roles) & {"input", "created", "modified"}))
                        or entity.branch_state == "promising"))
    pending = tuple(sorted({candidate for obligation in obligations
                            if obligation.open_under_controller_rules
                            and obligation.recorded_state in {"candidate_pending_attack", "reframe_pending_attack"}
                            for candidate in (*obligation.candidate_ids, *(route.candidate_id for route in obligation.routes))}))
    open_ids = tuple(o.entity_id for o in obligations if o.open_under_controller_rules)
    description = ("Human judgment requested by the recorded stop." if stop_reason == "human_judgment_required" else
                   "Independent candidate attacks are pending." if pending else
                   "Open obligations remain; inspect candidates and recorded blockers." if open_ids else
                   "Candidate evidence is available; bounded attacks do not verify proofs." if candidates else
                   "No open proof obligations or active candidates are recorded.")
    notes = [
        "Lifecycle status describes execution, not scientific truth. Persisted metadata is a record, not proof evidence.",
        "Quarantined model artifacts are not assumptions. Sourced means identifiable provenance, never theorem verification.",
        "A surviving candidate or bypass passed only a bounded attack; no proof verification is implied.",
        "Receipt/review uncertainty is unreconciled history; model output is not necessarily an accepted graph artifact.",
        "Token tallies are known subtotals; unknown_calls distinguishes missing telemetry from zero. Costs are recorded estimates.",
    ]
    if any(e.trust_state == "sourced" and not any(source.identifiable_origin for source in e.sources) for e in entities):
        notes.append("A sourced label lacks an identifiable origin in this snapshot; incomplete locators are notes, not source support.")
    if len(primary_ids) != 1:
        notes.append("No unique primary goal input is recorded; use the workstream goal and inspect its contract inputs.")
    if any(o.recorded_state in {"bypassed", "unnecessary"} and not any(route.survives_under_recorded_graph for route in o.routes) for o in obligations):
        notes.append("A recorded bypass has no surviving route in this snapshot; this report does not reactivate obligations.")
    return ResearchReport(
        workstream_id=workstream["id"], goal=workstream["goal"], primary_goal_ids=primary_ids,
        lifecycle_status=workstream["status"], summary=workstream["summary"], stop_reason=stop_reason,
        stop_reason_source=stop_source, last_recorded_stop_reason=last_stop,
        frontier=FrontierReport(description=description, open_obligation_ids=open_ids, pending_attack_ids=pending,
                                candidate_ids=candidates, blocked_branch_ids=closed),
        entities=entities, obligations=obligations, artifact_groups=_groups(entities, obligations),
        latest_material_progress=next((i for i in reversed(iterations) if i.status == "completed" and i.material_progress), None),
        recent_iterations=tuple(reversed(iterations[-8:])), iteration_count=len(iterations),
        reviews=tuple(ReviewReport(**{key: row[key] for key in ReviewReport.model_fields}) for row in snapshot.reviews),
        unresolved_items=_unresolved(snapshot, entities, stop_reason), execution=_execution(snapshot.calls), notes=tuple(notes),
    )
