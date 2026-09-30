"""Rich presentation of the shared immutable research report contract."""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .research_report import EntityReport, ResearchReport, TokenTally


def _short(value: str, limit: int = 160) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit - 1] + "…"


def _ids(values: tuple[int, ...]) -> str:
    return ", ".join(f"#{value}" for value in values) or "—"


def _trust(entity: EntityReport) -> str:
    if entity.trust_state == "quarantined":
        return "quarantined model artifact" if entity.model_artifact else "quarantined"
    if entity.trust_state == "sourced":
        return "source-backed; not verified" if any(s.identifiable_origin for s in entity.sources) else "sourced label; origin missing"
    return entity.trust_state


def _tokens(tally: TokenTally) -> str:
    suffix = f" + unknown ({tally.unknown_calls} calls)" if tally.unknown_calls else ""
    return f"{tally.known_tokens:,}{suffix}"


def _table(console: Console, title: str, columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    table = Table(title=title, title_justify="left", expand=True, show_lines=False)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*(Text(value) for value in row))
    console.print(table)


def render_research_report(report: ResearchReport, console: Console) -> None:
    entities = {entity.id: entity for entity in report.entities}
    obligation_states = {o.entity_id: o.recorded_state for o in report.obligations}
    frontier = report.frontier
    usage = report.execution.total
    overview = Text()
    overview.append(report.goal + "\n", style="bold")
    if len(report.primary_goal_ids) == 1:
        overview.append(f"Research object: {entities[report.primary_goal_ids[0]].title}\n")
    overview.append(f"Lifecycle: {report.lifecycle_status}  |  Stop: {report.stop_reason or 'not recorded'}\n")
    overview.append(f"Frontier: {len(frontier.open_obligation_ids)} active open obligations · "
                    f"{len(frontier.candidate_ids)} active candidates · {len(frontier.blocked_branch_ids)} closed branches\n")
    overview.append(frontier.description + "\n")
    overview.append(f"Calls: {report.execution.strategy_calls} strategy / {report.execution.execution_calls} execution"
                    f" / {report.execution.other_calls} other  |  Recorded cost: ${usage.cost_usd:.6f}")
    console.print(Panel(overview, title=f"Research #{report.workstream_id}", border_style="cyan"))
    console.print(Text("Execution status is not scientific truth. Sourced ≠ verified; bounded attacks are not proof verification.", style="dim"))
    if report.summary:
        console.print(Text("Recorded summary: " + _short(report.summary, 300)))

    _table(console, "Proof obligations", ("Object", "Recorded state", "Candidates / route"), [
        (f"#{o.entity_id} {_short(entities[o.entity_id].title, 90)}",
         f"{o.recorded_state or 'open (implicit)'}" +
         (f"; audit: {o.necessity_audit_state}" if o.necessity_audit_state else "") +
         (f"; route inactive: owners {_ids(o.owning_bypass_candidate_ids)}"
          if o.route_inactive_reason == "no_live_owning_bypass" else
          f"; route inactive: constructions {_ids(o.owning_construction_route_ids)}"
          if o.route_inactive_reason == "no_live_owning_construction" else
          "; route inactive: replacement is terminal or challenged"
          if o.route_inactive_reason == "replacement_not_live" else
          f"; route inactive: parents {_ids(o.parent_obligation_ids)}"
          if o.route_inactive_reason == "inactive_parent_obligation" else ""),
         f"attempts: {_ids(o.candidate_ids)}" +
         (f"; survivor: #{o.surviving_candidate_id}" if o.surviving_candidate_id else "") +
         ("; reactivated" if o.reactivations else "")) for o in report.obligations
    ] or [("None recorded", "—", "—")])

    if any(o.routes or o.reactivations for o in report.obligations):
        _table(console, "Reframes and reversible bypasses", ("Obligation", "Route", "Replacement premises", "Recorded graph"), [
            (f"#{o.entity_id}", f"#{route.candidate_id}", _ids(route.replacement_obligation_ids),
             "surviving bounded route" if route.survives_under_recorded_graph else
             "not activated" if route.activated_iteration_id is None else "no surviving route")
            for o in report.obligations for route in o.routes
        ])
        for obligation in report.obligations:
            for event in obligation.reactivations:
                console.print(Text(f"#{obligation.entity_id} reactivated from {event.previous_state or 'unknown'}; "
                                   f"previous routes {_ids(event.candidate_ids)} ({event.timestamp or 'time unrecorded'})."))

    if frontier.candidate_ids:
        _table(console, "Active candidates — provisional", ("Object", "Attempts", "Trust / bounded attack"), [
            (f"#{i} {_short(entities[i].title, 100)}", _ids(entities[i].attempted_obligation_ids),
             f"{_trust(entities[i])}; {entities[i].bounded_attack_state or 'attack not recorded'}")
            for i in frontier.candidate_ids
        ])
    if frontier.blocked_branch_ids:
        _table(console, "Blocked / failed / refuted branches", ("Object", "Recorded state", "Trust"), [
            (f"#{i} {_short(entities[i].title, 110)}", entities[i].branch_state or obligation_states.get(i) or
             ("contradicted" if entities[i].trust_state == "contradicted" else entities[i].lifecycle_status),
             _trust(entities[i])) for i in frontier.blocked_branch_ids
        ])

    latest = report.latest_material_progress
    console.print(Text("Latest material progress: " + (
        f"iteration {latest.number}: {latest.operation} — {latest.progress_level or 'legacy material flag'}"
        f" / {latest.progress_class or 'event not recorded'}; artifacts {_ids(latest.artifact_ids)}"
        if latest else "none recorded"), style="bold"))
    if report.recent_iterations:
        _table(console, "Recent iterations (newest first)", ("Operation", "Result", "Progress / evidence"), [
            (f"iteration {i.number}: {i.operation} → #{i.target_entity_id}",
             i.status + (f"; stop: {i.stop_reason}" if i.stop_reason else ""),
             f"{i.progress_level or ('legacy material flag' if i.material_progress else 'not recorded')}"
             + (f" / {i.progress_class}" if i.progress_class else "")
             + (f"; bounded attack: {i.bounded_attack_outcome}" if i.bounded_attack_outcome != "not_applicable" else "")
             + (f"; audit: {i.necessity_outcome}" if i.necessity_outcome not in {None, "not_applicable"} else "")
             + (f"; error: {_short(i.error_message)}" if i.error_message else ""))
            for i in report.recent_iterations
        ])
        if report.iteration_count > len(report.recent_iterations):
            console.print(Text(f"Showing {len(report.recent_iterations)} of {report.iteration_count} iterations."))

    if report.unresolved_items:
        _table(console, "Recorded uncertainty / human judgment", ("Origin", "Scope", "Item"), [
            (f"{item.origin} #{item.record_id}: {item.kind}", item.scope.replace("_", " "), _short(item.text, 220))
            for item in report.unresolved_items
        ])
    if report.reviews:
        _table(console, "Recorded reviews — not verification", ("Review", "Result", "Issues"), [
            (f"#{review.id} {review.review_type}", review.result, _short(review.issues)) for review in report.reviews
        ])

    if report.artifact_groups:
        _table(console, "Scientific artifacts by role", ("Role", "Object", "Scientific content", "Trust"), [
            (group.role.replace("_", " ") if index == 0 else "",
             f"#{i} {entities[i].entity_type}", _short(entities[i].statement, 180), _trust(entities[i]))
            for group in report.artifact_groups for index, i in enumerate(group.entity_ids)
        ])
    sources = [(entity.id, source) for entity in report.entities for source in entity.sources]
    if sources:
        _table(console, "Source provenance", ("Object / source", "Origin", "Locator"), [
            (f"#{entity_id} / source #{source.id}", source.paper_title or source.external_url or "note; origin missing",
             "; ".join(value for value in (
                 f"page {source.page}" if source.page else "", source.section or "", source.theorem or "",
             ) if value) or "locator not recorded") for entity_id, source in sources
        ])
    _table(console, "Execution accounting", ("Provider / model", "Calls", "Input / output", "Cache read / write", "Recorded cost"), [
        (f"{model.provider} / {model.model}", str(model.usage.calls),
         f"{model.usage.input_tokens:,} / {model.usage.output_tokens:,}",
         f"{_tokens(model.usage.cache_read)} / {_tokens(model.usage.cache_write)}", f"${model.usage.cost_usd:.6f}")
        for model in report.execution.models
    ] or [("No model calls", "0", "0 / 0", "0 / 0", "$0.000000")])
    console.print(Text(f"Receipts: {usage.completed_calls} completed, {usage.failed_calls} failed, {usage.pending_calls} pending; "
                       f"{usage.unmetered_calls} unmetered. Cache writes: 5m {_tokens(usage.cache_write_5m)}; "
                       f"1h {_tokens(usage.cache_write_1h)}. Pending admission: ${usage.pending_admission_usd:.6f}.", style="dim"))
    for note in report.notes:
        console.print(Text(note, style="dim"))
