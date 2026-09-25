from __future__ import annotations
import json
from pathlib import Path
import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from .attack import attack as run_attack
from .config import Config
from .db import connect, initialize, monthly_spend, utcnow
from .errors import TheoryError
from .graph import (
    add_entity,
    add_relation,
    add_review,
    add_source,
    compare_attributes,
    create_workstream,
    link_workstream_entity,
    list_attributes,
    normalize_attribute_key,
    set_attribute,
    set_entity_trust,
    set_workstream_status,
)
from .papers import import_pdf
from .paths import STATE_DIR, PAPERS_DIR, require_workspace
from .trust import EntityType, parse_enum
from .workflows import investigate as run_investigation

app = typer.Typer(no_args_is_help=True, help="Local theoretical-research workbench.")
idea_app = typer.Typer(no_args_is_help=True)
paper_app = typer.Typer(no_args_is_help=True)
run_app = typer.Typer(no_args_is_help=True)
entity_app = typer.Typer(no_args_is_help=True)
attr_app = typer.Typer(no_args_is_help=True)
relation_app = typer.Typer(no_args_is_help=True)
source_app = typer.Typer(no_args_is_help=True)
workstream_app = typer.Typer(no_args_is_help=True)
review_app = typer.Typer(no_args_is_help=True)
app.add_typer(idea_app, name="idea")
app.add_typer(paper_app, name="paper")
app.add_typer(run_app, name="run")
app.add_typer(entity_app, name="entity")
app.add_typer(attr_app, name="attr")
app.add_typer(relation_app, name="relation")
app.add_typer(source_app, name="source")
app.add_typer(workstream_app, name="workstream")
app.add_typer(review_app, name="review")
console = Console()


@app.command()
def init(name: str, monthly_budget: float = typer.Option(100.0, min=1.0)):
    """Create a research workspace in the current directory."""
    if STATE_DIR.exists():
        raise typer.BadParameter(".theory already exists in this directory.")
    name = name.strip()
    if not name:
        raise typer.BadParameter("Project name cannot be empty.")
    cfg = Config(monthly_budget_usd=monthly_budget)
    STATE_DIR.mkdir(parents=True)
    PAPERS_DIR.mkdir(parents=True)
    initialize(name)
    cfg.save()
    console.print(f"[green]Created[/green] '{escape(name)}' with ${monthly_budget:.2f}/month budget.")


@idea_app.command("add")
def idea_add(statement: str, notes: str = ""):
    require_workspace()
    statement = statement.strip()
    if not statement:
        raise typer.BadParameter("Idea statement cannot be empty.")
    with connect() as con:
        cur = con.execute(
            "INSERT INTO ideas(statement,notes,created_at) VALUES(?,?,?)",
            (statement, notes, utcnow()),
        )
        idea_id = int(cur.lastrowid)
        entity_id = add_entity(con, "ResearchIdea", statement, body=notes)
        con.execute(
            "INSERT INTO legacy_entity_links VALUES(1,'ideas',?,?)", (idea_id, entity_id)
        )
    console.print(
        f"Added idea [bold]#{idea_id}[/bold] as entity [bold]#{entity_id}[/bold]: "
        f"{escape(statement)}"
    )


@idea_app.command("list")
def idea_list():
    require_workspace()
    with connect() as con:
        rows = con.execute("SELECT id,statement,status FROM ideas ORDER BY id").fetchall()
    table = Table("ID", "Status", "Idea")
    for r in rows:
        table.add_row(str(r["id"]), Text(r["status"]), Text(r["statement"]))
    console.print(table)


@paper_app.command("add")
def paper_add(path: Path, title: str | None = None):
    require_workspace()
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise typer.BadParameter("Path must be an existing PDF.")
    title = (title or path.stem).strip()
    if not title:
        raise typer.BadParameter("Paper title cannot be empty.")
    with connect() as con:
        cur = con.execute(
            "INSERT INTO papers(title,local_path,added_at) VALUES(?,'',?)",
            (title, utcnow()),
        )
        paper_id = int(cur.lastrowid)
    try:
        imported = import_pdf(path, paper_id)
    except Exception as exc:
        with connect() as con:
            con.execute("DELETE FROM papers WHERE id=?", (paper_id,))
        raise TheoryError(f"Could not import PDF {path}: {exc}") from exc
    with connect() as con:
        con.execute(
            """
            UPDATE papers SET local_path=?,text_path=?,sha256=?,page_count=? WHERE id=?
            """,
            (
                str(imported.pdf_path),
                str(imported.text_path),
                imported.sha256,
                imported.page_count,
                paper_id,
            ),
        )
        entity_id = add_entity(
            con, "Paper", title, body=f"Local PDF: {imported.pdf_path}"
        )
        con.execute(
            "INSERT INTO legacy_entity_links VALUES(1,'papers',?,?)", (paper_id, entity_id)
        )
    console.print(
        f"Added paper [bold]#{paper_id}[/bold] as entity [bold]#{entity_id}[/bold]: "
        f"{escape(title)}"
    )


@paper_app.command("list")
def paper_list():
    require_workspace()
    with connect() as con:
        rows = con.execute("SELECT id,title,local_path FROM papers ORDER BY id").fetchall()
    table = Table("ID", "Title", "Local file")
    for r in rows:
        table.add_row(str(r["id"]), Text(r["title"]), Text(r["local_path"]))
    console.print(table)


@app.command()
def investigate(idea_id: int, provider: str = typer.Option("openai", help="openai or anthropic")):
    require_workspace()
    if provider not in {"openai", "anthropic"}:
        raise typer.BadParameter("provider must be openai or anthropic")
    run_id = run_investigation(idea_id, provider)
    console.print(f"[green]Saved investigation run #{run_id}[/green]")
    _show_run(run_id)


@app.command("attack")
def attack_command(
    workstream_id: int,
    provider: str = typer.Option("openai", help="openai or anthropic"),
):
    """Run one graph-scoped adversarial pass for an active attack workstream."""
    require_workspace()
    if provider not in {"openai", "anthropic"}:
        raise typer.BadParameter("provider must be openai or anthropic")
    outcome = run_attack(workstream_id, provider)
    console.print(
        f"[green]Attack completed[/green]: review #{outcome.review_id} "
        f"({outcome.review_result}), {len(outcome.artifact_ids)} candidate artifact(s)."
    )
    workstream_show(workstream_id)


def _show_run(run_id: int):
    with connect() as con:
        row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        calls = con.execute(
            """
            SELECT COUNT(*) AS n, SUM(cost_usd) AS c,
                   SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed
            FROM api_calls WHERE run_id=?
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        raise typer.BadParameter(f"Run #{run_id} does not exist.")
    console.print(Panel(
        f"{escape(row['provider'])} / {escape(row['model'])} | status: {escape(row['status'])} | "
        f"{calls['n']} model call(s), {calls['failed']} failed | "
        f"estimated cost ${float(calls['c'] or 0):.4f}",
        title=f"Investigation #{run_id}",
    ))
    if row["error_message"]:
        console.print(f"[red]Error:[/red] {escape(row['error_message'])}")

    literature = json.loads(row["literature_json"] or "{}")
    if isinstance(literature, dict):
        sources = literature.get("sources", [])
        failed_searches = [
            item for item in literature.get("searches", []) if item.get("status") == "failed"
        ]
        source_count = len(sources)
        failures = len(failed_searches)
        if source_count or failures:
            console.print(f"Retrieved sources: {source_count}; failed searches: {failures}")
        if sources:
            console.print("\n[bold]Retrieved Literature[/bold]")
            for source in sources:
                year = f" ({source['year']})" if source.get("year") else ""
                locator = source.get("doi") or source.get("url") or source.get("openalex_id", "")
                console.print(
                    f" • [bold]{escape(str(source.get('source_id', '?')))}[/bold] — "
                    f"{escape(str(source.get('title', 'Untitled')))}{escape(year)} "
                    f"[dim]{escape(str(locator))}[/dim]"
                )
        for search in failed_searches:
            console.print(
                f"[yellow]Literature search failed:[/yellow] "
                f"{escape(str(search.get('query', '')))} — {escape(str(search.get('error', '')))}"
            )

    report = json.loads(row["report_json"] or "{}")
    for key, value in report.items():
        console.print(f"\n[bold]{key.replace('_', ' ').title()}[/bold]")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "statement" in item:
                    status = item.get("epistemic_status", "unresolved")
                    sources = ", ".join(item.get("source_ids", []))
                    suffix = f"; sources: {sources}" if sources else ""
                    console.print(
                        f" • {escape(str(item['statement']))} "
                        f"[dim]({escape(str(status))}{escape(suffix)})[/dim]"
                    )
                else:
                    console.print(f" • {escape(str(item))}")
        else:
            console.print(escape(str(value)))


@run_app.command("show")
def run_show(run_id: int):
    require_workspace()
    _show_run(run_id)


@app.command()
def budget():
    require_workspace()
    cfg = Config.load()
    spent = monthly_spend()
    pct = 100 * spent / cfg.monthly_budget_usd if cfg.monthly_budget_usd else 0
    console.print(
        f"Estimated API spend this month: [bold]${spent:.2f}[/bold] / "
        f"${cfg.monthly_budget_usd:.2f} ({pct:.1f}%)"
    )


@entity_app.command("add")
def entity_add(
    entity_type: str,
    title: str,
    body: str = typer.Option("", help="Longer statement or notes."),
    status: str = typer.Option("active"),
    trust_state: str = typer.Option("unverified", "--trust-state"),
    confidence: float = typer.Option(0.0, min=0.0, max=1.0),
    source: list[int] | None = typer.Option(None, "--source", help="Existing source ID."),
):
    """Create a typed research object."""
    require_workspace()
    with connect() as con:
        entity_id = add_entity(
            con,
            entity_type,
            title,
            body=body,
            status=status,
            trust_state=trust_state,
            confidence=confidence,
            source_ids=source or (),
        )
        entity = con.execute("SELECT entity_type FROM entities WHERE id=?", (entity_id,)).fetchone()
    console.print(
        f"Added {escape(entity['entity_type'])} [bold]#{entity_id}[/bold]: {escape(title.strip())}"
    )


@entity_app.command("list")
def entity_list(entity_type: str | None = typer.Option(None, "--type")):
    """List research objects."""
    require_workspace()
    params: tuple[str, ...] = ()
    where = ""
    if entity_type is not None:
        parsed = parse_enum(EntityType, entity_type, "entity type")
        where = "WHERE entity_type=?"
        params = (parsed.value,)
    with connect() as con:
        rows = con.execute(
            f"SELECT id,entity_type,status,trust_state,title FROM entities {where} ORDER BY id",
            params,
        ).fetchall()
    table = Table("ID", "Type", "Status", "Trust", "Title")
    for row in rows:
        table.add_row(
            str(row["id"]),
            Text(row["entity_type"]),
            Text(row["status"]),
            Text(row["trust_state"]),
            Text(row["title"]),
        )
    console.print(table)


@entity_app.command("show")
def entity_show(entity_id: int):
    """Show an entity, its attributes, provenance, and direct relations."""
    require_workspace()
    with connect() as con:
        entity = con.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
        if entity is None:
            raise TheoryError(f"Entity #{entity_id} does not exist.")
        attrs = con.execute(
            "SELECT key,value FROM entity_attributes WHERE entity_id=? ORDER BY key", (entity_id,)
        ).fetchall()
        sources = con.execute(
            """
            SELECT s.* FROM entity_sources es JOIN sources s ON s.id=es.source_id
            WHERE es.entity_id=? ORDER BY s.id
            """,
            (entity_id,),
        ).fetchall()
        relations = con.execute(
            """
            SELECT * FROM relations
            WHERE source_entity_id=? OR target_entity_id=? ORDER BY id
            """,
            (entity_id, entity_id),
        ).fetchall()
    console.print(
        Panel(
            f"{escape(entity['entity_type'])} | status: {escape(entity['status'])} | "
            f"trust: {escape(entity['trust_state'])} | confidence: {entity['confidence']:.2f}\n\n"
            f"{escape(entity['body'])}",
            title=f"Entity #{entity_id}: {escape(entity['title'])}",
        )
    )
    if attrs:
        console.print("[bold]Attributes[/bold]")
        for row in attrs:
            console.print(f"  {escape(row['key'])} = {escape(row['value'])}")
    if sources:
        console.print("[bold]Sources[/bold]")
        for row in sources:
            locator = row["theorem"] or row["section"] or row["external_url"] or ""
            page = f" page {row['page']}" if row["page"] else ""
            console.print(f"  #{row['id']}{page} {escape(locator)}")
    if relations:
        console.print("[bold]Relations[/bold]")
        for row in relations:
            console.print(
                f"  #{row['id']} {row['source_entity_id']} {row['relation_type']} "
                f"{row['target_entity_id']} [{row['trust_state']}]"
            )


@entity_app.command("trust")
def entity_trust(entity_id: int, trust_state: str):
    """Explicitly change an entity's epistemic state."""
    require_workspace()
    with connect() as con:
        set_entity_trust(con, entity_id, trust_state)
    console.print(f"Entity #{entity_id} trust state set to {escape(trust_state)}.")


@attr_app.command("set")
def attr_set(entity_id: int, key: str, value: str):
    """Set a generic structured attribute on an entity."""
    require_workspace()
    with connect() as con:
        set_attribute(con, entity_id, key, value)
    console.print(
        f"Set entity #{entity_id}: {escape(normalize_attribute_key(key))} = {escape(value)}"
    )


@attr_app.command("list")
def attr_list(entity_id: int):
    require_workspace()
    with connect() as con:
        attrs = list_attributes(con, entity_id)
    table = Table("Key", "Value")
    for key, value in attrs.items():
        table.add_row(Text(key), Text(value))
    console.print(table)


@relation_app.command("add")
def relation_add(
    source_entity_id: int,
    relation_type: str,
    target_entity_id: int,
    evidence_source: int | None = typer.Option(None, "--evidence-source"),
    trust_state: str = typer.Option("unverified", "--trust-state"),
    confidence: float = typer.Option(0.0, min=0.0, max=1.0),
):
    """Create a validated directed relation."""
    require_workspace()
    with connect() as con:
        relation_id = add_relation(
            con,
            source_entity_id,
            relation_type,
            target_entity_id,
            evidence_source_id=evidence_source,
            trust_state=trust_state,
            confidence=confidence,
        )
        relation = con.execute(
            "SELECT relation_type FROM relations WHERE id=?", (relation_id,)
        ).fetchone()
    console.print(
        f"Added relation [bold]#{relation_id}[/bold]: {source_entity_id} "
        f"{relation['relation_type']} {target_entity_id}"
    )


@relation_app.command("list")
def relation_list(entity_id: int | None = None):
    require_workspace()
    with connect() as con:
        if entity_id is None:
            rows = con.execute("SELECT * FROM relations ORDER BY id").fetchall()
        else:
            rows = con.execute(
                """
                SELECT * FROM relations
                WHERE source_entity_id=? OR target_entity_id=? ORDER BY id
                """,
                (entity_id, entity_id),
            ).fetchall()
    table = Table("ID", "Source", "Relation", "Target", "Trust", "Evidence")
    for row in rows:
        table.add_row(
            str(row["id"]),
            str(row["source_entity_id"]),
            Text(row["relation_type"]),
            str(row["target_entity_id"]),
            Text(row["trust_state"]),
            str(row["evidence_source_id"] or ""),
        )
    console.print(table)


@source_app.command("add")
def source_add(
    entity_id: int,
    paper: int | None = typer.Option(None, "--paper", help="Paper entity ID."),
    source_type: str = typer.Option("paper_locator", "--type"),
    page: int | None = typer.Option(None, min=1),
    section: str | None = None,
    theorem: str | None = None,
    excerpt: str | None = None,
    external_url: str | None = typer.Option(None, "--external-url"),
):
    """Persist and attach a precise provenance locator."""
    require_workspace()
    with connect() as con:
        source_id = add_source(
            con,
            entity_id,
            paper_entity_id=paper,
            source_type=source_type,
            page=page,
            section=section,
            theorem=theorem,
            excerpt=excerpt,
            external_url=external_url,
        )
    console.print(f"Added source [bold]#{source_id}[/bold] to entity #{entity_id}.")


@source_app.command("list")
def source_list(entity_id: int | None = None):
    require_workspace()
    with connect() as con:
        if entity_id is None:
            rows = con.execute(
                """
                SELECT NULL AS target_entity_id,s.* FROM sources s ORDER BY s.id
                """
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT es.entity_id AS target_entity_id,s.*
                FROM entity_sources es JOIN sources s ON s.id=es.source_id
                WHERE es.entity_id=? ORDER BY s.id
                """,
                (entity_id,),
            ).fetchall()
    table = Table("ID", "Entity", "Paper", "Page", "Section/Theorem", "URL")
    for row in rows:
        table.add_row(
            str(row["id"]),
            str(row["target_entity_id"] or ""),
            str(row["paper_entity_id"] or ""),
            str(row["page"] or ""),
            Text(row["theorem"] or row["section"] or ""),
            Text(row["external_url"] or ""),
        )
    console.print(table)


@workstream_app.command("create")
def workstream_create(
    workstream_type: str,
    goal: str,
    status: str = typer.Option("active"),
    summary: str = "",
):
    """Create a durable focused research effort."""
    require_workspace()
    with connect() as con:
        workstream_id = create_workstream(
            con, workstream_type, goal, status=status, summary=summary
        )
    console.print(f"Created workstream [bold]#{workstream_id}[/bold]: {escape(goal)}")


@workstream_app.command("list")
def workstream_list():
    require_workspace()
    with connect() as con:
        rows = con.execute("SELECT * FROM workstreams ORDER BY id").fetchall()
    table = Table("ID", "Type", "Status", "Goal")
    for row in rows:
        table.add_row(
            str(row["id"]), Text(row["workstream_type"]), Text(row["status"]), Text(row["goal"])
        )
    console.print(table)


@workstream_app.command("show")
def workstream_show(workstream_id: int):
    require_workspace()
    with connect() as con:
        row = con.execute("SELECT * FROM workstreams WHERE id=?", (workstream_id,)).fetchone()
        if row is None:
            raise TheoryError(f"Workstream #{workstream_id} does not exist.")
        entities = con.execute(
            """
            SELECT we.role,e.id,e.entity_type,e.title,e.trust_state
            FROM workstream_entities we JOIN entities e ON e.id=we.entity_id
            WHERE we.workstream_id=? ORDER BY e.id,we.role
            """,
            (workstream_id,),
        ).fetchall()
        reviews = con.execute(
            "SELECT * FROM reviews WHERE workstream_id=? ORDER BY id", (workstream_id,)
        ).fetchall()
        calls = con.execute(
            """
            SELECT provider,model,purpose,status,cost_usd,estimated_max_cost_usd,error_message
            FROM api_calls WHERE workstream_id=? ORDER BY id
            """,
            (workstream_id,),
        ).fetchall()
    console.print(
        Panel(
            f"type: {row['workstream_type']} | status: {row['status']}\n\n"
            f"{escape(row['summary'])}",
            title=f"Workstream #{workstream_id}: {escape(row['goal'])}",
        )
    )
    for entity in entities:
        console.print(
            f"  [{entity['role']}] #{entity['id']} {entity['entity_type']}: "
            f"{escape(entity['title'])} ({entity['trust_state']})"
        )
    for review in reviews:
        console.print(f"  review #{review['id']}: {review['review_type']} — {review['result']}")
        if review["issues"]:
            console.print(f"    {escape(review['issues'])}")
    for call in calls:
        console.print(
            f"  model call: {call['provider']} / {call['model']} | {call['purpose']} | "
            f"{call['status']} | estimated cost ${float(call['cost_usd']):.4f} "
            f"(admission cap ${float(call['estimated_max_cost_usd']):.4f})"
        )
        if call["error_message"]:
            console.print(f"    error: {escape(call['error_message'])}")


@workstream_app.command("link")
def workstream_link(
    workstream_id: int, entity_id: int, role: str = typer.Argument("input")
):
    require_workspace()
    with connect() as con:
        link_workstream_entity(con, workstream_id, entity_id, role)
    console.print(f"Linked entity #{entity_id} to workstream #{workstream_id} as {escape(role)}.")


@workstream_app.command("status")
def workstream_status(
    workstream_id: int,
    status: str,
    summary: str | None = typer.Option(None),
):
    """Set execution lifecycle; scientific outcomes belong in reviews/artifacts."""
    require_workspace()
    with connect() as con:
        set_workstream_status(con, workstream_id, status, summary=summary)
    console.print(f"Workstream #{workstream_id} status set to {escape(status)}.")


@review_app.command("add")
def review_add(
    review_type: str,
    result: str,
    entity: int | None = typer.Option(None, "--entity"),
    workstream: int | None = typer.Option(None, "--workstream"),
    issues: str = "",
    provider: str | None = None,
    model: str | None = None,
    run_id: int | None = typer.Option(None, "--run-id"),
):
    """Store a bounded review result; 'verified' is deliberately unsupported."""
    require_workspace()
    with connect() as con:
        review_id = add_review(
            con,
            review_type,
            result,
            target_entity_id=entity,
            workstream_id=workstream,
            issues=issues,
            provider=provider,
            model=model,
            run_id=run_id,
        )
    console.print(f"Added review [bold]#{review_id}[/bold]: {escape(result)}")


@review_app.command("list")
def review_list():
    require_workspace()
    with connect() as con:
        rows = con.execute("SELECT * FROM reviews ORDER BY id").fetchall()
    table = Table("ID", "Target", "Type", "Result", "Provider / model")
    for row in rows:
        target = (
            f"entity #{row['target_entity_id']}"
            if row["target_entity_id"] is not None
            else f"workstream #{row['workstream_id']}"
        )
        provider_model = " / ".join(x for x in (row["provider"], row["model"]) if x)
        table.add_row(
            str(row["id"]), target, Text(row["review_type"]), Text(row["result"]), provider_model
        )
    console.print(table)


@app.command()
def delta(entity_a: int, entity_b: int):
    """Deterministically compare attributes of two theorem-like entities."""
    require_workspace()
    with connect() as con:
        result = compare_attributes(con, entity_a, entity_b)

    console.print("[bold]UNCHANGED[/bold]")
    for key, value in result.unchanged:
        console.print(f"{escape(key)} = {escape(value)}")
    console.print("\n[bold]CHANGED[/bold]")
    for key, value_a, value_b in result.changed:
        console.print(
            f"{escape(key)}:\n  A = {escape(value_a)}\n  B = {escape(value_b)}"
        )
    console.print("\n[bold]ONLY IN A[/bold]")
    for key, value in result.only_a:
        console.print(f"{escape(key)} = {escape(value)}")
    console.print("\n[bold]ONLY IN B[/bold]")
    for key, value in result.only_b:
        console.print(f"{escape(key)} = {escape(value)}")


def main() -> None:
    try:
        app()
    except TheoryError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}", highlight=False)
        raise SystemExit(1) from exc
