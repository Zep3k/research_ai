from __future__ import annotations
import json
from pathlib import Path
import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from .config import Config
from .db import connect, initialize, monthly_spend, utcnow
from .errors import TheoryError
from .papers import import_pdf
from .paths import STATE_DIR, PAPERS_DIR, require_workspace
from .workflows import investigate as run_investigation

app = typer.Typer(no_args_is_help=True, help="Local theoretical-research workbench.")
idea_app = typer.Typer(no_args_is_help=True)
paper_app = typer.Typer(no_args_is_help=True)
run_app = typer.Typer(no_args_is_help=True)
app.add_typer(idea_app, name="idea")
app.add_typer(paper_app, name="paper")
app.add_typer(run_app, name="run")
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
    console.print(f"Added idea [bold]#{idea_id}[/bold]: {escape(statement)}")


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
    console.print(f"Added paper [bold]#{paper_id}[/bold]: {escape(title)}")


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


def main() -> None:
    try:
        app()
    except TheoryError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}", highlight=False)
        raise SystemExit(1) from exc
