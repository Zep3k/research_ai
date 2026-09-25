import json
from collections.abc import Iterable

from .config import Config
from .db import connect, utcnow
from .errors import ModelOutputError, TheoryError
from .jsonutil import parse_json_model
from .model_calls import budget_guard as _budget_guard
from .model_calls import call_model as _call_model
from .models import (
    Formalization,
    LiteratureBundle,
    LiteratureSearch,
    ResearchReport,
    RetrievedSource,
)
from .openalex import search_works
from .providers import get_model_spec, get_provider

FORMALIZE_MAX_OUTPUT_TOKENS = 8_000
ANALYZE_MAX_OUTPUT_TOKENS = 16_000
REPORT_FINDING_FIELDS = (
    "nearest_results",
    "reasons_to_continue",
    "reasons_to_stop_or_reframe",
    "hidden_assumptions",
    "counterexample_targets",
    "smallest_decisive_subproblems",
    "kill_conditions",
    "next_high_information_actions",
    "epistemic_notes",
)


def _project_context() -> str:
    with connect() as con:
        project = con.execute("SELECT * FROM projects WHERE id=1").fetchone()
        ideas = con.execute(
            "SELECT id,statement,status FROM ideas ORDER BY id DESC LIMIT 12"
        ).fetchall()
        papers = con.execute("SELECT id,title FROM papers ORDER BY id DESC LIMIT 12").fetchall()
    if project is None:
        raise TheoryError("Workspace database has no project record. Reinitialize the workspace.")
    lines = [f"Project: {project['name']}", f"Description: {project['description']}"]
    if ideas:
        lines.append("Recent ideas:")
        lines.extend(f"- #{x['id']} [{x['status']}]: {x['statement']}" for x in ideas)
    if papers:
        lines.append("Locally imported papers (titles only; contents are not evidence in this run):")
        lines.extend(f"- #{x['id']}: {x['title']}" for x in papers)
    return "\n".join(lines)


def _retrieve_literature(queries: Iterable[str]) -> LiteratureBundle:
    searches: list[LiteratureSearch] = []
    unique: dict[str, dict] = {}
    for query in queries:
        try:
            hits = search_works(query, per_page=6)
        except Exception as exc:
            searches.append(
                LiteratureSearch(
                    query=query,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}"[:1000],
                )
            )
            continue

        searches.append(
            LiteratureSearch(
                query=query,
                status="ok",
                openalex_ids=[hit.openalex_id for hit in hits],
            )
        )
        for hit in hits:
            key = (hit.openalex_id or hit.doi or hit.title).casefold()
            if key not in unique:
                unique[key] = {"hit": hit, "queries": []}
            unique[key]["queries"].append(query)

    sources = []
    for index, item in enumerate(list(unique.values())[:16], start=1):
        sources.append(
            RetrievedSource(
                **item["hit"].model_dump(),
                source_id=f"S{index}",
                retrieved_for_queries=list(dict.fromkeys(item["queries"])),
            )
        )
    return LiteratureBundle(sources=sources, searches=searches)


def _validate_report_sources(report: ResearchReport, valid_source_ids: set[str]) -> None:
    unknown: set[str] = set()
    for field_name in REPORT_FINDING_FIELDS:
        for finding in getattr(report, field_name):
            unknown.update(set(finding.source_ids) - valid_source_ids)
    if unknown:
        raise ModelOutputError(
            "Report cited source IDs that were not retrieved: " + ", ".join(sorted(unknown))
        )


def _mark_run_failed(run_id: int, error: Exception) -> None:
    with connect() as con:
        con.execute(
            "UPDATE runs SET status='failed',error_message=?,completed_at=? WHERE id=?",
            (f"{type(error).__name__}: {error}"[:4000], utcnow(), run_id),
        )


def investigate(idea_id: int, provider_name: str) -> int:
    cfg = Config.load()
    models = {"openai": cfg.openai_model, "anthropic": cfg.anthropic_model}
    if provider_name not in models:
        raise TheoryError(f"Unknown provider: {provider_name}")
    model = models[provider_name]
    get_model_spec(model, provider_name)

    with connect() as con:
        idea = con.execute("SELECT * FROM ideas WHERE id=?", (idea_id,)).fetchone()
    if idea is None:
        raise TheoryError(f"Idea #{idea_id} does not exist.")

    prompt1 = f'''You are assisting with theoretical STEM research. Do not praise the idea.
Make it precise enough to investigate and expose ambiguity. Do not make a novelty claim.

PROJECT CONTEXT
{_project_context()}

IDEA
{idea['statement']}

Return ONLY valid JSON with exactly these keys:
{{
  "precise_question": "string",
  "assumptions_to_pin_down": ["..."],
  "search_queries": ["..."],
  "possible_variants": ["..."],
  "immediate_failure_modes": ["..."]
}}
Use 3-6 targeted scholarly search queries, including terminology the researcher may not know.'''
    max_cost1 = _budget_guard(
        cfg,
        model=model,
        prompt=prompt1,
        max_output_tokens=FORMALIZE_MAX_OUTPUT_TOKENS,
        purpose="formalization",
    )
    provider = get_provider(provider_name)
    with connect() as con:
        cur = con.execute(
            """
            INSERT INTO runs(idea_id,provider,model,status,report_json,created_at)
            VALUES(?,?,?,'running','{}',?)
            """,
            (idea_id, provider_name, model, utcnow()),
        )
        run_id = int(cur.lastrowid)

    try:
        r1 = _call_model(
            run_id=run_id,
            provider=provider,
            provider_name=provider_name,
            model=model,
            purpose="formalize",
            prompt=prompt1,
            max_output_tokens=FORMALIZE_MAX_OUTPUT_TOKENS,
            estimated_max_cost_usd=max_cost1,
        )
        formal = parse_json_model(r1.text, Formalization)
        with connect() as con:
            con.execute(
                "UPDATE runs SET formalization_json=? WHERE id=?",
                (formal.model_dump_json(indent=2), run_id),
            )

        literature = _retrieve_literature(formal.search_queries[:6])
        with connect() as con:
            con.execute(
                "UPDATE runs SET literature_json=? WHERE id=?",
                (literature.model_dump_json(indent=2), run_id),
            )
        source_payload = [
            {
                **source.model_dump(exclude={"abstract"}),
                "abstract": (source.abstract or "")[:1800],
            }
            for source in literature.sources
        ]

        finding_shape = (
            '{{"statement":"...","epistemic_status":"sourced|inference|speculation|unresolved",'
            '"source_ids":["S1"]}}'
        )
        prompt2 = f'''You are a skeptical research collaborator for theoretical STEM research.
Do not infer novelty from memory. The OpenAlex records below are discovery metadata and
abstract-level evidence, not primary-source verification. Attack the direction before trying
to rescue it. Optimize for deciding what deserves more researcher time.

QUESTION
{formal.precise_question}

ASSUMPTIONS TO PIN DOWN
{json.dumps(formal.assumptions_to_pin_down)}

IMMEDIATE FAILURE MODES
{json.dumps(formal.immediate_failure_modes)}

OPENALEX SOURCES
{json.dumps(source_payload, indent=2)}

For every finding, use this object shape: {finding_shape}
- "sourced" requires one or more source_ids from the supplied S-identifiers.
- "inference" is a conclusion drawn from supplied material or reasoning.
- "speculation" is a plausible but weakly supported possibility.
- "unresolved" marks a question or claim requiring primary-source/manual checking.
- Never invent a source_id. A source being nearby does not establish its theorem statement.

Return ONLY valid JSON with exactly these keys, with arrays of finding objects:
{{
  "precise_question": "string",
  "nearest_results": [],
  "reasons_to_continue": [],
  "reasons_to_stop_or_reframe": [],
  "hidden_assumptions": [],
  "counterexample_targets": [],
  "smallest_decisive_subproblems": [],
  "kill_conditions": [],
  "next_high_information_actions": [],
  "epistemic_notes": []
}}'''
        max_cost2 = _budget_guard(
            cfg,
            model=model,
            prompt=prompt2,
            max_output_tokens=ANALYZE_MAX_OUTPUT_TOKENS,
            purpose="analysis",
        )
        r2 = _call_model(
            run_id=run_id,
            provider=provider,
            provider_name=provider_name,
            model=model,
            purpose="analyze",
            prompt=prompt2,
            max_output_tokens=ANALYZE_MAX_OUTPUT_TOKENS,
            estimated_max_cost_usd=max_cost2,
        )
        report = parse_json_model(r2.text, ResearchReport)
        _validate_report_sources(report, {source.source_id for source in literature.sources})

        with connect() as con:
            con.execute(
                """
                UPDATE runs
                SET status='completed',report_json=?,error_message=NULL,completed_at=?
                WHERE id=?
                """,
                (report.model_dump_json(indent=2), utcnow(), run_id),
            )
        return run_id
    except Exception as exc:
        _mark_run_failed(run_id, exc)
        raise
