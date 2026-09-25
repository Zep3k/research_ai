import json

import pymupdf
from typer.testing import CliRunner

from theory.cli import app
from theory.db import connect, utcnow


def test_local_cli_flow_without_paid_calls(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["init", "Master thesis", "--monthly-budget", "25"])
    assert result.exit_code == 0, result.output
    assert "Created" in result.output

    result = runner.invoke(app, ["idea", "add", "Can this bound be improved?"])
    assert result.exit_code == 0, result.output
    assert "#1" in result.output

    result = runner.invoke(app, ["idea", "list"])
    assert result.exit_code == 0, result.output
    assert "Can this bound be improved?" in result.output

    result = runner.invoke(app, ["budget"])
    assert result.exit_code == 0, result.output
    assert "$0.00" in result.output
    assert "$25.00" in result.output

    pdf_path = tmp_path / "paper.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "A local lemma")
        doc.save(pdf_path)
    result = runner.invoke(app, ["paper", "add", str(pdf_path), "--title", "Local paper"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["paper", "list"])
    assert result.exit_code == 0, result.output
    assert "Local paper" in result.output

    literature = {
        "sources": [
            {
                "source_id": "S1",
                "openalex_id": "https://openalex.org/W1",
                "title": "A Nearby Result",
                "year": 2025,
                "doi": "https://doi.org/10.1/example",
            }
        ],
        "searches": [],
    }
    report = {
        "precise_question": "Can the bound be improved?",
        "nearest_results": [
            {
                "statement": "This paper studies a nearby model.",
                "epistemic_status": "sourced",
                "source_ids": ["S1"],
            }
        ],
    }
    with connect() as con:
        con.execute(
            """
            INSERT INTO runs(
                idea_id,provider,model,status,literature_json,report_json,created_at,completed_at
            ) VALUES(1,'openai','gpt-5.6-sol','completed',?,?,?,?)
            """,
            (json.dumps(literature), json.dumps(report), utcnow(), utcnow()),
        )

    result = runner.invoke(app, ["run", "show", "1"])
    assert result.exit_code == 0, result.output
    assert "Retrieved Literature" in result.output
    assert "A Nearby Result" in result.output
    assert "sourced; sources: S1" in result.output
