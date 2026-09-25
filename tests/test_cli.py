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


def test_graph_cli_and_deterministic_delta(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["init", "Delta demo"]).exit_code == 0

    result = runner.invoke(app, ["entity", "add", "theorem", "Known result"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["entity", "add", "conjecture", "Possible improvement"])
    assert result.exit_code == 0, result.output

    commands = [
        ["attr", "set", "1", "synchrony", "asynchronous"],
        ["attr", "set", "2", "synchrony", "asynchronous"],
        ["attr", "set", "1", "communication", "O(n^3)"],
        ["attr", "set", "2", "communication", "O(n^2)"],
        ["relation", "add", "2", "EXTENDS", "1"],
        ["workstream", "create", "attack", "Try to refute conjecture #2"],
        ["workstream", "link", "1", "2", "input"],
        ["workstream", "status", "1", "failed", "--summary", "No refutation found."],
    ]
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["delta", "1", "2"])
    assert result.exit_code == 0, result.output
    assert "UNCHANGED" in result.output
    assert "synchrony = asynchronous" in result.output
    assert "CHANGED" in result.output
    assert "A = O(n^3)" in result.output
    assert "B = O(n^2)" in result.output

    result = runner.invoke(app, ["workstream", "show", "1"])
    assert result.exit_code == 0, result.output
    assert "failed" in result.output
    assert "No refutation found." in result.output
