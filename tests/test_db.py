import sqlite3

from theory.db import SCHEMA_VERSION, connect


OLD_SCHEMA = """
CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT, description TEXT, created_at TEXT);
CREATE TABLE papers (id INTEGER PRIMARY KEY, title TEXT, local_path TEXT, text_path TEXT, added_at TEXT);
CREATE TABLE runs (
    id INTEGER PRIMARY KEY, idea_id INTEGER, provider TEXT, model TEXT,
    report_json TEXT, created_at TEXT
);
CREATE TABLE api_calls (
    id INTEGER PRIMARY KEY, run_id INTEGER, provider TEXT, model TEXT, purpose TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, created_at TEXT
);
INSERT INTO projects VALUES (1, 'old', '', '2026-01-01T00:00:00+00:00');
INSERT INTO runs VALUES (1, 1, 'openai', 'gpt-5.6-sol', '{}', '2026-01-01T00:00:00+00:00');
"""


def test_existing_v01_database_is_migrated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    raw = sqlite3.connect(tmp_path / ".theory" / "research.db")
    raw.executescript(OLD_SCHEMA)
    raw.commit()
    raw.close()

    with connect() as con:
        run = con.execute("SELECT status,literature_json FROM runs WHERE id=1").fetchone()
        version = con.execute("PRAGMA user_version").fetchone()[0]
        api_columns = {row[1] for row in con.execute("PRAGMA table_info(api_calls)")}

    assert run["status"] == "completed"
    assert run["literature_json"] == "[]"
    assert "response_text" in api_columns
    assert version == SCHEMA_VERSION
