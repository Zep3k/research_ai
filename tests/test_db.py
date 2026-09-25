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

CURRENT_V01_SCHEMA = """
CREATE TABLE projects (
    id INTEGER PRIMARY KEY, name TEXT, description TEXT, created_at TEXT
);
CREATE TABLE ideas (
    id INTEGER PRIMARY KEY, statement TEXT, status TEXT, notes TEXT, created_at TEXT
);
CREATE TABLE papers (
    id INTEGER PRIMARY KEY, title TEXT, local_path TEXT, text_path TEXT,
    sha256 TEXT, page_count INTEGER, added_at TEXT
);
CREATE TABLE runs (
    id INTEGER PRIMARY KEY, idea_id INTEGER, provider TEXT, model TEXT,
    status TEXT, formalization_json TEXT, literature_json TEXT, report_json TEXT,
    error_message TEXT, created_at TEXT, completed_at TEXT
);
CREATE TABLE api_calls (
    id INTEGER PRIMARY KEY, run_id INTEGER, provider TEXT, model TEXT, purpose TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
    estimated_max_cost_usd REAL, status TEXT, error_message TEXT,
    response_text TEXT, created_at TEXT
);
INSERT INTO projects VALUES (1, 'v01', '', '2026-01-01T00:00:00+00:00');
INSERT INTO ideas VALUES (
    7, 'Can the bound improve?', 'exploring', 'Preserve these notes',
    '2026-01-02T00:00:00+00:00'
);
INSERT INTO papers VALUES (
    4, 'A source paper', '.theory/papers/source.pdf', '.theory/papers/source.txt',
    'abc', 12, '2026-01-03T00:00:00+00:00'
);
PRAGMA user_version = 2;
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


def test_current_v01_state_is_backfilled_once_into_graph(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    raw = sqlite3.connect(tmp_path / ".theory" / "research.db")
    raw.executescript(CURRENT_V01_SCHEMA)
    raw.commit()
    raw.close()

    with connect() as con:
        entities = con.execute(
            "SELECT entity_type,title,body,trust_state FROM entities ORDER BY id"
        ).fetchall()
        links = con.execute(
            "SELECT legacy_table,legacy_id,entity_id FROM legacy_entity_links ORDER BY legacy_table"
        ).fetchall()
        api_columns = {row[1] for row in con.execute("PRAGMA table_info(api_calls)")}
        migrations = con.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()

    # Opening again must not duplicate a migrated legacy object.
    with connect() as con:
        count = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    assert [(row["entity_type"], row["title"]) for row in entities] == [
        ("ResearchIdea", "Can the bound improve?"),
        ("Paper", "A source paper"),
    ]
    assert entities[0]["body"] == "Preserve these notes"
    assert all(row["trust_state"] == "unverified" for row in entities)
    assert [(row["legacy_table"], row["legacy_id"]) for row in links] == [
        ("ideas", 7),
        ("papers", 4),
    ]
    assert "workstream_id" in api_columns
    assert [row[0] for row in migrations] == [1, 2, 3]
    assert count == 2
