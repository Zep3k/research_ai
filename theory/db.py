from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator
from .paths import DB_PATH

SCHEMA_VERSION = 2

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY CHECK (id = 1), name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT, statement TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'exploring', notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
    local_path TEXT NOT NULL, text_path TEXT, sha256 TEXT,
    page_count INTEGER, added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, idea_id INTEGER NOT NULL,
    provider TEXT NOT NULL, model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    formalization_json TEXT, literature_json TEXT NOT NULL DEFAULT '[]',
    report_json TEXT NOT NULL DEFAULT '{}', error_message TEXT,
    created_at TEXT NOT NULL, completed_at TEXT,
    FOREIGN KEY(idea_id) REFERENCES ideas(id)
);
CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER,
    provider TEXT NOT NULL, model TEXT NOT NULL, purpose TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0, estimated_max_cost_usd REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'completed', error_message TEXT, response_text TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_runs_idea_id ON runs(idea_id);
CREATE INDEX IF NOT EXISTS idx_api_calls_run_id ON api_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_api_calls_created_at ON api_calls(created_at);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _add_column(con: sqlite3.Connection, table: str, definition: str) -> None:
    name = definition.split()[0]
    columns = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    if name not in columns:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def _migrate_existing(con: sqlite3.Connection) -> None:
    if not _table_exists(con, "projects"):
        return

    _add_column(con, "papers", "sha256 TEXT")
    _add_column(con, "papers", "page_count INTEGER")
    _add_column(con, "runs", "status TEXT NOT NULL DEFAULT 'completed'")
    _add_column(con, "runs", "formalization_json TEXT")
    _add_column(con, "runs", "literature_json TEXT NOT NULL DEFAULT '[]'")
    _add_column(con, "runs", "error_message TEXT")
    _add_column(con, "runs", "completed_at TEXT")
    _add_column(con, "api_calls", "estimated_max_cost_usd REAL NOT NULL DEFAULT 0")
    _add_column(con, "api_calls", "status TEXT NOT NULL DEFAULT 'completed'")
    _add_column(con, "api_calls", "error_message TEXT")
    _add_column(con, "api_calls", "response_text TEXT")
    con.executescript("""
        CREATE INDEX IF NOT EXISTS idx_runs_idea_id ON runs(idea_id);
        CREATE INDEX IF NOT EXISTS idx_api_calls_run_id ON api_calls(run_id);
        CREATE INDEX IF NOT EXISTS idx_api_calls_created_at ON api_calls(created_at);
    """)
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA busy_timeout = 5000")
    try:
        _migrate_existing(con)
        yield con
        con.commit()
    finally:
        con.close()


def initialize(name: str) -> None:
    with connect() as con:
        con.executescript(SCHEMA)
        _migrate_existing(con)
        con.execute(
            "INSERT OR REPLACE INTO projects(id,name,description,created_at) VALUES(1,?,'',?)",
            (name, utcnow()),
        )


def monthly_spend() -> float:
    with connect() as con:
        row = con.execute("""
            SELECT COALESCE(SUM(cost_usd),0) AS total FROM api_calls
            WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m','now')
        """).fetchone()
        return float(row["total"])
