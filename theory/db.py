from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from .paths import DB_PATH


SCHEMA_VERSION = 6

# V0.1 tables remain available while the old investigate workflow is retired
# gradually. New research state belongs in the typed graph tables below.
SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'exploring',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    local_path TEXT NOT NULL,
    text_path TEXT,
    sha256 TEXT,
    page_count INTEGER,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idea_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    formalization_json TEXT,
    literature_json TEXT NOT NULL DEFAULT '[]',
    report_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY(idea_id) REFERENCES ideas(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL CHECK (entity_type IN (
        'Paper', 'Theorem', 'Definition', 'Assumption', 'Model', 'Technique',
        'ResearchIdea', 'Conjecture', 'OpenQuestion', 'ProofAttempt', 'Lemma',
        'Counterexample', 'Obstruction', 'FailedApproach', 'Finding'
    )),
    title TEXT NOT NULL CHECK (length(trim(title)) > 0),
    body TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'abandoned', 'resolved')),
    trust_state TEXT NOT NULL DEFAULT 'unverified'
        CHECK (trust_state IN (
            'unverified', 'sourced', 'inferred', 'speculative',
            'contradicted', 'quarantined'
        )),
    confidence REAL NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (id, project_id),
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS workstreams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    workstream_type TEXT NOT NULL
        CHECK (workstream_type IN (
            'literature', 'explore', 'attack', 'develop', 'research', 'proof'
        )),
    goal TEXT NOT NULL CHECK (length(trim(goal)) > 0),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN (
            'active', 'completed', 'blocked', 'abandoned', 'error', 'legacy_failed'
        )),
    summary TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (id, project_id),
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    workstream_id INTEGER,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    purpose TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    estimated_max_cost_usd REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'completed',
    error_message TEXT,
    response_text TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE RESTRICT,
    FOREIGN KEY(workstream_id) REFERENCES workstreams(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    paper_entity_id INTEGER,
    source_type TEXT NOT NULL DEFAULT 'paper_locator'
        CHECK (length(trim(source_type)) > 0),
    page INTEGER CHECK (page IS NULL OR page > 0),
    section TEXT,
    theorem TEXT,
    excerpt TEXT,
    external_url TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (id, project_id),
    CHECK (
        paper_entity_id IS NOT NULL OR page IS NOT NULL OR section IS NOT NULL OR
        theorem IS NOT NULL OR excerpt IS NOT NULL OR external_url IS NOT NULL
    ),
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE RESTRICT,
    FOREIGN KEY(paper_entity_id, project_id)
        REFERENCES entities(id, project_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS entity_sources (
    project_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(entity_id, source_id),
    FOREIGN KEY(entity_id, project_id)
        REFERENCES entities(id, project_id) ON DELETE CASCADE,
    FOREIGN KEY(source_id, project_id)
        REFERENCES sources(id, project_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    source_entity_id INTEGER NOT NULL,
    relation_type TEXT NOT NULL CHECK (relation_type IN (
        'USES', 'DEPENDS_ON', 'EXTENDS', 'IMPROVES', 'CONTRADICTS', 'SUPPORTS',
        'REFUTES', 'BLOCKS', 'ATTEMPTS', 'FAILS_AT', 'SOURCED_FROM'
    )),
    target_entity_id INTEGER NOT NULL,
    evidence_source_id INTEGER,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'retired')),
    trust_state TEXT NOT NULL DEFAULT 'unverified'
        CHECK (trust_state IN (
            'unverified', 'sourced', 'inferred', 'speculative',
            'contradicted', 'quarantined'
        )),
    confidence REAL NOT NULL DEFAULT 0.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    created_at TEXT NOT NULL,
    CHECK (source_entity_id <> target_entity_id),
    CHECK (trust_state <> 'sourced' OR evidence_source_id IS NOT NULL),
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE RESTRICT,
    FOREIGN KEY(source_entity_id, project_id)
        REFERENCES entities(id, project_id) ON DELETE RESTRICT,
    FOREIGN KEY(target_entity_id, project_id)
        REFERENCES entities(id, project_id) ON DELETE RESTRICT,
    FOREIGN KEY(evidence_source_id, project_id)
        REFERENCES sources(id, project_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS entity_attributes (
    project_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    key TEXT NOT NULL CHECK (length(trim(key)) > 0),
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(entity_id, key),
    FOREIGN KEY(entity_id, project_id)
        REFERENCES entities(id, project_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workstream_entities (
    project_id INTEGER NOT NULL,
    workstream_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN (
        'input', 'created', 'modified', 'evidence', 'blocked_by'
    )),
    created_at TEXT NOT NULL,
    PRIMARY KEY(workstream_id, entity_id, role),
    FOREIGN KEY(workstream_id, project_id)
        REFERENCES workstreams(id, project_id) ON DELETE CASCADE,
    FOREIGN KEY(entity_id, project_id)
        REFERENCES entities(id, project_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    target_entity_id INTEGER,
    workstream_id INTEGER,
    review_type TEXT NOT NULL CHECK (review_type IN (
        'literature_check', 'counterexample_attempt', 'independent_proof_attempt',
        'proof_critique', 'source_verification'
    )),
    result TEXT NOT NULL CHECK (result IN (
        'no_flaw_found', 'issue_found', 'inconclusive'
    )),
    issues TEXT NOT NULL DEFAULT '',
    provider TEXT,
    model TEXT,
    run_id INTEGER,
    created_at TEXT NOT NULL,
    CHECK ((target_entity_id IS NULL) <> (workstream_id IS NULL)),
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE RESTRICT,
    FOREIGN KEY(target_entity_id, project_id)
        REFERENCES entities(id, project_id) ON DELETE RESTRICT,
    FOREIGN KEY(workstream_id, project_id)
        REFERENCES workstreams(id, project_id) ON DELETE RESTRICT,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS research_iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    workstream_id INTEGER NOT NULL,
    iteration_number INTEGER NOT NULL CHECK (iteration_number > 0),
    operation TEXT NOT NULL CHECK (operation IN (
        'develop', 'attack', 'synthesize', 'prove'
    )),
    target_entity_id INTEGER NOT NULL,
    rationale TEXT NOT NULL CHECK (length(trim(rationale)) > 0),
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'error')),
    material_progress INTEGER NOT NULL DEFAULT 0
        CHECK (material_progress IN (0, 1)),
    artifact_ids_json TEXT NOT NULL DEFAULT '[]',
    duplicate_count INTEGER NOT NULL DEFAULT 0 CHECK (duplicate_count >= 0),
    attack_outcome TEXT NOT NULL DEFAULT 'not_applicable'
        CHECK (attack_outcome IN (
            'not_applicable', 'critical_issue', 'no_critical_issue', 'inconclusive'
        )),
    stop_reason TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(workstream_id, iteration_number),
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE RESTRICT,
    FOREIGN KEY(workstream_id, project_id)
        REFERENCES workstreams(id, project_id) ON DELETE CASCADE,
    FOREIGN KEY(target_entity_id, project_id)
        REFERENCES entities(id, project_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS legacy_entity_links (
    project_id INTEGER NOT NULL,
    legacy_table TEXT NOT NULL CHECK (legacy_table IN ('ideas', 'papers')),
    legacy_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY(legacy_table, legacy_id),
    UNIQUE(entity_id),
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE RESTRICT,
    FOREIGN KEY(entity_id, project_id)
        REFERENCES entities(id, project_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_idea_id ON runs(idea_id);
CREATE INDEX IF NOT EXISTS idx_api_calls_run_id ON api_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_api_calls_workstream_id ON api_calls(workstream_id);
CREATE INDEX IF NOT EXISTS idx_api_calls_created_at ON api_calls(created_at);
CREATE INDEX IF NOT EXISTS idx_entities_project_type ON entities(project_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_sources_paper ON sources(paper_entity_id);
CREATE INDEX IF NOT EXISTS idx_workstreams_project_status
    ON workstreams(project_id, status);
CREATE INDEX IF NOT EXISTS idx_research_iterations_workstream
    ON research_iterations(workstream_id, iteration_number);

CREATE TRIGGER IF NOT EXISTS api_call_workstream_insert_guard
BEFORE INSERT ON api_calls
WHEN NEW.workstream_id IS NOT NULL
 AND NOT EXISTS (SELECT 1 FROM workstreams WHERE id = NEW.workstream_id)
BEGIN
    SELECT RAISE(ABORT, 'api call references a missing workstream');
END;

CREATE TRIGGER IF NOT EXISTS api_call_workstream_update_guard
BEFORE UPDATE OF workstream_id ON api_calls
WHEN NEW.workstream_id IS NOT NULL
 AND NOT EXISTS (SELECT 1 FROM workstreams WHERE id = NEW.workstream_id)
BEGIN
    SELECT RAISE(ABORT, 'api call references a missing workstream');
END;
"""


# These triggers are installed separately so migration 4 can replace V0.2's
# weaker "any source row is provenance" guards in existing databases.
TRUST_TRIGGERS = """
DROP TRIGGER IF EXISTS entity_sourced_insert_guard;
DROP TRIGGER IF EXISTS entity_sourced_update_guard;
DROP TRIGGER IF EXISTS sourced_entity_last_source_guard;
DROP TRIGGER IF EXISTS sourced_entity_origin_guard;
DROP TRIGGER IF EXISTS sourced_relation_insert_guard;
DROP TRIGGER IF EXISTS sourced_relation_update_guard;
DROP TRIGGER IF EXISTS source_origin_update_guard;

CREATE TRIGGER entity_sourced_insert_guard
BEFORE INSERT ON entities
WHEN NEW.trust_state = 'sourced'
BEGIN
    SELECT RAISE(ABORT, 'sourced entity requires identifiable provenance');
END;

CREATE TRIGGER entity_sourced_update_guard
BEFORE UPDATE OF trust_state ON entities
WHEN NEW.trust_state = 'sourced' AND NOT EXISTS (
    SELECT 1
    FROM entity_sources es
    JOIN sources s ON s.id = es.source_id
    WHERE es.entity_id = NEW.id
      AND es.project_id = NEW.project_id
      AND (
          s.paper_entity_id IS NOT NULL OR
          length(trim(COALESCE(s.external_url, ''))) > 0
      )
)
BEGIN
    SELECT RAISE(ABORT, 'sourced entity requires identifiable provenance');
END;

CREATE TRIGGER sourced_entity_origin_guard
BEFORE DELETE ON entity_sources
WHEN (SELECT trust_state FROM entities WHERE id = OLD.entity_id) = 'sourced'
 AND EXISTS (
    SELECT 1 FROM sources s
    WHERE s.id = OLD.source_id
      AND (
          s.paper_entity_id IS NOT NULL OR
          length(trim(COALESCE(s.external_url, ''))) > 0
      )
 )
 AND NOT EXISTS (
    SELECT 1
    FROM entity_sources es
    JOIN sources s ON s.id = es.source_id
    WHERE es.entity_id = OLD.entity_id
      AND es.source_id <> OLD.source_id
      AND (
          s.paper_entity_id IS NOT NULL OR
          length(trim(COALESCE(s.external_url, ''))) > 0
      )
 )
BEGIN
    SELECT RAISE(ABORT, 'cannot remove the last identifiable source from a sourced entity');
END;

CREATE TRIGGER sourced_relation_insert_guard
BEFORE INSERT ON relations
WHEN NEW.trust_state = 'sourced' AND NOT EXISTS (
    SELECT 1 FROM sources s
    WHERE s.id = NEW.evidence_source_id
      AND s.project_id = NEW.project_id
      AND (
          s.paper_entity_id IS NOT NULL OR
          length(trim(COALESCE(s.external_url, ''))) > 0
      )
)
BEGIN
    SELECT RAISE(ABORT, 'sourced relation requires identifiable provenance');
END;

CREATE TRIGGER sourced_relation_update_guard
BEFORE UPDATE OF trust_state, evidence_source_id ON relations
WHEN NEW.trust_state = 'sourced' AND NOT EXISTS (
    SELECT 1 FROM sources s
    WHERE s.id = NEW.evidence_source_id
      AND s.project_id = NEW.project_id
      AND (
          s.paper_entity_id IS NOT NULL OR
          length(trim(COALESCE(s.external_url, ''))) > 0
      )
)
BEGIN
    SELECT RAISE(ABORT, 'sourced relation requires identifiable provenance');
END;

CREATE TRIGGER source_origin_update_guard
BEFORE UPDATE OF paper_entity_id, external_url ON sources
WHEN NEW.paper_entity_id IS NULL
 AND length(trim(COALESCE(NEW.external_url, ''))) = 0
 AND (
    EXISTS (
        SELECT 1 FROM relations r
        WHERE r.evidence_source_id = OLD.id AND r.trust_state = 'sourced'
    )
    OR EXISTS (
        SELECT 1
        FROM entity_sources es
        JOIN entities e ON e.id = es.entity_id
        WHERE es.source_id = OLD.id
          AND e.trust_state = 'sourced'
          AND NOT EXISTS (
              SELECT 1
              FROM entity_sources other_es
              JOIN sources other_s ON other_s.id = other_es.source_id
              WHERE other_es.entity_id = es.entity_id
                AND other_es.source_id <> OLD.id
                AND (
                    other_s.paper_entity_id IS NOT NULL OR
                    length(trim(COALESCE(other_s.external_url, ''))) > 0
                )
          )
    )
 )
BEGIN
    SELECT RAISE(ABORT, 'cannot remove provenance origin used by sourced state');
END;
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _add_column(con: sqlite3.Connection, table: str, definition: str) -> None:
    if not _table_exists(con, table):
        return
    name = definition.split()[0]
    columns = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    if name not in columns:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def _record_migration(con: sqlite3.Connection, version: int, name: str) -> None:
    con.execute(
        "INSERT OR IGNORE INTO schema_migrations(version,name,applied_at) VALUES(?,?,?)",
        (version, name, utcnow()),
    )


def _migrate_v01_columns(con: sqlite3.Connection) -> None:
    """Normalize both early and current V0.1 databases before graph migration."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            statement TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'exploring',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )
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
    con.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_idea_id ON runs(idea_id);
        CREATE INDEX IF NOT EXISTS idx_api_calls_run_id ON api_calls(run_id);
        CREATE INDEX IF NOT EXISTS idx_api_calls_created_at ON api_calls(created_at);
        """
    )


def _backfill_legacy_entities(con: sqlite3.Connection) -> None:
    for row in con.execute("SELECT * FROM ideas ORDER BY id").fetchall():
        linked = con.execute(
            "SELECT 1 FROM legacy_entity_links WHERE legacy_table='ideas' AND legacy_id=?",
            (row["id"],),
        ).fetchone()
        if linked:
            continue
        status = row["status"] if row["status"] in {"active", "abandoned", "resolved"} else "active"
        cur = con.execute(
            """
            INSERT INTO entities(
                project_id,entity_type,title,body,status,trust_state,confidence,
                created_at,updated_at
            ) VALUES(1,'ResearchIdea',?,?,?,'unverified',0,?,?)
            """,
            (row["statement"], row["notes"] or "", status, row["created_at"], row["created_at"]),
        )
        con.execute(
            "INSERT INTO legacy_entity_links VALUES(1,'ideas',?,?)",
            (row["id"], int(cur.lastrowid)),
        )

    for row in con.execute("SELECT * FROM papers ORDER BY id").fetchall():
        linked = con.execute(
            "SELECT 1 FROM legacy_entity_links WHERE legacy_table='papers' AND legacy_id=?",
            (row["id"],),
        ).fetchone()
        if linked:
            continue
        body = f"Local PDF: {row['local_path']}" if row["local_path"] else ""
        cur = con.execute(
            """
            INSERT INTO entities(
                project_id,entity_type,title,body,status,trust_state,confidence,
                created_at,updated_at
            ) VALUES(1,'Paper',?,?,'active','unverified',0,?,?)
            """,
            (row["title"], body, row["added_at"], row["added_at"]),
        )
        con.execute(
            "INSERT INTO legacy_entity_links VALUES(1,'papers',?,?)",
            (row["id"], int(cur.lastrowid)),
        )


def _migrate_workstreams_v4(con: sqlite3.Connection) -> None:
    """Replace V0.2's ambiguous `failed` lifecycle without guessing its meaning."""
    table_sql_row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='workstreams'"
    ).fetchone()
    if table_sql_row is None or "legacy_failed" in (table_sql_row[0] or ""):
        return

    # SQLite cannot alter a CHECK constraint. Keep child tables in place, rebuild
    # the parent with the same IDs, then re-enable and audit foreign keys.
    con.commit()
    con.execute("PRAGMA foreign_keys = OFF")
    try:
        con.executescript(
            """
            DROP TRIGGER IF EXISTS api_call_workstream_insert_guard;
            DROP TRIGGER IF EXISTS api_call_workstream_update_guard;
            BEGIN IMMEDIATE;
            CREATE TABLE workstreams_v4 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                workstream_type TEXT NOT NULL
                    CHECK (workstream_type IN ('literature', 'explore', 'attack', 'proof')),
                goal TEXT NOT NULL CHECK (length(trim(goal)) > 0),
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN (
                        'active', 'completed', 'blocked', 'abandoned',
                        'error', 'legacy_failed'
                    )),
                summary TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (id, project_id),
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE RESTRICT
            );
            INSERT INTO workstreams_v4(
                id,project_id,workstream_type,goal,status,summary,created_at,updated_at
            )
            SELECT
                id,project_id,workstream_type,goal,
                CASE WHEN status='failed' THEN 'legacy_failed' ELSE status END,
                summary,created_at,updated_at
            FROM workstreams;
            DROP TABLE workstreams;
            ALTER TABLE workstreams_v4 RENAME TO workstreams;
            COMMIT;

            CREATE TRIGGER api_call_workstream_insert_guard
            BEFORE INSERT ON api_calls
            WHEN NEW.workstream_id IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM workstreams WHERE id = NEW.workstream_id)
            BEGIN
                SELECT RAISE(ABORT, 'api call references a missing workstream');
            END;

            CREATE TRIGGER api_call_workstream_update_guard
            BEFORE UPDATE OF workstream_id ON api_calls
            WHEN NEW.workstream_id IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM workstreams WHERE id = NEW.workstream_id)
            BEGIN
                SELECT RAISE(ABORT, 'api call references a missing workstream');
            END;
            """
        )
    except Exception:
        con.rollback()
        raise
    finally:
        con.execute("PRAGMA foreign_keys = ON")

    violations = con.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError("Foreign-key violation after V0.2 workstream migration.")


def _migrate_epistemic_v4(con: sqlite3.Connection) -> None:
    """Downgrade old sourced state that lacks an identifiable document origin."""
    now = utcnow()
    con.execute(
        """
        UPDATE entities
        SET trust_state='unverified',updated_at=?
        WHERE trust_state='sourced'
          AND NOT EXISTS (
              SELECT 1
              FROM entity_sources es
              JOIN sources s ON s.id=es.source_id
              WHERE es.entity_id=entities.id
                AND (
                    s.paper_entity_id IS NOT NULL OR
                    length(trim(COALESCE(s.external_url, ''))) > 0
                )
          )
        """,
        (now,),
    )
    con.execute(
        """
        UPDATE relations
        SET trust_state='unverified'
        WHERE trust_state='sourced'
          AND NOT EXISTS (
              SELECT 1 FROM sources s
              WHERE s.id=relations.evidence_source_id
                AND (
                    s.paper_entity_id IS NOT NULL OR
                    length(trim(COALESCE(s.external_url, ''))) > 0
                )
          )
        """
    )


def _migrate_workstreams_v5(con: sqlite3.Connection) -> None:
    """Add the dedicated develop workstream type without changing existing rows."""
    table_sql_row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='workstreams'"
    ).fetchone()
    if table_sql_row is None or "'develop'" in (table_sql_row[0] or ""):
        return

    # SQLite cannot alter a CHECK constraint. Preserve parent IDs while foreign
    # keys are disabled, then audit every child reference after the rebuild.
    con.commit()
    con.execute("PRAGMA foreign_keys = OFF")
    try:
        con.executescript(
            """
            DROP TRIGGER IF EXISTS api_call_workstream_insert_guard;
            DROP TRIGGER IF EXISTS api_call_workstream_update_guard;
            BEGIN IMMEDIATE;
            CREATE TABLE workstreams_v5 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                workstream_type TEXT NOT NULL
                    CHECK (workstream_type IN (
                        'literature', 'explore', 'attack', 'develop', 'proof'
                    )),
                goal TEXT NOT NULL CHECK (length(trim(goal)) > 0),
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN (
                        'active', 'completed', 'blocked', 'abandoned',
                        'error', 'legacy_failed'
                    )),
                summary TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (id, project_id),
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE RESTRICT
            );
            INSERT INTO workstreams_v5(
                id,project_id,workstream_type,goal,status,summary,created_at,updated_at
            )
            SELECT id,project_id,workstream_type,goal,status,summary,created_at,updated_at
            FROM workstreams;
            DROP TABLE workstreams;
            ALTER TABLE workstreams_v5 RENAME TO workstreams;
            COMMIT;

            CREATE TRIGGER api_call_workstream_insert_guard
            BEFORE INSERT ON api_calls
            WHEN NEW.workstream_id IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM workstreams WHERE id = NEW.workstream_id)
            BEGIN
                SELECT RAISE(ABORT, 'api call references a missing workstream');
            END;

            CREATE TRIGGER api_call_workstream_update_guard
            BEFORE UPDATE OF workstream_id ON api_calls
            WHEN NEW.workstream_id IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM workstreams WHERE id = NEW.workstream_id)
            BEGIN
                SELECT RAISE(ABORT, 'api call references a missing workstream');
            END;
            """
        )
    except Exception:
        con.rollback()
        raise
    finally:
        con.execute("PRAGMA foreign_keys = ON")

    violations = con.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError("Foreign-key violation after develop-workstream migration.")


def _migrate_workstreams_v6(con: sqlite3.Connection) -> None:
    """Add the bounded research-controller workstream type without reinterpreting rows."""
    table_sql_row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='workstreams'"
    ).fetchone()
    if table_sql_row is None or "'research'" in (table_sql_row[0] or ""):
        return

    con.commit()
    con.execute("PRAGMA foreign_keys = OFF")
    try:
        con.executescript(
            """
            DROP TRIGGER IF EXISTS api_call_workstream_insert_guard;
            DROP TRIGGER IF EXISTS api_call_workstream_update_guard;
            BEGIN IMMEDIATE;
            CREATE TABLE workstreams_v6 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                workstream_type TEXT NOT NULL
                    CHECK (workstream_type IN (
                        'literature', 'explore', 'attack', 'develop', 'research', 'proof'
                    )),
                goal TEXT NOT NULL CHECK (length(trim(goal)) > 0),
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN (
                        'active', 'completed', 'blocked', 'abandoned',
                        'error', 'legacy_failed'
                    )),
                summary TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (id, project_id),
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE RESTRICT
            );
            INSERT INTO workstreams_v6(
                id,project_id,workstream_type,goal,status,summary,created_at,updated_at
            )
            SELECT id,project_id,workstream_type,goal,status,summary,created_at,updated_at
            FROM workstreams;
            DROP TABLE workstreams;
            ALTER TABLE workstreams_v6 RENAME TO workstreams;
            COMMIT;

            CREATE TRIGGER api_call_workstream_insert_guard
            BEFORE INSERT ON api_calls
            WHEN NEW.workstream_id IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM workstreams WHERE id = NEW.workstream_id)
            BEGIN
                SELECT RAISE(ABORT, 'api call references a missing workstream');
            END;

            CREATE TRIGGER api_call_workstream_update_guard
            BEFORE UPDATE OF workstream_id ON api_calls
            WHEN NEW.workstream_id IS NOT NULL
             AND NOT EXISTS (SELECT 1 FROM workstreams WHERE id = NEW.workstream_id)
            BEGIN
                SELECT RAISE(ABORT, 'api call references a missing workstream');
            END;
            """
        )
    except Exception:
        con.rollback()
        raise
    finally:
        con.execute("PRAGMA foreign_keys = ON")

    violations = con.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError("Foreign-key violation after research-workstream migration.")


def _migrate_existing(con: sqlite3.Connection) -> None:
    if not _table_exists(con, "projects"):
        return

    version = int(con.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Workspace schema version {version} is newer than supported version {SCHEMA_VERSION}."
        )

    if version < 2:
        _migrate_v01_columns(con)

    needs_graph_backfill = version < 3
    _add_column(con, "api_calls", "workstream_id INTEGER")
    # CREATE IF NOT EXISTS also makes migration repairable after an interrupted DDL step.
    con.executescript(SCHEMA)
    if version < 4:
        _migrate_workstreams_v4(con)
        _migrate_epistemic_v4(con)
    if version < 5:
        _migrate_workstreams_v5(con)
    if version < 6:
        _migrate_workstreams_v6(con)
    con.executescript(TRUST_TRIGGERS)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_api_calls_workstream_id ON api_calls(workstream_id)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_workstreams_project_status "
        "ON workstreams(project_id,status)"
    )
    if needs_graph_backfill:
        _backfill_legacy_entities(con)

    _record_migration(con, 1, "initial_v01")
    _record_migration(con, 2, "harden_v01_calls_and_sources")
    _record_migration(con, 3, "typed_research_graph")
    _record_migration(con, 4, "epistemic_guards_and_workstream_lifecycle")
    _record_migration(con, 5, "develop_workstream_type")
    _record_migration(con, 6, "bounded_research_controller")
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
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def initialize(name: str) -> None:
    with connect() as con:
        con.executescript(SCHEMA)
        con.executescript(TRUST_TRIGGERS)
        con.execute(
            "INSERT OR REPLACE INTO projects(id,name,description,created_at) VALUES(1,?,'',?)",
            (name, utcnow()),
        )
        _record_migration(con, 1, "initial_v01")
        _record_migration(con, 2, "harden_v01_calls_and_sources")
        _record_migration(con, 3, "typed_research_graph")
        _record_migration(con, 4, "epistemic_guards_and_workstream_lifecycle")
        _record_migration(con, 5, "develop_workstream_type")
        _record_migration(con, 6, "bounded_research_controller")
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def monthly_spend() -> float:
    with connect() as con:
        row = con.execute(
            """
            SELECT COALESCE(SUM(cost_usd),0) AS total FROM api_calls
            WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m','now')
            """
        ).fetchone()
        return float(row["total"])
