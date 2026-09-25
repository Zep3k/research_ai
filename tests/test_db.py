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

V02_SCHEMA_WITH_AMBIGUOUS_STATE = """
CREATE TABLE projects (
    id INTEGER PRIMARY KEY CHECK (id = 1), name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL, trust_state TEXT NOT NULL, confidence REAL NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(id,project_id),
    FOREIGN KEY(project_id) REFERENCES projects(id)
);
CREATE TABLE workstreams (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL,
    workstream_type TEXT NOT NULL, goal TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active','completed','failed','abandoned','blocked')),
    summary TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE(id,project_id), FOREIGN KEY(project_id) REFERENCES projects(id)
);
CREATE TABLE sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL,
    paper_entity_id INTEGER, source_type TEXT NOT NULL, page INTEGER,
    section TEXT, theorem TEXT, excerpt TEXT, external_url TEXT, created_at TEXT NOT NULL,
    UNIQUE(id,project_id)
);
CREATE TABLE entity_sources (
    project_id INTEGER NOT NULL, entity_id INTEGER NOT NULL, source_id INTEGER NOT NULL,
    created_at TEXT NOT NULL, PRIMARY KEY(entity_id,source_id)
);
CREATE TABLE relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL,
    source_entity_id INTEGER NOT NULL, relation_type TEXT NOT NULL,
    target_entity_id INTEGER NOT NULL, evidence_source_id INTEGER,
    status TEXT NOT NULL, trust_state TEXT NOT NULL, confidence REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE workstream_entities (
    project_id INTEGER NOT NULL, workstream_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL, role TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY(workstream_id,entity_id,role),
    FOREIGN KEY(workstream_id,project_id) REFERENCES workstreams(id,project_id),
    FOREIGN KEY(entity_id,project_id) REFERENCES entities(id,project_id)
);
INSERT INTO projects VALUES(1,'v02','','2026-01-01T00:00:00+00:00');
INSERT INTO entities VALUES
    (1,1,'Conjecture','Old sourced claim','','active','sourced',0.5,'t','t'),
    (2,1,'Theorem','Related theorem','','active','unverified',0.0,'t','t');
INSERT INTO sources VALUES(1,1,NULL,'paper_locator',7,'Lemma 4',NULL,NULL,NULL,'t');
INSERT INTO entity_sources VALUES(1,1,1,'t');
INSERT INTO relations VALUES(1,1,1,'EXTENDS',2,1,'active','sourced',0.5,'t');
INSERT INTO workstreams VALUES(1,1,'attack','Old ambiguous attack','failed','No refutation found','t','t');
INSERT INTO workstream_entities VALUES(1,1,1,'input','t');
PRAGMA user_version = 3;
"""

V04_SCHEMA_WITH_WORKSTREAM_CALL = """
CREATE TABLE projects (
    id INTEGER PRIMARY KEY CHECK (id = 1), name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE TABLE workstreams (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL,
    workstream_type TEXT NOT NULL
        CHECK(workstream_type IN ('literature','explore','attack','proof')),
    goal TEXT NOT NULL, status TEXT NOT NULL
        CHECK(status IN ('active','completed','blocked','abandoned','error','legacy_failed')),
    summary TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE(id,project_id), FOREIGN KEY(project_id) REFERENCES projects(id)
);
CREATE TABLE api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, workstream_id INTEGER,
    provider TEXT NOT NULL, model TEXT NOT NULL, purpose TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0, estimated_max_cost_usd REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'completed', error_message TEXT, response_text TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(workstream_id) REFERENCES workstreams(id) ON DELETE RESTRICT
);
INSERT INTO projects VALUES(1,'v04','','2026-01-01T00:00:00+00:00');
INSERT INTO workstreams VALUES(
    9,1,'attack','Preserve me','completed','Old result','t','t'
);
INSERT INTO api_calls(
    id,workstream_id,provider,model,purpose,input_tokens,output_tokens,cost_usd,
    estimated_max_cost_usd,status,created_at
) VALUES(3,9,'openai','gpt-5.6-sol','attack',10,5,0.01,0.2,'completed','t');
PRAGMA user_version = 4;
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
    assert [row[0] for row in migrations] == [1, 2, 3, 4, 5, 6]
    assert count == 2


def test_v02_migration_preserves_ambiguous_failed_and_downgrades_weak_sources(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    raw = sqlite3.connect(tmp_path / ".theory" / "research.db")
    raw.executescript(V02_SCHEMA_WITH_AMBIGUOUS_STATE)
    raw.commit()
    raw.close()

    with connect() as con:
        workstream = con.execute("SELECT * FROM workstreams WHERE id=1").fetchone()
        entity = con.execute("SELECT trust_state FROM entities WHERE id=1").fetchone()
        relation = con.execute("SELECT trust_state FROM relations WHERE id=1").fetchone()
        link = con.execute(
            "SELECT role FROM workstream_entities WHERE workstream_id=1"
        ).fetchone()
        version = con.execute("PRAGMA user_version").fetchone()[0]
        violations = con.execute("PRAGMA foreign_key_check").fetchall()

    assert workstream["status"] == "legacy_failed"
    assert workstream["summary"] == "No refutation found"
    assert entity["trust_state"] == "unverified"
    assert relation["trust_state"] == "unverified"
    assert link["role"] == "input"
    assert version == SCHEMA_VERSION
    assert violations == []


def test_v04_migration_adds_develop_type_and_preserves_workstream_calls(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    raw = sqlite3.connect(tmp_path / ".theory" / "research.db")
    raw.executescript(V04_SCHEMA_WITH_WORKSTREAM_CALL)
    raw.commit()
    raw.close()

    with connect() as con:
        old_workstream = con.execute(
            "SELECT * FROM workstreams WHERE id=9"
        ).fetchone()
        old_call = con.execute("SELECT * FROM api_calls WHERE id=3").fetchone()
        con.execute(
            """
            INSERT INTO workstreams(
                project_id,workstream_type,goal,status,summary,created_at,updated_at
            ) VALUES(1,'develop','Advance the idea','active','','t','t')
            """
        )
        con.execute(
            """
            INSERT INTO workstreams(
                project_id,workstream_type,goal,status,summary,created_at,updated_at
            ) VALUES(1,'research','Run bounded controller','active','','t','t')
            """
        )
        version = con.execute("PRAGMA user_version").fetchone()[0]
        violations = con.execute("PRAGMA foreign_key_check").fetchall()

    assert old_workstream["goal"] == "Preserve me"
    assert old_workstream["status"] == "completed"
    assert old_call["workstream_id"] == 9
    assert version == SCHEMA_VERSION
    assert violations == []
