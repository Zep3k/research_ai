"""Offline recovery of controller rows left open by interrupted processes."""
import pytest

from theory.db import connect, utcnow
from theory.errors import TheoryError
from theory.research import (
    INTERRUPTED_API_CALL_ERROR,
    INTERRUPTED_ITERATION_ERROR,
    _reconcile_stale_research,
    _research_controller_lock,
    research,
)
from test_research import (
    DynamicProvider,
    artifact,
    init_workspace,
    make_research_workstream,
    step_report,
)


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    init_workspace(monkeypatch, tmp_path)
    return make_research_workstream()


def insert_iteration(con, workstream, target, number, status, *, material_progress=0):
    completed_at = utcnow() if status != "running" else None
    return con.execute(
        """
        INSERT INTO research_iterations(
            project_id,workstream_id,iteration_number,operation,target_entity_id,
            rationale,status,material_progress,created_at,completed_at,error_message
        ) VALUES(1,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            workstream, number, "develop", target, f"historical {status}", status,
            material_progress, "2026-01-02T03:04:05+00:00", completed_at,
            "existing error" if status == "error" else None,
        ),
    ).lastrowid


def insert_call(con, workstream, status, purpose, *, telemetry=False):
    return con.execute(
        """
        INSERT INTO api_calls(
            workstream_id,provider,model,purpose,input_tokens,output_tokens,cost_usd,
            estimated_max_cost_usd,status,error_message,response_text,
            uncached_input_tokens,cache_read_input_tokens,cache_write_input_tokens,
            cache_write_5m_input_tokens,cache_write_1h_input_tokens,reasoning_tokens,
            uncached_input_cost_usd,cache_read_cost_usd,cache_write_cost_usd,
            output_cost_usd,prompt_utf8_bytes,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            workstream, "openai", "gpt-6-sol", purpose,
            17 if telemetry else 0, 19 if telemetry else 0, 0.123 if telemetry else 0,
            0.8, status, "existing failure" if status == "failed" else None,
            "partial structured response" if telemetry else None,
            11 if telemetry else None, 3 if telemetry else None, 3 if telemetry else None,
            1 if telemetry else None, 2 if telemetry else None, 13 if telemetry else None,
            0.021 if telemetry else None, 0.002 if telemetry else None,
            0.004 if telemetry else None, 0.096 if telemetry else None,
            777 if telemetry else 12, "2026-01-02T03:04:05+00:00",
        ),
    ).lastrowid


def test_reconcile_stale_rows_is_zero_call_and_preserves_telemetry(workspace, monkeypatch):
    workstream, target = workspace
    with connect() as con:
        running = insert_iteration(con, workstream, target, 1, "running", material_progress=1)
        completed = insert_iteration(con, workstream, target, 2, "completed", material_progress=1)
        errored = insert_iteration(con, workstream, target, 3, "error")
        started = insert_call(con, workstream, "started", "research:attack", telemetry=True)
        completed_call = insert_call(con, workstream, "completed", "research:prove", telemetry=True)
        failed_call = insert_call(con, workstream, "failed", "research:strategy", telemetry=True)
        stable_iterations = {
            row["id"]: tuple(row) for row in con.execute(
                "SELECT * FROM research_iterations WHERE id IN (?,?) ORDER BY id",
                (completed, errored),
            )
        }
        stable_calls = {
            row["id"]: tuple(row) for row in con.execute(
                "SELECT * FROM api_calls WHERE id IN (?,?) ORDER BY id",
                (completed_call, failed_call),
            )
        }
        telemetry_before = dict(con.execute("SELECT * FROM api_calls WHERE id=?", (started,)).fetchone())

    monkeypatch.setattr("theory.research.get_provider", lambda *_: pytest.fail("No provider"))
    monkeypatch.setattr("theory.research.call_model", lambda **_: pytest.fail("No model call"))
    assert _reconcile_stale_research(workstream) == (1, 1)
    assert _reconcile_stale_research(workstream) == (0, 0)

    with connect() as con:
        iteration = dict(con.execute(
            "SELECT * FROM research_iterations WHERE id=?", (running,)
        ).fetchone())
        call = dict(con.execute("SELECT * FROM api_calls WHERE id=?", (started,)).fetchone())
        assert iteration["status"] == "error"
        assert iteration["error_message"] == INTERRUPTED_ITERATION_ERROR
        assert iteration["completed_at"] is not None
        assert iteration["material_progress"] == 0
        assert iteration["progress_class"] is None
        assert call["status"] == "failed"
        assert call["error_message"] == INTERRUPTED_API_CALL_ERROR
        for key, value in telemetry_before.items():
            if key not in {"status", "error_message"}:
                assert call[key] == value
        assert stable_iterations == {
            row["id"]: tuple(row) for row in con.execute(
                "SELECT * FROM research_iterations WHERE id IN (?,?) ORDER BY id",
                (completed, errored),
            )
        }
        assert stable_calls == {
            row["id"]: tuple(row) for row in con.execute(
                "SELECT * FROM api_calls WHERE id IN (?,?) ORDER BY id",
                (completed_call, failed_call),
            )
        }


def test_reconciliation_precedes_scheduling_and_numbering_stays_monotonic(
    workspace, monkeypatch
):
    workstream, target = workspace
    with connect() as con:
        stale_iteration = insert_iteration(con, workstream, target, 7, "running")
        stale_call = insert_call(con, workstream, "started", "research:strategy", telemetry=True)

    provider = DynamicProvider(
        lambda decision, _: step_report(
            decision,
            [artifact(
                "finding", "A bounded construction was recorded after recovery.",
                "post_recovery_construction", [decision["target_entity_id"]],
            )],
        )
    )
    monkeypatch.setattr("theory.research.get_provider", lambda name: provider)
    outcome = research(
        workstream, provider_name="openai", strategy="off", max_calls=1
    )
    assert outcome.calls_made == 1 and outcome.strategy_calls_made == 0
    assert len(provider.calls) == 1

    with connect() as con:
        rows = con.execute(
            "SELECT id,iteration_number,status,material_progress FROM research_iterations "
            "WHERE workstream_id=? ORDER BY iteration_number",
            (workstream,),
        ).fetchall()
        assert [tuple(row[1:3]) for row in rows] == [(7, "error"), (8, "completed")]
        assert rows[0]["id"] == stale_iteration and rows[0]["material_progress"] == 0
        assert con.execute(
            "SELECT status FROM api_calls WHERE id=?", (stale_call,)
        ).fetchone()[0] == "failed"
        assert con.execute(
            "SELECT COUNT(*) FROM api_calls WHERE workstream_id=?", (workstream,)
        ).fetchone()[0] == 2


def test_active_controller_lock_blocks_reconciliation(workspace, monkeypatch):
    workstream, target = workspace
    with connect() as con:
        stale = insert_iteration(con, workstream, target, 1, "running")
    monkeypatch.setattr("theory.research.get_provider", lambda *_: pytest.fail("No provider"))
    with _research_controller_lock(workstream):
        with pytest.raises(TheoryError, match="already active"):
            research(workstream, provider_name="openai", strategy="off", max_calls=1)
        with connect() as con:
            row = con.execute(
                "SELECT status,error_message,completed_at FROM research_iterations WHERE id=?",
                (stale,),
            ).fetchone()
            assert tuple(row) == ("running", None, None)
