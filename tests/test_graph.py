import sqlite3

import pytest

from theory.db import SCHEMA_VERSION, connect, initialize
from theory.errors import TheoryError, TrustError
from theory.graph import (
    add_entity,
    add_relation,
    add_review,
    add_source,
    compare_attributes,
    create_workstream,
    link_workstream_entity,
    list_attributes,
    normalize_attribute_key,
    set_attribute,
    set_entity_trust,
    set_workstream_status,
)
from theory.trust import source_has_identifiable_origin


def workspace(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    initialize("Graph tests")


def test_schema_initialization_has_v02_tables_and_version(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)

    with connect() as con:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        versions = [
            row[0] for row in con.execute("SELECT version FROM schema_migrations ORDER BY version")
        ]
        version = con.execute("PRAGMA user_version").fetchone()[0]
        foreign_keys = con.execute("PRAGMA foreign_keys").fetchone()[0]
        violations = con.execute("PRAGMA foreign_key_check").fetchall()

    assert {
        "entities",
        "relations",
        "sources",
        "entity_sources",
        "entity_attributes",
        "workstreams",
        "workstream_entities",
        "reviews",
        "schema_migrations",
    } <= tables
    assert versions == [1, 2, 3, 4]
    assert version == SCHEMA_VERSION
    assert foreign_keys == 1
    assert violations == []


def test_entities_are_typed_and_attribute_keys_are_mechanically_normalized(
    monkeypatch, tmp_path
):
    workspace(monkeypatch, tmp_path)
    with connect() as con:
        theorem = add_entity(con, "theorem", "Known lower bound")
        set_attribute(con, theorem, "fault_model", "Byzantine")
        set_attribute(con, theorem, " Communication--Complexity ", "O(n^3)")
        set_attribute(con, theorem, "communication complexity", "O(n^2)")
        attrs = list_attributes(con, theorem)

        with pytest.raises(TheoryError, match="Invalid entity type"):
            add_entity(con, "ChatMessage", "Not a research object")

    assert attrs == {
        "communication_complexity": "O(n^2)",
        "fault_model": "Byzantine",
    }
    assert normalize_attribute_key(" fault-- model ") == "fault_model"
    assert normalize_attribute_key("communication") != normalize_attribute_key(
        "communication complexity"
    )


def test_valid_and_invalid_relations_and_delete_behavior(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    with connect() as con:
        known = add_entity(con, "Theorem", "Known")
        conjecture = add_entity(con, "Conjecture", "Improvement")
        disposable = add_entity(con, "Lemma", "Disposable")
        set_attribute(con, disposable, "model", "LOCAL")
        relation = add_relation(con, conjecture, "extends", known)

        assert con.execute(
            "SELECT relation_type FROM relations WHERE id=?", (relation,)
        ).fetchone()[0] == "EXTENDS"
        with pytest.raises(TheoryError, match="Invalid relation type"):
            add_relation(con, conjecture, "TALKS_ABOUT", known)
        with pytest.raises(TheoryError, match="does not exist"):
            add_relation(con, conjecture, "EXTENDS", 999)
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("DELETE FROM entities WHERE id=?", (known,))

        con.execute("DELETE FROM entities WHERE id=?", (disposable,))
        assert con.execute(
            "SELECT COUNT(*) FROM entity_attributes WHERE entity_id=?", (disposable,)
        ).fetchone()[0] == 0


def test_sourced_claims_require_persisted_provenance(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    with connect() as con:
        with pytest.raises(TrustError, match="identifiable origin"):
            add_entity(con, "Finding", "Claim", trust_state="sourced")

        paper = add_entity(con, "Paper", "Primary paper")
        claim = add_entity(con, "Theorem", "Theorem 3.2")
        incomplete_source = add_source(
            con, claim, page=6, section="Nearby discussion"
        )
        incomplete = con.execute(
            "SELECT * FROM sources WHERE id=?", (incomplete_source,)
        ).fetchone()
        assert source_has_identifiable_origin(incomplete) is False
        with pytest.raises(TrustError, match="identifiable origin"):
            set_entity_trust(con, claim, "sourced")
        with pytest.raises(sqlite3.IntegrityError, match="identifiable provenance"):
            con.execute(
                "UPDATE entities SET trust_state='sourced' WHERE id=?", (claim,)
            )

        source_id = add_source(
            con,
            claim,
            paper_entity_id=paper,
            page=7,
            section="Main results",
            theorem="Theorem 3.2",
            excerpt="Under assumptions A and B, the bound holds.",
            external_url="https://doi.org/10.1/example",
        )
        identifiable = con.execute(
            "SELECT * FROM sources WHERE id=?", (source_id,)
        ).fetchone()
        assert source_has_identifiable_origin(identifiable) is True
        set_entity_trust(con, claim, "sourced")

        row = con.execute("SELECT trust_state FROM entities WHERE id=?", (claim,)).fetchone()
        assert row[0] == "sourced"
        assert source_id in {
            row[0]
            for row in con.execute(
                "SELECT source_id FROM entity_sources WHERE entity_id=?", (claim,)
            ).fetchall()
        }
        with pytest.raises(sqlite3.IntegrityError, match="remove provenance origin"):
            con.execute(
                "UPDATE sources SET paper_entity_id=NULL,external_url=NULL WHERE id=?",
                (source_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="last identifiable source"):
            con.execute(
                "DELETE FROM entity_sources WHERE entity_id=? AND source_id=?",
                (claim, source_id),
            )


def test_external_url_is_an_identifiable_origin(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    with connect() as con:
        claim = add_entity(con, "Finding", "Externally documented claim")
        source_id = add_source(
            con,
            claim,
            page=3,
            external_url="https://arxiv.org/abs/2401.00001",
        )
        set_entity_trust(con, claim, "sourced")
        source = con.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        assert source_has_identifiable_origin(source) is True
        assert con.execute(
            "SELECT trust_state FROM entities WHERE id=?", (claim,)
        ).fetchone()[0] == "sourced"


def test_trust_transitions_are_explicit_and_quarantine_is_sticky(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    with connect() as con:
        claim = add_entity(con, "Conjecture", "Candidate", trust_state="speculative")
        with pytest.raises(TrustError, match="identifiable origin"):
            set_entity_trust(con, claim, "sourced")

        paper = add_entity(con, "Paper", "Reference")
        add_source(con, claim, paper_entity_id=paper, page=2)
        set_entity_trust(con, claim, "sourced")
        set_entity_trust(con, claim, "contradicted")
        with pytest.raises(TrustError, match="not allowed"):
            set_entity_trust(con, claim, "sourced")
        set_entity_trust(con, claim, "unverified")
        set_entity_trust(con, claim, "sourced")


def test_sourced_relations_require_evidence(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    with connect() as con:
        a = add_entity(con, "Conjecture", "A")
        b = add_entity(con, "Theorem", "B")
        incomplete = add_source(con, a, page=4, section="Claimed theorem")
        with pytest.raises(TrustError, match="identifiable origin"):
            add_relation(
                con,
                a,
                "EXTENDS",
                b,
                trust_state="sourced",
                evidence_source_id=incomplete,
            )
        with pytest.raises(TrustError, match="identifiable origin"):
            add_relation(con, a, "EXTENDS", b, trust_state="sourced")

        paper = add_entity(con, "Paper", "Reference")
        source_id = add_source(con, a, paper_entity_id=paper, page=4)
        relation = add_relation(
            con,
            a,
            "EXTENDS",
            b,
            trust_state="sourced",
            evidence_source_id=source_id,
        )
        assert con.execute(
            "SELECT trust_state FROM relations WHERE id=?", (relation,)
        ).fetchone()[0] == "sourced"


def test_workstream_lifecycle_is_separate_from_scientific_outcome(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    with connect() as con:
        conjecture = add_entity(con, "Conjecture", "False direction")
        obstruction = add_entity(con, "Obstruction", "Indistinguishability argument")
        workstream = create_workstream(con, "attack", "Try to refute the conjecture")
        link_workstream_entity(con, workstream, conjecture, "input")
        link_workstream_entity(con, workstream, obstruction, "created")
        set_workstream_status(con, workstream, "completed", summary="Attack pass finished.")
        add_review(
            con,
            "counterexample_attempt",
            "no_flaw_found",
            workstream_id=workstream,
            issues="This pass found no refutation; correctness is not implied.",
        )

    with connect() as con:
        row = con.execute("SELECT * FROM workstreams WHERE id=?", (workstream,)).fetchone()
        links = con.execute(
            "SELECT role FROM workstream_entities WHERE workstream_id=? ORDER BY role",
            (workstream,),
        ).fetchall()

    assert row["status"] == "completed"
    assert row["summary"] == "Attack pass finished."
    assert [link[0] for link in links] == ["created", "input"]


def test_review_results_cannot_claim_proof_verification(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    with connect() as con:
        theorem = add_entity(con, "Theorem", "Candidate proof")
        review = add_review(
            con,
            "proof_critique",
            "no_flaw_found",
            target_entity_id=theorem,
            issues="Checked the induction step only.",
        )
        assert con.execute("SELECT result FROM reviews WHERE id=?", (review,)).fetchone()[0] == (
            "no_flaw_found"
        )
        with pytest.raises(TheoryError, match="Invalid review result"):
            add_review(
                con, "proof_critique", "verified", target_entity_id=theorem
            )


def test_theorem_delta_covers_all_four_sections(monkeypatch, tmp_path):
    workspace(monkeypatch, tmp_path)
    with connect() as con:
        a = add_entity(con, "Theorem", "Known")
        b = add_entity(con, "Conjecture", "Improved")
        for entity_id, key, value in (
            (a, "synchrony", "asynchronous"),
            (b, "synchrony", "asynchronous"),
            (a, "communication", "O(n^3)"),
            (b, "communication", "O(n^2)"),
            (a, "authentication", "signatures"),
            (b, "randomization", "private coins"),
        ):
            set_attribute(con, entity_id, key, value)

        delta = compare_attributes(con, a, b)

    assert delta.unchanged == (("synchrony", "asynchronous"),)
    assert delta.changed == (("communication", "O(n^3)", "O(n^2)"),)
    assert delta.only_a == (("authentication", "signatures"),)
    assert delta.only_b == (("randomization", "private coins"),)
