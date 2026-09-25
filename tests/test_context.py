from theory.db import connect, initialize
from theory.graph import (
    add_entity,
    add_relation,
    add_source,
    create_workstream,
    link_workstream_entity,
    set_entity_trust,
)
from theory.research_context import for_entity, for_workstream


def test_context_selects_deterministic_graph_neighborhood(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".theory").mkdir()
    initialize("Context tests")

    with connect() as con:
        target = add_entity(con, "Conjecture", "Quadratic communication is possible")
        assumption = add_entity(con, "Assumption", "Authenticated channels")
        theorem = add_entity(con, "Theorem", "Known cubic protocol")
        attempt = add_entity(con, "ProofAttempt", "Compress transcript")
        counterexample = add_entity(con, "Counterexample", "Adaptive schedule")
        blocker = add_entity(con, "Obstruction", "Information bottleneck")
        finding = add_entity(con, "Finding", "The paper assumes static faults")
        inferred = add_entity(con, "Finding", "The target appears to need signatures", trust_state="inferred")
        speculative = add_entity(
            con, "OpenQuestion", "Could private coins avoid the barrier?", trust_state="speculative"
        )
        contradicted = add_entity(
            con, "Finding", "An obsolete optimistic claim", trust_state="contradicted"
        )
        quarantined = add_entity(
            con, "Counterexample", "Unreviewed model-generated schedule", trust_state="quarantined"
        )
        retired_neighbor = add_entity(con, "Finding", "Retired evidence")
        unrelated = add_entity(con, "Technique", "Unrelated tool")
        paper = add_entity(con, "Paper", "Primary source")

        add_relation(con, target, "DEPENDS_ON", assumption)
        add_relation(con, target, "EXTENDS", theorem)
        add_relation(con, attempt, "ATTEMPTS", target)
        add_relation(con, counterexample, "REFUTES", target)
        add_relation(con, blocker, "BLOCKS", target)
        add_relation(con, finding, "SUPPORTS", target)
        add_relation(con, inferred, "SUPPORTS", target, trust_state="inferred")
        add_relation(con, speculative, "SUPPORTS", target, trust_state="speculative")
        add_relation(con, contradicted, "CONTRADICTS", target, trust_state="contradicted")
        add_relation(con, quarantined, "REFUTES", target, trust_state="quarantined")
        retired_relation = add_relation(
            con, retired_neighbor, "SUPPORTS", target, status="retired", trust_state="sourced",
            evidence_source_id=add_source(
                con,
                retired_neighbor,
                external_url="https://example.test/retired",
            ),
        )
        add_source(con, finding, paper_entity_id=paper, page=9)
        set_entity_trust(con, finding, "sourced")

        workstream = create_workstream(con, "attack", "Attack the quadratic conjecture")
        link_workstream_entity(con, workstream, target, "input")
        link_workstream_entity(con, workstream, counterexample, "created")

    context = for_entity(target)
    ids = {entity["id"] for entity in context.entities}
    assert context.target_entity["id"] == target
    assert unrelated not in ids
    assert retired_neighbor not in ids
    assert retired_relation not in {relation["id"] for relation in context.relations}
    assert context.selections == {
        "assumptions": (assumption,),
        "nearest_theorems": (theorem,),
        "proof_attempts": (attempt,),
        "counterexamples": (counterexample, quarantined),
        "blockers": (blocker,),
        "source_backed_findings": (finding,),
    }
    assert any(source["entity_id"] == finding and source["page"] == 9 for source in context.sources)
    assert {entity["id"] for entity in context.epistemic["sourced"].entities} == {finding}
    assert inferred in {entity["id"] for entity in context.epistemic["inferred"].entities}
    assert speculative in {
        entity["id"] for entity in context.epistemic["speculative"].entities
    }
    assert contradicted in {
        entity["id"] for entity in context.epistemic["contradicted"].entities
    }
    assert quarantined in {
        entity["id"] for entity in context.epistemic["quarantined"].entities
    }
    payload = context.as_model_payload()
    assert "NOT THEOREM-VERIFIED" in payload["epistemic_partitions"]["sourced"]["label"]
    assert "DO NOT ASSUME TRUE" in payload["epistemic_partitions"]["quarantined"]["label"]
    assert any("Quarantined objects must never" in rule for rule in payload["epistemic_policy"])

    workstream_context = for_workstream(workstream)
    assert workstream_context.workstream["id"] == workstream
    assert target in {entity["id"] for entity in workstream_context.entities}
    assert counterexample in {entity["id"] for entity in workstream_context.entities}
