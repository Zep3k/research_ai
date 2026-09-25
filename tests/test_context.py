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
        unrelated = add_entity(con, "Technique", "Unrelated tool")
        paper = add_entity(con, "Paper", "Primary source")

        add_relation(con, target, "DEPENDS_ON", assumption)
        add_relation(con, target, "EXTENDS", theorem)
        add_relation(con, attempt, "ATTEMPTS", target)
        add_relation(con, counterexample, "REFUTES", target)
        add_relation(con, blocker, "BLOCKS", target)
        add_relation(con, finding, "SUPPORTS", target)
        add_source(con, finding, paper_entity_id=paper, page=9)
        set_entity_trust(con, finding, "sourced")

        workstream = create_workstream(con, "attack", "Attack the quadratic conjecture")
        link_workstream_entity(con, workstream, target, "input")
        link_workstream_entity(con, workstream, counterexample, "created")

    context = for_entity(target)
    ids = {entity["id"] for entity in context.entities}
    assert context.target_entity["id"] == target
    assert unrelated not in ids
    assert context.selections == {
        "assumptions": (assumption,),
        "nearest_theorems": (theorem,),
        "proof_attempts": (attempt,),
        "counterexamples": (counterexample,),
        "blockers": (blocker,),
        "source_backed_findings": (finding,),
    }
    assert any(source["entity_id"] == finding and source["page"] == 9 for source in context.sources)

    workstream_context = for_workstream(workstream)
    assert workstream_context.workstream["id"] == workstream
    assert target in {entity["id"] for entity in workstream_context.entities}
    assert counterexample in {entity["id"] for entity in workstream_context.entities}
