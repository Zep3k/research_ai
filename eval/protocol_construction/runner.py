"""Offline real-controller runs. Only provider completion is replaced with scripts.

Run from the repository root: python -m eval.protocol_construction.runner
The JSON inside artifact statements is an evaluation DSL, not a production schema.
No exact prose comparison, network, extra model calls, or existing workspace writes.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import theory.research as controller
from theory.config import Config
from theory.db import connect, initialize
from theory.graph import add_entity, add_relation, create_workstream, link_workstream_entity, set_attribute
from theory.models import ModelResult
from theory.prompts import render_prompt
from theory.research_context import for_workstream

from .cases import CASES, Case, Piece


def structure(piece: Piece, *, bad_assumption: bool = False) -> dict:
    return {"concepts": piece.concepts, "rules": piece.rules, "state": piece.state,
            "messages": piece.messages, "invariant": piece.invariant,
            "dependencies": piece.needs, "candidate": piece.final,
            "assumptions": ["fifo_channel"] if bad_assumption else []}


def artifact(piece: Piece, references: list[int], *, bad_assumption: bool = False) -> dict:
    return {"artifact_type": piece.kind, "material_key": piece.key,
            "statement": json.dumps(structure(piece, bad_assumption=bad_assumption), sort_keys=True),
            "reasoning_summary": f"Contract-scoped structural fixture for {piece.key}; not verified.",
            "related_entity_ids": sorted(set(references)), "source_ids": [],
            "epistemic_status": "speculation", "branch_status": piece.status}


@contextmanager
def workspace():
    previous = Path.cwd()
    with TemporaryDirectory(prefix="theory-protocol-eval-") as directory:
        try:
            os.chdir(directory)
            Path(".theory").mkdir()
            initialize("Offline protocol evaluation")
            Config(monthly_budget_usd=100.0).save()
            yield
        finally:
            os.chdir(previous)


class ScriptedProvider:
    def __init__(self, case: Case, fault: str | None):
        self.case, self.fault = case, fault
        self.cursor = 0
        self.execution_calls = self.strategy_calls = 0
        self.diagnostics: list[str] = []
        self.context_ids: list[list[int]] = []

    def complete(self, **kwargs):
        prompt = render_prompt(kwargs["prompt"])
        if "CONTROLLER DECISION\n" not in prompt:
            self.strategy_calls += 1
            state = json.loads(prompt.split("RESEARCH STATE\n", 1)[1])
            moves = state["legal_moves"]
            # A bounded selector: prefer the requested combination if it is legal,
            # otherwise retain the controller's first offered/default choice.
            expected = self.case.stages[min(self.cursor, len(self.case.stages) - 1)]
            move = next((m for m in moves if m["operation"] == expected.operation), moves[0])
            response = {"selected_move_id": move["move_id"], "rationale": "Select the requested structural stage if offered, otherwise the first legal move."}
        else:
            self.execution_calls += 1
            decision_text, graph_text = prompt.split("CONTROLLER DECISION\n", 1)[1].split("\n\nGRAPH CONTEXT\n", 1)
            decision = json.loads(decision_text)
            graph, _ = json.JSONDecoder().raw_decode(graph_text)
            attrs = graph["attributes_by_entity_id"]
            keys = {a["research_material_key"]: int(i) for i, a in attrs.items() if "research_material_key" in a}
            self.context_ids.append(sorted(graph["context_scope"]["included_entity_ids"]))
            stage = self.case.stages[min(self.cursor, len(self.case.stages) - 1)]
            missing = sorted(set(stage.needs) - keys.keys())
            wrong_move = decision["operation"] != stage.operation
            pieces = ()
            if missing or wrong_move:
                self.diagnostics.append(f"call {self.execution_calls}: expected {stage.operation}; got {decision['operation']}; missing components {missing}")
            else:
                pieces = stage.pieces
                self.cursor += 1
            # Honest failed prerequisite/proof work emits missing-premise artifacts,
            # not a candidate fabricated from context that the executor never saw.
            if self.fault == "obligation_fanout" or ((missing or wrong_move) and (decision["currently_open_obligation_ids"] or decision["operation"] == "prove")):
                pieces = tuple(Piece(f"missing_{self.execution_calls}_{i}", (("capacity storage allocation finite memory bound" if i == 0 else "delivery scheduling fairness eventual message reception") + f" stage{self.execution_calls}",), invariant=f"unestablished_{i}", kind="proof_obligation") for i in range(2))
            result_artifacts = []
            for piece in pieces:
                refs = [decision["target_entity_id"], *decision["required_consumed_entity_ids"]]
                refs.extend(keys[key] for key in piece.needs if key in keys)
                result_artifacts.append(artifact(piece, refs, bad_assumption=self.fault == "forbidden_assumption" and piece.final))
            response = {"operation": decision["operation"], "target_entity_id": decision["target_entity_id"],
                        "summary": "Bounded deterministic protocol stage.", "artifacts": result_artifacts,
                        "consumed_entity_ids": decision["required_consumed_entity_ids"], "addressed_obligation_ids": [],
                        "attack_outcome": "inconclusive" if decision["operation"] == "attack" else "not_applicable",
                        "could_not_determine": ["Required construction stage or component unavailable."] if missing or wrong_move else [],
                        "human_judgment_required": False, "human_judgment_reason": None}
        return ModelResult(text=json.dumps(response), input_tokens=0, output_tokens=0, cost_usd=0)


def check_milestones(case: Case, context, traces: list[dict]) -> dict[str, bool]:
    records = {}
    for entity in context.entities:
        attrs = context.attributes.get(int(entity["id"]), {})
        if "research_material_key" in attrs:
            records[attrs["research_material_key"]] = (entity, attrs, json.loads(attrs["research_statement"]))
    concepts = {concept for _, _, record in records.values() for concept in record["concepts"]}
    candidates = [(entity, attrs, record) for entity, attrs, record in records.values() if record["candidate"]]
    expected = next(piece for stage in case.stages for piece in stage.pieces if piece.final)
    ids = {key: int(value[0]["id"]) for key, value in records.items()}
    def valid_final(item):
        _, attrs, record = item
        return (attrs["research_artifact_type"] == expected.kind
                and attrs.get("research_branch_status") == "promising"
                and set(expected.concepts) <= set(record["concepts"])
                and set(case.final_rules) <= set(record["rules"])
                and set(expected.state) <= set(record["state"])
                and set(expected.messages) <= set(record["messages"])
                and set(expected.needs) <= set(record["dependencies"])
                and all(key in ids and ids[key] in json.loads(attrs["related_entity_ids"]) for key in expected.needs)
                and record["invariant"] == case.invariant)
    return {
        "required_concepts": set(case.required_concepts) <= concepts,
        "key_invariant": any(r["invariant"] == case.invariant for _, _, r in records.values()),
        "no_forbidden_assumptions": all(not set(r["assumptions"]) & set(case.forbidden_assumptions) for _, _, r in candidates),
        "final_candidate_structure": any(valid_final(item) for item in candidates),
        "bounded_obligations": max((len(t["open_obligations"]) for t in traces), default=0) <= case.max_obligations,
        "quarantine_preserved": all(entity["trust_state"] == "quarantined" for entity, _, _ in records.values()),
        "no_unnecessary_reframe": all(t["operation"] != "reframe" for t in traces),
        "components_combined": case.name != "combine_components" or any(t["operation"] == "synthesize" and len(t["consumed_entity_ids"]) == 2 for t in traces),
        "failed_route_preserved": case.name != "failed_route_correction" or any(attrs.get("research_branch_status") == "failed" for _, attrs, _ in records.values()),
    }


def run_case(case: Case, *, fault: str | None = None) -> dict:
    with workspace():
        with connect() as con:
            primary = add_entity(con, "ResearchIdea", case.name, body=case.contract)
            ws = create_workstream(con, "research", case.contract)
            link_workstream_entity(con, ws, primary, "input")
            ids = {}
            for piece in case.seeds:
                entity = add_entity(con, "Technique", piece.key, body=json.dumps(structure(piece)), generated_by_llm=True, trust_state="quarantined")
                link_workstream_entity(con, ws, entity, "created")
                refs = [ids[key] for key in piece.needs] or [primary]
                for key, value in {"research_material_key": piece.key, "research_statement": json.dumps(structure(piece)), "research_artifact_type": piece.kind, "related_entity_ids": json.dumps(refs), "precise_candidate": "true", "research_branch_status": piece.status}.items():
                    set_attribute(con, entity, key, value)
                ids[piece.key] = entity
        provider = ScriptedProvider(case, fault)
        traces = []
        complete = controller._complete_iteration
        def record(iteration_id, workstream_id, choice, report, progress):
            complete(iteration_id, workstream_id, choice, report, progress)
            row = controller._history(ws)[-1]
            context = for_workstream(ws)
            accepted = json.loads(row["artifact_ids_json"])
            accepted_keys = [context.attributes[i]["research_material_key"] for i in accepted]
            # Fixture DSL dependencies between co-produced objects cannot name
            # graph IDs until the receipt is persisted. Reconstruct those exact
            # edges before the next execution; contract links are not dependencies.
            siblings = dict(zip(accepted_keys, accepted))
            with connect() as con:
                for entity_id in accepted:
                    declared = json.loads(context.attributes[entity_id]["research_statement"])["dependencies"]
                    for key in declared:
                        if key in siblings and siblings[key] != entity_id:
                            add_relation(con, entity_id, "DEPENDS_ON", siblings[key], trust_state="quarantined")
            remaining = Counter(accepted_keys)
            duplicate_keys = []
            for item in report.artifacts:
                if remaining[item.material_key]:
                    remaining[item.material_key] -= 1
                else:
                    duplicate_keys.append(item.material_key)
            traces.append({"iteration": row["iteration_number"], "legal_moves": json.loads(row["legal_move_ids_json"]),
                           "selected_move": row["selected_move_id"], "operation": choice.operation,
                           "focused_context_ids": provider.context_ids[-1], "consumed_entity_ids": list(choice.consumed_entity_ids),
                           "accepted_artifacts": accepted, "accepted_keys": accepted_keys, "duplicate_keys": duplicate_keys,
                           "duplicate_count": progress.duplicate_count,
                           "generated_obligations": [i for i in accepted if context.attributes[i].get("is_proof_obligation") == "true"],
                           "open_obligations": list(controller._open_obligation_ids(context, ws, primary)),
                           "branch_states": {str(i): a["research_branch_status"] for i, a in sorted(context.attributes.items()) if "research_branch_status" in a},
                           "material_progress": progress.material_progress, "resolution_progress": progress.resolution_progress,
                           "progress_class": progress.progress_class})
        with patch.object(controller, "get_provider", return_value=provider), patch.object(controller, "_complete_iteration", side_effect=record):
            outcome = controller.research(ws, max_calls=len(case.stages), strategy=case.strategy)
        checks = check_milestones(case, for_workstream(ws), traces)
        with connect() as con:
            logged_calls = con.execute("SELECT COUNT(*) FROM api_calls WHERE workstream_id=?", (ws,)).fetchone()[0]
        return {"case": case.name, "fault": fault, "passed": all(checks.values()), "checks": checks,
                "diagnostics": provider.diagnostics, "execution_calls": provider.execution_calls,
                "strategy_calls": provider.strategy_calls, "logged_calls": logged_calls,
                "stop_reason": outcome.stop_reason, "trace": traces}


def benchmark() -> dict:
    results = [run_case(case) for case in CASES]
    controls = [run_case(CASES[0], fault="forbidden_assumption"), run_case(CASES[-1], fault="obligation_fanout")]
    return {"passed": sum(r["passed"] for r in results), "total": len(results), "results": results,
            "negative_controls": controls, "note": "Structural orchestration checks with scripted executors; no evidence of real-model intelligence or proof verification."}


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
