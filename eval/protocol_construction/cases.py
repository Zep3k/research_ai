"""Small protocol contracts and inspectable structural oracles, not proof checkers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Piece:
    key: str
    concepts: tuple[str, ...]
    rules: tuple[str, ...] = ()
    state: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()
    invariant: str = ""
    needs: tuple[str, ...] = ()
    kind: str = "protocol_component"
    status: str = "unresolved"
    final: bool = False


@dataclass(frozen=True)
class Stage:
    operation: str
    pieces: tuple[Piece, ...]
    needs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Case:
    name: str
    contract: str
    stages: tuple[Stage, ...]
    required_concepts: tuple[str, ...]
    final_rules: tuple[str, ...]
    invariant: str
    forbidden_assumptions: tuple[str, ...] = ("fifo_channel", "perfect_failure_detector", "atomic_network_delivery")
    max_obligations: int = 0
    seeds: tuple[Piece, ...] = ()
    strategy: str = "off"


SEND = Piece("send_once", ("sender", "receiver"), ("start:send(DATA)", "DATA:deliver"), (), ("DATA",), "one_send_one_delivery", status="promising", final=True)
SEEN = Piece("seen_memory", ("deduplication",), state=("seen_ids",), invariant="seen_monotone")
GUARD = Piece("guard_handler", ("guarded_delivery",), ("DATA(id):if_unseen_record_then_deliver",), messages=("DATA(id)",), needs=("seen_memory",))
DEDUP = Piece("dedup_candidate", ("deduplication", "guarded_delivery"), GUARD.rules, SEEN.state, GUARD.messages, "seen_monotone", ("seen_memory", "guard_handler"), status="promising", final=True)
RETRY = Piece("retry_sender", ("retransmission",), ("timeout:resend_DATA", "ACK:stop_retry"), ("pending",), ("DATA(id)", "ACK(id)"), status="promising")
ACK = Piece("ack_receiver", ("acknowledgment", "deduplication"), ("DATA:ack_always_deliver_if_unseen",), ("seen_ids",), ("ACK(id)",), "seen_monotone", status="promising")
COMBINED = Piece("combined_candidate", ("retransmission", "acknowledgment", "deduplication"), RETRY.rules + ACK.rules, RETRY.state + ACK.state, RETRY.messages, "seen_monotone", (RETRY.key, ACK.key), kind="synthesis", status="promising", final=True)
QUEUE = Piece("queue_storage", ("bounded_queue",), state=("buffer", "capacity"), invariant="occupancy_bounded")
OBLIGATION = Piece("queue_bound", ("capacity_invariant",), invariant="occupancy_bounded", kind="proof_obligation")
ENQUEUE = Piece("queue_guard", ("guarded_enqueue",), ("enqueue:append_iff_below_capacity",), needs=(QUEUE.key, OBLIGATION.key))
QUEUE_FINAL = Piece("queue_candidate", ("bounded_queue", "guarded_enqueue", "dequeue"), ENQUEUE.rules + ("dequeue:pop_iff_nonempty",), QUEUE.state, invariant="occupancy_bounded", needs=(QUEUE.key, ENQUEUE.key, OBLIGATION.key), status="promising", final=True)
BASE = Piece("epoch_storage", ("epoch_state",), state=("epoch",), invariant="epoch_nondecreasing")
MIDDLE = Piece("epoch_merge", ("maximum_merge",), ("receive(e):epoch=max(epoch,e)",), needs=(BASE.key,))
TOP = Piece("epoch_broadcast", ("broadcast",), ("local_increment:broadcast_epoch",), messages=("EPOCH(e)",), needs=(MIDDLE.key,))
EPOCH_FINAL = Piece("epoch_candidate", ("epoch_state", "maximum_merge", "broadcast"), MIDDLE.rules + TOP.rules, BASE.state, TOP.messages, BASE.invariant, (BASE.key, MIDDLE.key, TOP.key), kind="proof_attempt", status="promising", final=True)

CASES = (
    Case("direct_contract", "One sender sends one value once over a reliable, nonduplicating channel to one receiver. No crashes. Deliver the value exactly once.", (Stage("develop", (SEND,)),), SEND.concepts, SEND.rules, SEND.invariant),
    Case("intermediate_invariant", "One sender sends uniquely identified values. Channel may duplicate/reorder but eventually delivers each value; no crashes. Deliver each ID exactly once. Construct state, invariant, then handler.", (Stage("develop", (SEEN,)), Stage("develop", (GUARD,), (SEEN.key,)), Stage("develop", (DEDUP,), (SEEN.key, GUARD.key))), DEDUP.concepts, DEDUP.rules, DEDUP.invariant),
    Case("failed_route_correction", "No crashes; a sent packet may arrive twice. Deliver once per ID. Examine delivery on every reception and correct it.", (Stage("develop", (Piece("naive_delivery_failure", ("duplicate_counterexample",), ("DATA(x),DATA(x):two_deliveries",), kind="failed_approach", status="failed"), SEEN)), Stage("develop", (Piece("corrected_candidate", DEDUP.concepts, DEDUP.rules, DEDUP.state, DEDUP.messages, DEDUP.invariant, (SEEN.key, "naive_delivery_failure"), status="promising", final=True),), (SEEN.key, "naive_delivery_failure"))), ("duplicate_counterexample", *DEDUP.concepts), DEDUP.rules, DEDUP.invariant),
    Case("combine_components", "No crashes. Fair-loss DATA and ACK channels; repeated sends eventually arrive. Provide eventual exactly-once delivery of a single ID by combining retries and a receiver that acknowledges duplicates but delivers once.", (Stage("develop", (RETRY, ACK)), Stage("synthesize", (COMBINED,), (RETRY.key, ACK.key))), COMBINED.concepts, COMBINED.rules, COMBINED.invariant, strategy="auto"),
    Case("no_reframing_needed", "Two processes gossip grow-only sets over reliable channels, without crashes. Construct union merge and dissemination for eventual agreement after updates cease.", (Stage("develop", (Piece("union_state", ("union_merge",), ("receive(S):state=union(state,S)",), ("set",), invariant="set_only_grows"),)), Stage("develop", (Piece("gossip_candidate", ("union_merge", "dissemination"), ("receive(S):state=union(state,S)", "update:broadcast_state"), ("set",), ("STATE(set)",), "set_only_grows", ("union_state",), status="promising", final=True),), ("union_state",))), ("union_merge", "dissemination"), ("receive(S):state=union(state,S)", "update:broadcast_state"), "set_only_grows"),
    Case("transitive_component_context", "No crashes; reliable broadcast. Each node increments its epoch locally and merges received epochs by max. Construct a monotonicity argument using the stored state, merge, and broadcast components.", (Stage("prove", (EPOCH_FINAL,), (BASE.key, MIDDLE.key, TOP.key)),), EPOCH_FINAL.concepts, EPOCH_FINAL.rules, EPOCH_FINAL.invariant, seeds=(BASE, MIDDLE, TOP)),
    Case("bounded_obligation_growth", "A sequential bounded FIFO queue of capacity K>0, with no concurrency. Reject enqueue when full and dequeue when empty. Specify storage, capacity guard, then full candidate; maintain 0<=occupancy<=K.", (Stage("develop", (QUEUE, OBLIGATION)), Stage("develop", (ENQUEUE, OBLIGATION), (QUEUE.key, OBLIGATION.key)), Stage("develop", (QUEUE_FINAL,), (QUEUE.key, ENQUEUE.key, OBLIGATION.key))), QUEUE_FINAL.concepts, QUEUE_FINAL.rules, QUEUE_FINAL.invariant, max_obligations=1),
)
