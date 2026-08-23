"""
state_graph/billing_dispute.py

Post-Event Billing Dispute state graph (Issue #6 / GitHub issue #68).

Problem this solves
--------------------
Post-event billing resolution can't be a single request/response call: it
waits on the client's reaction to an invoice, and if the client disputes it,
resolving that can take several rounds of negotiation before either side is
satisfied -- or before it needs to be escalated to a human. That requires a
graph whose state survives between turns, not a one-shot function.

Shape of the graph
-------------------
    generate_invoice -> reconcile_ledger -> [entry/edge router] --+
                                                                   |
        +---------------- client ACCEPTED -----------------------+--> finalize_billing -> END
        |
        +---- client DISPUTED (rounds < MAX_NEGOTIATION_ROUNDS) ----> draft_dispute_email -> END (turn)
        |
        +-- client REJECTED_FINAL, or negotiation deadlocked ------> escalate_to_finance
                                                                           |
                                                        (graph pauses here, interrupt_before)
                                                                           v
                                                                  human_finance_review -> finalize_billing -> END

* `reconcile_ledger` implements **Task Decomposition**: the reconciliation
  goal is broken into an ordered set of dependent subtasks (pull ledger
  facts -> compute expected charges -> diff expected vs. recorded), each
  executed and traced individually, mirroring the decompose -> execute ->
  trace shape used by the Planning Agent's task graphs
  (agent/planning_agent_executor.py).
* `draft_dispute_email` implements **Tree of Thoughts**: it generates
  several candidate negotiation strategies ("thoughts"), scores each one
  against the reconciliation facts and how many rounds of back-and-forth
  have already happened, and selects the best-scoring candidate.
* `escalate_to_finance` + `human_finance_review` implement the **HITL
  node**: the graph is compiled with `interrupt_before=["human_finance_review"]`,
  so as soon as `escalate_to_finance` writes an `admin_tickets` row and the
  graph is about to run `human_finance_review`, execution pauses. A finance
  admin (via the future admin dashboard, Issue #73 / #71) reviews the ticket
  and calls `resume_after_finance_review(...)`, which is the only way
  `human_finance_review` is allowed to run.

Deliberate scoping note
------------------------
This module intentionally does NOT import
`planning/planning_lab/algorithms/{decomposition,tree_of_thoughts}.py`.
Those modules are built around the Planning Agent's task-graph / MCP
tool-calling domain (NetworkX DAGs, `ChatGroq`, MCP tool execution), not a
persistent LangGraph state machine. Reusing them here would mean guessing
at internals this file doesn't own. The same decompose -> execute -> trace
and generate -> evaluate -> select patterns they document are reproduced
locally instead, in a form that fits LangGraph nodes.

For the same reason, email drafting here is template + heuristic-scored
rather than an LLM call: it keeps this module deterministic and runnable
without a `GROQ_API_KEY`, in the same spirit as `context_eval/evaluate.py`
and `planning_eval/evaluate_planning.py` being offline, reproducible
evaluators. Swapping `_draft_email_body` / `_score_candidate` for Groq
calls (mirroring `agent/client.py`'s `handle_sampling`) is a natural
follow-up but is not required for this graph to be correct.

Persistence
-----------
Checkpointing uses `langgraph_checkpoint_sqlite.SqliteSaver` against the
SAME `db/aurelia.db` file that `db/setup_db.py` already prepares
checkpoint tables in ("LangGraph Checkpoint Tables ... inside the SAME
aurelia.db file"). Each event's dispute lives on its own thread
(`thread_id = event_id`), so re-invoking the graph for the same event
resumes exactly where that event's negotiation left off.

Run the built-in demo
----------------------
    python db/setup_db.py      # once, to create + seed db/aurelia.db
    python -m state_graph.billing_dispute
"""

import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional, TypedDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from langgraph.graph import END, StateGraph
from state_graph.recovery import with_error_handling
from state_graph.tickets import open_hitl_ticket

# Define the database path dynamically, matching mcp_server/server.py's convention.
DB_PATH = os.path.join(REPO_ROOT, 'db', 'aurelia.db')

GRAPH_ID = "billing_dispute"

# --- Lab pricing policy placeholder (mirrors the PIN=1234 placeholder in
# mcp_server/server.py -- a stand-in for a real rate card / billing system). ---
PER_GUEST_RATE = 150.00
MAX_NEGOTIATION_ROUNDS = 3


class BillingDisputeState(TypedDict, total=False):
    event_id: str
    invoice: dict
    reconciliation: dict
    client_status: Literal["PENDING", "ACCEPTED", "DISPUTED", "REJECTED_FINAL"]
    client_feedback: Optional[str]
    negotiation_round: int
    candidate_emails: list
    draft_email: Optional[str]
    escalation_ticket_id: Optional[str]
    finance_decision: Optional[str]
    resolution: Optional[str]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        raise RuntimeError(
            f"Database not found at {DB_PATH}. Run `python db/setup_db.py` first."
        )
    return sqlite3.connect(DB_PATH)


def _fetch_event(event_id: str) -> dict:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT event_id, guest_id, room_id, status, headcount, deposit_required "
        "FROM events WHERE event_id = ?",
        (event_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"Event not found: {event_id}")
    return {
        "event_id": row[0],
        "guest_id": row[1],
        "room_id": row[2],
        "status": row[3],
        "headcount": row[4],
        "deposit_required": row[5],
    }


# ---------------------------------------------------------------------------
# NODE: generate_invoice
# ---------------------------------------------------------------------------

@with_error_handling("billing_dispute", "generate_invoice")
def generate_invoice(state: BillingDisputeState) -> dict:
    event = _fetch_event(state["event_id"])
    subtotal = round(event["headcount"] * PER_GUEST_RATE, 2)
    deposit_credited = event["deposit_required"] if event["status"] == "CONFIRMED" else 0.0
    balance_due = round(subtotal - deposit_credited, 2)

    invoice = {
        "event_id": event["event_id"],
        "headcount": event["headcount"],
        "per_guest_rate": PER_GUEST_RATE,
        "subtotal": subtotal,
        "deposit_required": event["deposit_required"],
        "deposit_credited": deposit_credited,
        "balance_due": balance_due,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "invoice": invoice,
        "client_status": state.get("client_status", "PENDING"),
        "negotiation_round": state.get("negotiation_round", 0),
    }


# ---------------------------------------------------------------------------
# NODE: reconcile_ledger -- TASK DECOMPOSITION
# ---------------------------------------------------------------------------
# The reconciliation goal ("does the ledger match what we billed?") is
# broken into an ordered set of dependent subtasks, each executed and
# traced individually, instead of one opaque calculation.

def _subtask_pull_ledger_facts(state: BillingDisputeState) -> dict:
    event = _fetch_event(state["event_id"])
    return {"status": "done", "result": event}


def _subtask_compute_expected_charges(state: BillingDisputeState) -> dict:
    invoice = state["invoice"]
    return {
        "status": "done",
        "result": {
            "expected_subtotal": invoice["subtotal"],
            "expected_deposit_credit": invoice["deposit_required"],
        },
    }


def _subtask_diff_recorded_vs_expected(state: BillingDisputeState) -> dict:
    invoice = state["invoice"]
    discrepancies = []
    if invoice["deposit_credited"] < invoice["deposit_required"]:
        discrepancies.append(
            f"Deposit of ${invoice['deposit_required']:.2f} is not yet reflected as paid "
            f"on the ledger (event status is not CONFIRMED)."
        )
    if invoice["balance_due"] < 0:
        discrepancies.append(
            f"Recorded deposit credit (${invoice['deposit_credited']:.2f}) exceeds the "
            f"computed subtotal (${invoice['subtotal']:.2f}); a refund of "
            f"${-invoice['balance_due']:.2f} may be owed to the client."
        )
    return {"status": "done", "result": {"discrepancies": discrepancies}}


_RECONCILIATION_SUBTASKS = [
    (
        "pull_ledger_facts",
        "Pull the event's recorded billing facts from the database.",
        _subtask_pull_ledger_facts,
    ),
    (
        "compute_expected_charges",
        "Compute the expected charges from headcount and the rate card.",
        _subtask_compute_expected_charges,
    ),
    (
        "diff_recorded_vs_expected",
        "Diff recorded ledger entries against expected charges to flag discrepancies.",
        _subtask_diff_recorded_vs_expected,
    ),
]

@with_error_handling("billing_dispute", "reconcile_ledger")
def reconcile_ledger(state: BillingDisputeState) -> dict:
    trace = []
    subtask_state = dict(state)
    for subtask_id, description, fn in _RECONCILIATION_SUBTASKS:
        outcome = fn(subtask_state)
        trace.append({"id": subtask_id, "description": description, **outcome})
        # Later subtasks can see earlier subtasks' results.
        subtask_state[f"_subtask_{subtask_id}"] = outcome.get("result")

    discrepancies = trace[-1]["result"]["discrepancies"]
    reconciliation = {
        "subtasks": trace,
        "discrepancies": discrepancies,
        "clean": len(discrepancies) == 0,
        "completed": True,
    }
    return {"reconciliation": reconciliation}


# ---------------------------------------------------------------------------
# NODE: draft_dispute_email -- TREE OF THOUGHTS
# ---------------------------------------------------------------------------
# Generates several candidate negotiation "thoughts" (distinct strategies),
# scores each against the reconciliation facts and the current negotiation
# round, and selects the highest-scoring candidate. Depth is bounded to one
# ply (generate -> evaluate -> select); repeated rounds happen by
# re-entering this node on the client's next turn -- the persistence the
# SqliteSaver checkpointing exists for.

_STRATEGIES = [
    {"name": "goodwill_concession", "tone": "warm and conciliatory", "concession_fraction": 0.15},
    {"name": "itemized_justification", "tone": "factual and detailed", "concession_fraction": 0.0},
    {"name": "firm_policy_restatement", "tone": "firm but respectful", "concession_fraction": 0.0},
]


def _draft_email_body(strategy: dict, state: BillingDisputeState) -> str:
    invoice = state["invoice"]
    reconciliation = state["reconciliation"]
    balance_due = invoice["balance_due"]
    concession = round(balance_due * strategy["concession_fraction"], 2)
    adjusted_balance = round(balance_due - concession, 2)
    discrepancy_lines = "\n".join(f"- {d}" for d in reconciliation["discrepancies"]) or (
        "- No discrepancies found; the ledger matches the invoiced amount."
    )

    if strategy["name"] == "goodwill_concession":
        return (
            f"Subject: Following up on your invoice for {invoice['event_id']}\n\n"
            f"Hello,\n\nThank you for raising your concerns about the final invoice. "
            f"We've reviewed the ledger for your event and want to make this right. "
            f"As a gesture of goodwill, we're reducing the outstanding balance by "
            f"${concession:.2f}, bringing the total due to ${adjusted_balance:.2f}.\n\n"
            f"Reconciliation notes:\n{discrepancy_lines}\n\n"
            f"Please let us know if this resolves your concern.\n\nWarm regards,\nAurelia Billing Team"
        )
    if strategy["name"] == "itemized_justification":
        return (
            f"Subject: Itemized breakdown for {invoice['event_id']}\n\n"
            f"Hello,\n\nWe'd like to walk through exactly how the ${invoice['subtotal']:.2f} "
            f"subtotal was calculated: {invoice['headcount']} guests at "
            f"${invoice['per_guest_rate']:.2f} per guest, less a deposit credit of "
            f"${invoice['deposit_credited']:.2f}, for a balance due of ${balance_due:.2f}.\n\n"
            f"Reconciliation notes:\n{discrepancy_lines}\n\n"
            f"We're happy to answer any questions about these figures.\n\nBest regards,\nAurelia Billing Team"
        )
    # firm_policy_restatement
    return (
        f"Subject: Re: Invoice for {invoice['event_id']}\n\n"
        f"Hello,\n\nWe understand this is frustrating, but our records show the invoiced "
        f"balance of ${balance_due:.2f} reflects our standard per-guest rate applied to "
        f"your confirmed headcount, per our venue policy.\n\n"
        f"Reconciliation notes:\n{discrepancy_lines}\n\n"
        f"We're glad to discuss a payment plan, but the balance itself is accurate.\n\n"
        f"Regards,\nAurelia Billing Team"
    )


def _score_candidate(strategy: dict, state: BillingDisputeState) -> float:
    reconciliation = state["reconciliation"]
    negotiation_round = state.get("negotiation_round", 0)
    score = 0.0

    # Reward grounding the response in what reconciliation actually found.
    if reconciliation["discrepancies"] and strategy["name"] != "firm_policy_restatement":
        score += 2.0
    if not reconciliation["discrepancies"] and strategy["name"] == "firm_policy_restatement":
        score += 2.0  # No discrepancy found -> holding the line is well-supported.

    # Escalate warmth/concession the longer the dispute drags on, to avoid deadlock.
    score += strategy["concession_fraction"] * (negotiation_round + 1) * 5

    # Slight preference for leading with facts before offering anything away.
    if negotiation_round == 0 and strategy["name"] == "itemized_justification":
        score += 1.0

    return round(score, 3)

@with_error_handling("billing_dispute", "draft_dispute_email")
def draft_dispute_email(state: BillingDisputeState) -> dict:
    candidates = []
    for strategy in _STRATEGIES:
        body = _draft_email_body(strategy, state)
        score = _score_candidate(strategy, state)
        candidates.append(
            {"strategy": strategy["name"], "tone": strategy["tone"], "score": score, "body": body}
        )

    best = max(candidates, key=lambda c: c["score"])
    return {
        "candidate_emails": candidates,
        "draft_email": best["body"],
        "negotiation_round": state.get("negotiation_round", 0) + 1,
    }


# ---------------------------------------------------------------------------
# NODES: escalate_to_finance / human_finance_review -- HITL
# ---------------------------------------------------------------------------
# escalate_to_finance runs immediately and opens an admin_tickets row. The
# compiled graph uses interrupt_before=["human_finance_review"], so
# execution pauses right after escalate_to_finance and before
# human_finance_review actually runs. A finance admin reviews the open
# ticket and calls resume_after_finance_review(...); only then does
# human_finance_review run and the graph continue.

def escalate_to_finance(state: BillingDisputeState) -> dict:
    snapshot = {
        "invoice": state.get("invoice"),
        "reconciliation": state.get("reconciliation"),
        "negotiation_round": state.get("negotiation_round", 0),
        "candidate_emails": state.get("candidate_emails", []),
        "client_feedback": state.get("client_feedback"),
    }

    # Opened through the same helper the other two graphs use, so the ticket
    # lands in the canonical 'pending_admin' status and is idempotent under
    # interrupt_before's replay.
    ticket_id = open_hitl_ticket(
        graph_id=GRAPH_ID,
        thread_id=state["event_id"],
        reason=(
            f"Client rejected the final invoice for {state['event_id']}: "
            f"{state.get('client_feedback') or 'no reason given'}"
        ),
        state_snapshot=json.dumps(snapshot, default=str),
        db_path=DB_PATH,
    )

    return {"escalation_ticket_id": ticket_id}


def human_finance_review(state: BillingDisputeState) -> dict:
    decision = state.get("finance_decision")
    if not decision:
        # Should not normally happen: submit_admin_decision() (via
        # state_graph/hitl.py) sets this before resuming the graph past the
        # interrupt_before breakpoint.
        raise RuntimeError(
            "human_finance_review reached without a finance_decision set. "
            "Resolve this ticket via state_graph.hitl.submit_admin_decision(...) "
            "instead of resuming this graph directly."
        )

    # Ticket resolution (status -> 'resolved', decision, resolved_at) is now
    # handled centrally by submit_admin_decision() after this node returns,
    # the same way it is for vip_dietary_agent and vendor_logistics.
    return {"resolution": f"Finance admin decision on {state['event_id']}: {decision}"}


# ---------------------------------------------------------------------------
# NODE: finalize_billing
# ---------------------------------------------------------------------------

def finalize_billing(state: BillingDisputeState) -> dict:
    resolution = state.get("resolution")
    if not resolution:
        resolution = f"Client accepted the invoice for {state['event_id']} as billed."

    # This graph runs after an event has already taken place, so overwriting
    # `events.status` here reflects the post-event billing phase rather than
    # the pre-event booking lifecycle mcp_server/server.py manages.
    new_status = "BILLING_ESCALATED" if state.get("escalation_ticket_id") else "BILLING_RESOLVED"

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("UPDATE events SET status = ? WHERE event_id = ?", (new_status, state["event_id"]))
    conn.commit()
    conn.close()

    return {"resolution": resolution}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
# Used both as the graph's conditional entry point (so a fresh invoke() can
# jump straight to the right step of an already-persisted thread) and as the
# conditional edge out of reconcile_ledger (so a first-ever invoke() that
# already supplies client_status can run straight through in one call).

def _route_next_step(state: BillingDisputeState) -> str:
    if not state.get("invoice"):
        return "generate_invoice"
    if not state.get("reconciliation", {}).get("completed"):
        return "reconcile_ledger"

    status = state.get("client_status", "PENDING")
    if status == "ACCEPTED":
        return "finalize_billing"
    if status == "REJECTED_FINAL":
        return "escalate_to_finance"
    if status == "DISPUTED":
        if state.get("negotiation_round", 0) >= MAX_NEGOTIATION_ROUNDS:
            # Negotiation is deadlocked; escalate rather than loop forever.
            return "escalate_to_finance"
        return "draft_dispute_email"
    # PENDING: invoice + reconciliation are ready, awaiting the client's response.
    return "end_turn"


_ROUTE_MAP = {
    "generate_invoice": "generate_invoice",
    "reconcile_ledger": "reconcile_ledger",
    "draft_dispute_email": "draft_dispute_email",
    "escalate_to_finance": "escalate_to_finance",
    "finalize_billing": "finalize_billing",
    "end_turn": END,
}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_billing_dispute_graph(checkpointer):
    graph = StateGraph(BillingDisputeState)

    graph.add_node("generate_invoice", generate_invoice)
    graph.add_node("reconcile_ledger", reconcile_ledger)
    graph.add_node("draft_dispute_email", draft_dispute_email)
    graph.add_node("escalate_to_finance", escalate_to_finance)
    graph.add_node("human_finance_review", human_finance_review)
    graph.add_node("finalize_billing", finalize_billing)

    graph.set_conditional_entry_point(_route_next_step, _ROUTE_MAP)
    graph.add_edge("generate_invoice", "reconcile_ledger")
    graph.add_conditional_edges("reconcile_ledger", _route_next_step, _ROUTE_MAP)
    graph.add_edge("draft_dispute_email", END)
    graph.add_edge("escalate_to_finance", "human_finance_review")
    graph.add_edge("human_finance_review", "finalize_billing")
    graph.add_edge("finalize_billing", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["human_finance_review"])


# ---------------------------------------------------------------------------
# Public helpers for driving a persistent, multi-turn dispute thread
# ---------------------------------------------------------------------------

async def run_turn(compiled_graph, event_id: str, **updates) -> dict:
    """Run one turn of the billing-dispute graph for `event_id`.

    `updates` are merged into the persisted thread state before this turn
    runs, e.g.:
        await run_turn(graph, "EVT_999", client_status="DISPUTED",
                        client_feedback="This headcount looks too high.")
    """
    config = {"configurable": {"thread_id": event_id}}
    payload = {"event_id": event_id, **updates}
    return await compiled_graph.ainvoke(payload, config)


async def resume_after_finance_review(compiled_graph, event_id: str, decision: str) -> dict:
    """Resume a graph paused at the human_finance_review breakpoint.

    `decision` is a short free-text summary of what the finance admin
    decided (e.g. "Approved a $200 goodwill write-off").
    """
    config = {"configurable": {"thread_id": event_id}}
    await compiled_graph.aupdate_state(config, {"finance_decision": decision})
    return await compiled_graph.ainvoke(None, config)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _print_state(label: str, state: dict) -> None:
    print(f"\n--- {label} ---")
    print(json.dumps(state, indent=2, default=str))


async def run_demo() -> None:
    from state_graph.checkpointer import get_checkpointer

    async with get_checkpointer() as checkpointer:
        graph = build_billing_dispute_graph(checkpointer)
        event_id = "EVT_999"

        state = await run_turn(graph, event_id)
        _print_state("Turn 1: invoice generated + ledger reconciled", state)

        state = await run_turn(
            graph, event_id,
            client_status="DISPUTED",
            client_feedback="This looks too high for what we agreed.",
        )
        _print_state("Turn 2: client disputed -> Tree-of-Thoughts drafted a negotiation email", state)
        chosen = next(c for c in state["candidate_emails"] if c["body"] == state["draft_email"])
        print(f"[ToT] Selected strategy: {chosen['strategy']} (score={chosen['score']})")

        state = await run_turn(
            graph, event_id,
            client_status="REJECTED_FINAL",
            client_feedback="I'm not paying this. Final answer.",
        )
        _print_state("Turn 3: client explicitly rejected the final invoice -> escalated & paused", state)

        snapshot = await graph.aget_state({"configurable": {"thread_id": event_id}})
        print(f"\n[HITL] Graph paused before: {snapshot.next}")
        print(f"[HITL] Open ticket: {state['escalation_ticket_id']}")

        state = await resume_after_finance_review(
            graph, event_id, "Approved a $200 goodwill write-off; balance settled by phone."
        )
        _print_state("Turn 4: finance admin resolved the ticket, graph resumed to completion", state)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_demo())