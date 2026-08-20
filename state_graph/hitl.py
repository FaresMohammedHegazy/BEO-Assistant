"""
Shared wiring between the admin_tickets table and the paused LangGraph
threads it represents.

state_graph/tickets.py knows how to read/write ticket rows but nothing
about LangGraph. This module is the other half: given a ticket_id and an
admin's decision, look up which graph paused it, resume that graph with
the decision, and mark the ticket resolved. It's the module the admin API
(platform/admin_api.py) calls into.

Each of the three state graphs pauses on its HITL node using a different
LangGraph mechanism, so each graph_id gets its own small resume() adapter:

  * vip_dietary_agent -- chef_signoff uses the dynamic `interrupt()`
    primitive, so it resumes via `ainvoke(Command(resume=...), config)`.
  * vendor_logistics -- hitl_approval is a static `interrupt_before` node,
    so it resumes by writing the decision into state with
    `aupdate_state(...)` and then continuing with `ainvoke(None, config)`.

A third graph (post-event billing, tracked separately) is not yet built --
see Issue #68. Wiring it in only requires adding one more entry to
GRAPH_REGISTRY below with the resume adapter that matches whichever
interrupt mechanism it ends up using; nothing else in this module changes.
"""
import json
from typing import Any, Awaitable, Callable, Optional

from langgraph.types import Command

from state_graph.checkpointer import get_checkpointer
from state_graph.tickets import get_ticket, resolve_ticket
from state_graph.vendor_logistics import compile_vendor_logistics_graph
from state_graph.vip_dietary import build_vip_dietary_graph

VALID_DECISIONS = ("approve", "reject", "modify")


async def _resume_vip_dietary(graph, config: dict, decision: str, payload: Optional[dict]) -> dict:
    resume_value = {"decision": decision, **(payload or {})}
    return await graph.ainvoke(Command(resume=resume_value), config=config)


async def _resume_vendor_logistics(graph, config: dict, decision: str, payload: Optional[dict]) -> dict:
    # hitl_approval only branches on `admin_approved`; treat "modify" as an
    # approval whose terms the admin has adjusted via `payload` (e.g. a
    # revised vendor_proposal_amount), and merge that into state alongside it.
    state_update = {"admin_approved": decision in ("approve", "modify")}
    state_update.update(payload or {})
    await graph.aupdate_state(config, state_update)
    return await graph.ainvoke(None, config=config)


# graph_id (as stored on the ticket row) -> how to build + resume that graph.
GRAPH_REGISTRY: dict[str, dict[str, Callable]] = {
    "vip_dietary_agent": {
        "build": lambda checkpointer: build_vip_dietary_graph(checkpointer=checkpointer),
        "resume": _resume_vip_dietary,
    },
    "vendor_logistics": {
        "build": lambda checkpointer: compile_vendor_logistics_graph(checkpointer=checkpointer),
        "resume": _resume_vendor_logistics,
    },
}


async def submit_admin_decision(ticket_id: str, decision: str,
                                 payload: Optional[dict] = None,
                                 db_path: Optional[str] = None,
                                 graph_registry: Optional[dict] = None) -> dict:
    """Resume the graph a pending_admin ticket represents with an admin's
    decision, then mark the ticket resolved.

    Args:
        ticket_id: the admin_tickets row to act on.
        decision: one of "approve", "reject", "modify".
        payload: optional extra fields (e.g. a modified proposal amount)
            merged into the graph's state alongside the decision.
        db_path: optional override of the aurelia.db path (tests only).
        graph_registry: optional override of GRAPH_REGISTRY (tests only),
            so callers can inject fake build/resume adapters instead of
            compiling the real graphs.

    Returns:
        The resumed graph's resulting state as a plain dict.

    Raises:
        ValueError: unknown decision, ticket not awaiting a decision, or no
            registered graph for the ticket's graph_id.
        LookupError: no ticket with that id.
    """
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Unknown decision {decision!r}; expected one of {VALID_DECISIONS}")

    ticket = get_ticket(ticket_id, db_path=db_path)
    if ticket is None:
        raise LookupError(f"No ticket with id {ticket_id!r}")
    if ticket["status"] != "pending_admin":
        raise ValueError(
            f"Ticket {ticket_id!r} is not awaiting an admin decision "
            f"(status={ticket['status']!r})"
        )

    registry = graph_registry if graph_registry is not None else GRAPH_REGISTRY
    graph_entry = registry.get(ticket["graph_id"])
    if graph_entry is None:
        raise ValueError(f"No resume handler registered for graph_id={ticket['graph_id']!r}")

    config = {
        "configurable": {
            "thread_id": ticket["thread_id"],
            "checkpoint_ns": ticket.get("checkpoint_ns") or "",
        }
    }

    async with get_checkpointer(db_path) as checkpointer:
        graph = graph_entry["build"](checkpointer)
        result_state = await graph_entry["resume"](graph, config, decision, payload)

    resolve_ticket(
        ticket_id,
        decision=decision,
        decision_payload=json.dumps(payload) if payload else None,
        db_path=db_path,
    )

    return dict(result_state) if result_state is not None else {}
