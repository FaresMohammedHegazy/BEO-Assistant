"""
platform/chat_agents.py

Issue #74 -- End-User Chat Interface with Agent Routing.

Thin adapters that let platform/chat_api.py drive any of the five backend
agents (Memory/RAG, Planning, and the three state-graph agents) through
one shared "start a session, send a message" shape, instead of the
terminal scripts (agent/client.py, agent/planning_client.py,
state_graph/*.py's own __main__ demos) an end user was previously forced
to run directly.

Nothing in those modules is modified here -- every adapter below only
calls the public functions/classes they already expose, the same way
platform/admin_api.py and state_graph/hitl.py already do.

Two different "shapes" of agent are wired in:

* memory_rag / planning are free-form chat: any message is a new
  instruction/goal, answered directly.
* vip_dietary / vendor_logistics / billing_dispute are the three
  LangGraph state graphs from state_graph/. They are started with a
  structured payload (an event/guest/vendor id, not a free-text goal)
  and then mostly run autonomously -- the chat's job for those three is
  almost entirely to *report* what the graph did and to notice, on each
  poll, whether a pause has been resolved out-of-band (an admin decision
  via platform/admin_api.py's ticket endpoints, or -- for billing_dispute
  only -- the client's own next chat message). See TurnResult below.
"""
from __future__ import annotations

import os
import sys
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")


# ---------------------------------------------------------------------------
# Shared result/message shapes
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """What a single agent turn produced, in chat-UI terms.

    `paused` and `finished` are mutually informative, not mutually
    exclusive with every field: a graph that just paused is neither
    "finished" nor able to take another chat turn until something
    external (an admin decision, a vendor reply) resolves it.
    """
    text: str
    paused: bool = False
    finished: bool = False
    ticket_id: Optional[str] = None


@dataclass
class ChatMessage:
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "created_at": self.created_at}


# ---------------------------------------------------------------------------
# Agent catalog -- what the end-user chat's agent switcher shows
# ---------------------------------------------------------------------------

AGENT_CATALOG: list[dict[str, Any]] = [
    {
        "key": "memory_rag",
        "label": "Memory & RAG Concierge",
        "description": (
            "Free-form chat grounded in Aurelia's policy knowledge base and "
            "durable guest memory."
        ),
        "kind": "freeform",
        "fields": [],
    },
    {
        "key": "planning",
        "label": "Planning Agent",
        "description": (
            "Decomposes a multi-step goal into a task DAG and executes it "
            "against the live hotel-operations tools."
        ),
        "kind": "freeform",
        "fields": [],
    },
    {
        "key": "vip_dietary",
        "label": "VIP Dietary Handoff",
        "description": (
            "Finds an allergy-safe menu pairing for a VIP guest and pauses "
            "for mandatory executive chef sign-off before confirming."
        ),
        "kind": "structured",
        "fields": [
            {"name": "event_id", "label": "Event ID", "example": "EVT_999"},
            {"name": "guest_id", "label": "Guest ID", "example": "GUEST_VIP_1"},
        ],
    },
    {
        "key": "vendor_logistics",
        "label": "Vendor Logistics",
        "description": (
            "Plans event vendor logistics and sends the request; pauses to "
            "wait for the vendor's reply, then for admin approval if the "
            "quote is over budget."
        ),
        "kind": "structured",
        "fields": [
            {"name": "event_id", "label": "Event ID", "example": "EVT_999"},
            {"name": "vendor_name", "label": "Vendor name", "example": "Acme Linens"},
            {"name": "logistics_goal", "label": "What do you need from them?",
             "example": "Confirm linens delivery for 250 guests by 9am."},
            {"name": "budget", "label": "Budget (USD)", "example": "1000"},
        ],
    },
    {
        "key": "billing_dispute",
        "label": "Post-Event Billing",
        "description": (
            "Generates the post-event invoice, then negotiates a dispute "
            "with you directly in chat -- escalating to finance if it can't "
            "be resolved."
        ),
        "kind": "structured",
        "fields": [
            {"name": "event_id", "label": "Event ID", "example": "EVT_999"},
        ],
    },
]

AGENT_KEYS = {a["key"] for a in AGENT_CATALOG}

# chat-routing key -> the graph_id string the state graphs/tickets actually use
_GRAPH_ID = {
    "vip_dietary": "vip_dietary_agent",
    "vendor_logistics": "vendor_logistics",
    "billing_dispute": "billing_dispute",
}


def new_thread_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# memory_rag -- long-lived engine wrapping agent/client.py's BEODemoAgent
# ---------------------------------------------------------------------------

class MemoryRagEngine:
    """One live MCP session shared across every turn of one chat session.

    Mirrors agent/client.py's run_main_demo() connection lifecycle (a
    single mcp_server subprocess kept open for the life of the
    conversation) instead of paying subprocess-spawn cost on every
    message.

    Deliberately does NOT declare the 'elicitation' client capability and
    never calls authenticate_director itself. mcp_server/server.py only
    exposes confirm_event_booking (the one tool that blocks on human
    elicitation) to a session that is both director-authenticated *and*
    elicitation-capable -- see its run_fallback_demo docstring. An
    end-user chat session simply never negotiates that capability, so it
    gets the same safe, read-mostly tool surface the fallback demo
    verifies, with no risk of this HTTP backend blocking on a human
    approval it has no way to collect synchronously. Booking confirmation
    stays an admin/director action taken through platform/app/admin.
    """

    def __init__(self) -> None:
        # Deferred import so importing this module doesn't require
        # GROQ_API_KEY / mcp to be installed unless a memory_rag session
        # is actually created.
        from agent.client import BEODemoAgent

        self.agent = BEODemoAgent()  # raises RuntimeError if GROQ_API_KEY is missing/malformed
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session = None

    async def connect(self) -> None:
        if self._session is not None:
            return
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env=os.environ.copy(),
            cwd=REPO_ROOT,
        )
        self._exit_stack = AsyncExitStack()
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream, sampling_callback=self._handle_sampling)
        )
        await self._session.initialize()
        tools_result = await self._session.list_tools()
        self.agent.tools_available = tools_result.tools

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._session = None

    async def _handle_sampling(self, context, params):
        """Answer mcp_server's plain (non-elicitation) sampling requests,
        e.g. draft_custom_menu reasoning over verified-safe ingredients.
        Mirrors the "standard sampling" branch of
        agent/client.py's run_main_demo.handle_sampling.
        """
        import mcp.types as types

        prompt = params.messages[0].content.text
        response = await self.agent.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text=response.choices[0].message.content),
            model=MODEL_NAME,
        )

    async def send(self, user_text: str) -> str:
        if self._session is None:
            await self.connect()
        await self.agent.chat_with_groq(user_text, self._session)
        # chat_with_groq's last write to the short-term buffer is always
        # the turn's final assistant message (see agent/client.py); read
        # it back instead of changing that method's (print-only) contract.
        buffer = self.agent.stm.buffer
        if buffer and buffer[-1].role == "assistant":
            return buffer[-1].content
        return ""


# ---------------------------------------------------------------------------
# planning -- stateless engine wrapping agent/planning_agent_executor.py
# ---------------------------------------------------------------------------

class PlanningEngine:
    """Each chat message is treated as a fresh natural-language goal,
    decomposed and executed against the live MCP tools -- the same entry
    point agent/planning_client.py's CLI drives, just without the
    terminal.
    """

    def __init__(self, mode: str = "decomposition", max_steps: int = 8) -> None:
        self.mode = mode if mode in ("decomposition", "dynamic") else "decomposition"
        self.max_steps = max_steps

    async def send(self, user_text: str) -> str:
        from agent.planning_agent_executor import PlanningAgentExecutor

        async with PlanningAgentExecutor(model_name=MODEL_NAME) as executor:
            if self.mode == "dynamic":
                result = await executor.run_dynamic(user_text, max_steps=self.max_steps)
            else:
                result = await executor.run_decomposition_first(user_text)
        return _format_plan_result(result)


def _format_plan_result(result) -> str:
    lines = [f"Plan ({result.method}) for: {result.goal}", ""]
    for step in result.steps:
        tag = f"-> {step.tool_name}" if step.tool_name else "(reasoning)"
        lines.append(f"- {step.instruction} {tag}")
        if step.output:
            lines.append(f"  {step.output}")
    lines.append("")
    lines.append(result.final_answer or "(no final answer produced)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# vip_dietary -- state_graph/vip_dietary.py
# ---------------------------------------------------------------------------

async def start_vip_dietary(thread_id: str, event_id: str, guest_id: str,
                             **build_kwargs: Any) -> TurnResult:
    from state_graph.checkpointer import get_checkpointer
    from state_graph.vip_dietary import build_vip_dietary_graph

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    async with get_checkpointer() as checkpointer:
        graph = build_vip_dietary_graph(checkpointer=checkpointer, **build_kwargs)
        try:
            values = await graph.ainvoke(
                {"event_id": event_id, "guest_id": guest_id, "_thread_id": thread_id},
                config=config,
            )
        except (RuntimeError, ValueError) as e:
            return TurnResult(text=f"Couldn't start the VIP dietary handoff: {e}", finished=True)
        snapshot = await graph.aget_state(config)
    return _vip_dietary_turn_text(dict(values), snapshot.next)


async def check_vip_dietary(thread_id: str) -> TurnResult:
    from state_graph.checkpointer import get_checkpointer
    from state_graph.vip_dietary import build_vip_dietary_graph

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    async with get_checkpointer() as checkpointer:
        graph = build_vip_dietary_graph(checkpointer=checkpointer)
        snapshot = await graph.aget_state(config)
    return _vip_dietary_turn_text(dict(snapshot.values), snapshot.next)


def _vip_dietary_turn_text(values: dict, next_nodes: tuple) -> TurnResult:
    if next_nodes:
        return TurnResult(
            text=(
                f"I've found a candidate menu pairing for {values.get('guest_id', 'the guest')}: "
                f"{values.get('current_combo')}. Every VIP allergy-safe menu change needs "
                "mandatory executive chef sign-off before it's confirmed -- I've opened an "
                "approval ticket for that, and I'll update you here as soon as the chef decides."
            ),
            paused=True,
        )
    status = values.get("status")
    if status == "CONFIRMED":
        return TurnResult(text=values.get("final_menu", "Menu confirmed."), finished=True)
    if status == "FAILED_TICKET":
        return TurnResult(
            text=(
                "I couldn't find a safe, in-stock, chef-approved pairing for this guest from "
                "the candidate ingredients on file. I've raised a ticket for a human to take "
                f"a look: {values.get('ticket_reason', 'no safe combination found.')}"
            ),
            finished=True,
        )
    return TurnResult(text=f"Working on it (status: {status or 'starting'}).")


# ---------------------------------------------------------------------------
# vendor_logistics -- state_graph/vendor_logistics.py
# ---------------------------------------------------------------------------

async def start_vendor_logistics(thread_id: str, event_id: str, vendor_name: str,
                                  logistics_goal: str, budget: float) -> TurnResult:
    from langchain_groq import ChatGroq

    from state_graph.checkpointer import get_checkpointer
    from state_graph.vendor_logistics import compile_vendor_logistics_graph

    llm = ChatGroq(model=MODEL_NAME, temperature=0.1)
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "", "llm": llm}}
    async with get_checkpointer() as checkpointer:
        graph = compile_vendor_logistics_graph(checkpointer=checkpointer)
        try:
            values = await graph.ainvoke(
                {
                    "thread_id": thread_id,
                    "event_id": event_id,
                    "vendor_name": vendor_name,
                    "logistics_goal": logistics_goal,
                    "budget": budget,
                },
                config=config,
            )
        except (RuntimeError, ValueError) as e:
            return TurnResult(text=f"Couldn't start vendor logistics: {e}", finished=True)
        snapshot = await graph.aget_state(config)
    return _vendor_logistics_turn_text(dict(values), snapshot.next, thread_id)


async def check_vendor_logistics(thread_id: str) -> TurnResult:
    from state_graph.checkpointer import get_checkpointer
    from state_graph.vendor_logistics import compile_vendor_logistics_graph

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    async with get_checkpointer() as checkpointer:
        graph = compile_vendor_logistics_graph(checkpointer=checkpointer)
        snapshot = await graph.aget_state(config)
    return _vendor_logistics_turn_text(dict(snapshot.values), snapshot.next, thread_id)


def _vendor_logistics_turn_text(values: dict, next_nodes: tuple, thread_id: str) -> TurnResult:
    if next_nodes:
        next_node = next_nodes[0]
        if next_node == "wait_for_vendor_reply":
            return TurnResult(
                text=(
                    f"I've put together the logistics plan and sent the request to "
                    f"{values.get('vendor_name', 'the vendor')}. I'm now waiting on their "
                    "reply -- I'll update you here as soon as they respond."
                ),
                paused=True,
            )
        if next_node == "hitl_approval":
            ticket_id = values.get("ticket_id")
            return TurnResult(
                text=(
                    "The vendor's proposal came back over budget, so this needs admin "
                    f"approval before it can go ahead (ticket {ticket_id}). "
                    "I'll let you know here as soon as it's approved or rejected."
                ),
                paused=True,
                ticket_id=ticket_id,
            )
        return TurnResult(text=f"Still working ({next_node} is next).", paused=True)
    status = values.get("status")
    if status == "completed":
        return TurnResult(text="The vendor logistics request is finalized.", finished=True)
    if status == "rejected":
        return TurnResult(text="An admin reviewed the over-budget proposal and rejected it.",
                           finished=True)
    return TurnResult(text=f"Working on it (status: {status or 'starting'}).")


# ---------------------------------------------------------------------------
# billing_dispute -- state_graph/billing_dispute.py
# ---------------------------------------------------------------------------
# The only one of the three state graphs where the *client's own chat
# messages* are what drives the graph forward turn to turn, so unlike
# vip_dietary/vendor_logistics this one has a real continue_* function,
# not just a status check.

_ACCEPT_PHRASES = (
    "i accept", "we accept", "accepted", "accept the invoice", "looks good",
    "sounds good", "that works for us", "i'll pay", "we'll pay", "agree to pay",
    "happy to pay", "ok to pay", "fine, i'll pay",
)
_FINAL_REJECT_PHRASES = (
    "final answer", "won't pay", "will not pay", "wont pay", "not paying",
    "refuse to pay", "absolutely not", "this is final", "final, i will not",
)


def _classify_client_reply(message: str) -> tuple[str, Optional[str]]:
    """Small keyword heuristic mapping the client's free-text chat reply
    onto billing_dispute's client_status enum -- deliberately simple and
    offline, in the same spirit as this graph's own _score_candidate /
    _draft_email_body (see the module docstring on why those are
    heuristic rather than LLM calls). A natural follow-up would swap this
    for a real Groq classification call the way agent/client.py's
    handle_sampling does, without changing the shape callers rely on.
    """
    text = (message or "").strip().lower()
    if any(p in text for p in _FINAL_REJECT_PHRASES):
        return "REJECTED_FINAL", message
    if any(p in text for p in _ACCEPT_PHRASES) and "dispute" not in text and "disagree" not in text:
        return "ACCEPTED", None
    # Default to DISPUTED rather than silently accepting anything
    # ambiguous -- an invoice should never be finalized on a guess.
    return "DISPUTED", message or "The client disputed the invoice."


def start_billing_dispute(event_id: str) -> TurnResult:
    from state_graph.billing_dispute import build_billing_dispute_graph, run_turn

    graph = build_billing_dispute_graph()
    try:
        values = run_turn(graph, event_id)
    except (RuntimeError, ValueError) as e:
        return TurnResult(text=f"Couldn't open the billing dispute thread: {e}", finished=True)
    snapshot = graph.get_state({"configurable": {"thread_id": event_id}})
    return _billing_dispute_turn_text(dict(values), snapshot.next, event_id)


def continue_billing_dispute(event_id: str, message: str) -> TurnResult:
    from state_graph.billing_dispute import (
        build_billing_dispute_graph,
        resume_after_finance_review,
        run_turn,
    )

    graph = build_billing_dispute_graph()
    config = {"configurable": {"thread_id": event_id}}
    snapshot = graph.get_state(config)

    if snapshot.next:
        # Paused at human_finance_review -- only a finance admin can
        # resolve this (platform/admin_api.py's /tickets endpoints), not
        # the client's chat message. Just report where things stand.
        return _billing_dispute_turn_text(dict(snapshot.values), snapshot.next, event_id)

    client_status, feedback = _classify_client_reply(message)
    updates: dict[str, Any] = {"client_status": client_status}
    if feedback:
        updates["client_feedback"] = feedback

    try:
        values = run_turn(graph, event_id, **updates)
    except (RuntimeError, ValueError) as e:
        return TurnResult(text=f"Something went wrong recording your response: {e}")
    snapshot = graph.get_state(config)
    return _billing_dispute_turn_text(dict(values), snapshot.next, event_id)


def check_billing_dispute(event_id: str) -> TurnResult:
    from state_graph.billing_dispute import build_billing_dispute_graph

    graph = build_billing_dispute_graph()
    snapshot = graph.get_state({"configurable": {"thread_id": event_id}})
    return _billing_dispute_turn_text(dict(snapshot.values), snapshot.next, event_id)


def _billing_dispute_turn_text(values: dict, next_nodes: tuple, event_id: str) -> TurnResult:
    if next_nodes:
        return TurnResult(
            text=(
                "I've escalated this to our finance team for review -- I wasn't able to "
                f"resolve it directly (ticket {values.get('escalation_ticket_id')}). "
                "I'll let you know here as soon as finance has made a decision."
            ),
            paused=True,
            ticket_id=values.get("escalation_ticket_id"),
        )
    if values.get("resolution"):
        return TurnResult(text=values["resolution"], finished=True)

    status = values.get("client_status")
    if status == "PENDING":
        invoice = values.get("invoice", {}) or {}
        text = (
            f"Here's the invoice for {event_id}: {invoice.get('headcount')} guests x "
            f"${invoice.get('per_guest_rate')}/guest = ${invoice.get('subtotal')}, "
            f"balance due ${invoice.get('balance_due')}.\n\n"
            "Let me know if you accept this invoice, or tell me what looks wrong and I'll "
            "put together a response."
        )
        discrepancies = (values.get("reconciliation") or {}).get("discrepancies") or []
        if discrepancies:
            text += "\n\nFor reference, our own reconciliation flagged: " + "; ".join(discrepancies)
        return TurnResult(text=text)
    if status == "DISPUTED":
        draft = values.get("draft_email", "")
        return TurnResult(
            text=(
                "Thanks -- here's how I'd like to respond to that:\n\n" + draft
                + "\n\nLet me know if that resolves it, or if you'd still like to dispute further."
            )
        )
    return TurnResult(text=f"Status: {status or 'starting'}.")
