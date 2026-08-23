"""
platform/chat_sessions.py

In-process, in-memory chat session registry for the end-user chat
interface (Issue #74). No external session store -- consistent with the
rest of this lab project (PlanningRouter's decision log, the admin
platform's request-scoped DB connections, etc. are all in-process too).

One ChatSession per "conversation" the switcher UI opens. For the three
state-graph agents the session's thread_id IS the LangGraph thread_id
that state_graph/checkpointer.py persists to db/aurelia.db, so a session
surviving only in this process's memory is fine -- the graph's actual
progress (including anything an admin does via platform/app/admin while
this process is even restarted) always lives in the database, and
refresh()/send_message() re-read it rather than trusting anything cached
here.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import chat_agents as agents


def _require(fields: dict, name: str) -> str:
    value = fields.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"'{name}' is required to start this agent.")
    return value


def _require_float(fields: dict, name: str) -> float:
    raw = _require(fields, name)
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number.")


@dataclass
class ChatSession:
    session_id: str
    agent_key: str
    thread_id: Optional[str]
    fields: dict[str, Any]
    messages: list[agents.ChatMessage] = field(default_factory=list)
    paused: bool = False
    finished: bool = False
    ticket_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    engine: Any = None  # MemoryRagEngine | PlanningEngine | None (state graphs need none)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "agent_key": self.agent_key,
            "thread_id": self.thread_id,
            "fields": self.fields,
            "messages": [m.to_dict() for m in self.messages],
            "paused": self.paused,
            "finished": self.finished,
            "ticket_id": self.ticket_id,
            "created_at": self.created_at,
        }


def _apply_turn_result(session: ChatSession, result: agents.TurnResult) -> None:
    session.messages.append(agents.ChatMessage("assistant", result.text))
    session.paused = result.paused
    session.finished = result.finished
    if result.ticket_id:
        session.ticket_id = result.ticket_id


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._registry_lock = asyncio.Lock()

    async def create(self, agent_key: str, fields: dict[str, Any]) -> ChatSession:
        session_id = uuid.uuid4().hex
        fields = fields or {}

        if agent_key == "memory_rag":
            engine = agents.MemoryRagEngine()  # raises RuntimeError if GROQ_API_KEY is bad/missing
            session = ChatSession(session_id, agent_key, thread_id=None, fields={}, engine=engine)
            session.messages.append(agents.ChatMessage(
                "system",
                "You're chatting with Aurelia's Memory & RAG concierge. Ask about hotel "
                "policies, facts on file for a guest, or day-to-day event questions.",
            ))

        elif agent_key == "planning":
            mode = fields.get("mode", "decomposition")
            max_steps = int(fields.get("max_steps") or 8)
            engine = agents.PlanningEngine(mode=mode, max_steps=max_steps)
            session = ChatSession(
                session_id, agent_key, thread_id=None,
                fields={"mode": engine.mode, "max_steps": max_steps}, engine=engine,
            )
            session.messages.append(agents.ChatMessage(
                "system",
                "Describe a multi-step goal and I'll decompose it into a task plan and "
                "execute it against the live hotel-operations tools.",
            ))

        elif agent_key == "vip_dietary":
            event_id = _require(fields, "event_id")
            guest_id = _require(fields, "guest_id")
            thread_id = agents.new_thread_id()
            result = await agents.start_vip_dietary(thread_id, event_id, guest_id)
            session = ChatSession(
                session_id, agent_key, thread_id=thread_id,
                fields={"event_id": event_id, "guest_id": guest_id},
            )
            _apply_turn_result(session, result)

        elif agent_key == "vendor_logistics":
            event_id = _require(fields, "event_id")
            vendor_name = _require(fields, "vendor_name")
            logistics_goal = _require(fields, "logistics_goal")
            budget = _require_float(fields, "budget")
            thread_id = agents.new_thread_id()
            result = await agents.start_vendor_logistics(
                thread_id, event_id, vendor_name, logistics_goal, budget
            )
            session = ChatSession(
                session_id, agent_key, thread_id=thread_id,
                fields={
                    "event_id": event_id, "vendor_name": vendor_name,
                    "logistics_goal": logistics_goal, "budget": budget,
                },
            )
            _apply_turn_result(session, result)

        elif agent_key == "billing_dispute":
            event_id = _require(fields, "event_id")
            thread_id = event_id  # state_graph/billing_dispute.py: thread_id = event_id
            result = await agents.start_billing_dispute(event_id)
            session = ChatSession(
                session_id, agent_key, thread_id=thread_id, fields={"event_id": event_id},
            )
            _apply_turn_result(session, result)

        else:
            raise ValueError(f"Unknown agent_key {agent_key!r}")

        async with self._registry_lock:
            self._sessions[session_id] = session
        return session

    async def get(self, session_id: str) -> Optional[ChatSession]:
        async with self._registry_lock:
            return self._sessions.get(session_id)

    async def list(self, agent_key: Optional[str] = None) -> list[ChatSession]:
        async with self._registry_lock:
            sessions = list(self._sessions.values())
        if agent_key:
            sessions = [s for s in sessions if s.agent_key == agent_key]
        return sessions

    async def delete(self, session_id: str) -> bool:
        async with self._registry_lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        if session.engine is not None and hasattr(session.engine, "close"):
            await session.engine.close()
        return True

    async def _check(self, session: ChatSession) -> agents.TurnResult:
        if session.agent_key == "vip_dietary":
            return await agents.check_vip_dietary(session.thread_id)
        if session.agent_key == "vendor_logistics":
            return await agents.check_vendor_logistics(session.thread_id)
        if session.agent_key == "billing_dispute":
            return await agents.check_billing_dispute(session.thread_id)
        raise ValueError(f"{session.agent_key} has no out-of-band status to check")

    async def _check_and_report(self, session: ChatSession, *, user_prompted: bool) -> None:
        """Re-check a state-graph session's persisted thread. Only appends
        a new chat message when something has actually changed (a pause
        just resolved, or the graph just finished) -- so GET-polling
        clients don't get a duplicate "still paused" bubble every refresh
        -- unless the user just explicitly sent a message, in which case
        they get a short acknowledgment even if nothing changed.
        """
        if session.finished:
            if user_prompted:
                last_assistant = next(
                    (m for m in reversed(session.messages) if m.role == "assistant"), None
                )
                session.messages.append(agents.ChatMessage(
                    "assistant",
                    "This request is already finished"
                    + (f": {last_assistant.content}" if last_assistant else "."),
                ))
            return
        result = await self._check(session)
        became_unpaused = session.paused and not result.paused
        became_finished = (not session.finished) and result.finished
        if became_unpaused or became_finished or not session.paused:
            session.messages.append(agents.ChatMessage("assistant", result.text))
        elif user_prompted:
            ticket_note = f" (ticket {session.ticket_id})" if session.ticket_id else ""
            session.messages.append(agents.ChatMessage(
                "assistant",
                f"Still waiting on the same thing{ticket_note} -- I'll update you here as "
                "soon as it's resolved.",
            ))
        session.paused = result.paused
        session.finished = result.finished
        if result.ticket_id:
            session.ticket_id = result.ticket_id

    async def refresh(self, session: ChatSession) -> None:
        """Called on every GET /sessions/{id} so a resolution that
        happened out-of-band (an admin decision) shows up even if the
        end user never sends another message.
        """
        if session.agent_key in ("vip_dietary", "vendor_logistics", "billing_dispute"):
            async with session.lock:
                await self._check_and_report(session, user_prompted=False)

    async def send_message(self, session: ChatSession, text: str) -> None:
        async with session.lock:
            session.messages.append(agents.ChatMessage("user", text))

            if session.agent_key == "memory_rag":
                reply = await session.engine.send(text)
                session.messages.append(agents.ChatMessage("assistant", reply))
                return

            if session.agent_key == "planning":
                reply = await session.engine.send(text)
                session.messages.append(agents.ChatMessage("assistant", reply))
                return

            if session.agent_key == "billing_dispute":
                if session.finished:
                    last_assistant = next(
                        (m for m in reversed(session.messages) if m.role == "assistant"), None
                    )
                    session.messages.append(agents.ChatMessage(
                        "assistant",
                        "This billing thread is already resolved"
                        + (f": {last_assistant.content}" if last_assistant else "."),
                    ))
                    return
                result = await agents.continue_billing_dispute(session.thread_id, text)
                _apply_turn_result(session, result)
                return

            if session.agent_key in ("vip_dietary", "vendor_logistics"):
                # These two run autonomously start-to-finish/pause the
                # moment they're created -- there's no mid-flight "client
                # turn" the underlying graph accepts. A chat message here
                # just asks "what's the status?".
                await self._check_and_report(session, user_prompted=True)
                return

            raise ValueError(f"Unknown agent_key {session.agent_key!r}")
