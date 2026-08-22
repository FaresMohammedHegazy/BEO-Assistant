"""
platform/chat_api.py

Issue #74 -- End-User Chat Interface with Agent Routing.

FastAPI router mounted at /api/chat (see main.py). Everything here is
in-process, in-memory session state, same scope as the rest of this lab
project (no Redis / external session store) -- see SessionStore below.

Endpoints
---------
GET    /api/chat/agents                      -- the agent switcher's catalog
POST   /api/chat/sessions                    -- start a chat with one agent
GET    /api/chat/sessions                    -- list sessions (optionally by agent_key)
GET    /api/chat/sessions/{session_id}       -- refresh a session's transcript
POST   /api/chat/sessions/{session_id}/message -- send a chat turn
DELETE /api/chat/sessions/{session_id}       -- end a session, release resources
"""
import os
import sys
from typing import Any, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import chat_agents as agents
from chat_sessions import SessionStore

router = APIRouter()
store = SessionStore()


class CreateSessionRequest(BaseModel):
    agent_key: str
    fields: dict[str, Any] = {}


class SendMessageRequest(BaseModel):
    message: str


@router.get("/agents")
def get_agents():
    return agents.AGENT_CATALOG


@router.get("/sessions")
async def list_sessions(agent_key: Optional[str] = None):
    return [s.to_dict() for s in await store.list(agent_key=agent_key)]


@router.post("/sessions")
async def create_session(body: CreateSessionRequest):
    if body.agent_key not in agents.AGENT_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown agent_key {body.agent_key!r}")
    try:
        session = await store.create(body.agent_key, body.fields)
    except (RuntimeError, ValueError) as e:
        # e.g. GROQ_API_KEY missing (memory_rag/planning), or a
        # required structured field missing/invalid (the state graphs).
        raise HTTPException(status_code=400, detail=str(e))
    return session.to_dict()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = await store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # For the three state-graph agents, re-check the persisted thread so a
    # resolution that happened out-of-band (an admin decision via
    # platform/app/admin, or -- for billing_dispute -- nothing further
    # needed) shows up here even if the end user never sends another
    # message. This is what turns "paused" into a real chat notification
    # instead of a UI that just silently stops updating.
    await store.refresh(session)
    return session.to_dict()


@router.post("/sessions/{session_id}/message")
async def send_message(session_id: str, body: SendMessageRequest):
    session = await store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    try:
        await store.send_message(session, body.message)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return session.to_dict()


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    closed = await store.delete(session_id)
    if not closed:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}
