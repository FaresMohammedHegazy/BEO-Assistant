import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sqlite3

from rag.vector_store import VectorStore
from state_graph.tickets import get_ticket, list_tickets
from state_graph.hitl import submit_admin_decision
from state_graph.recovery import resume_from_ticket

router = APIRouter()
DB_PATH = os.path.join(REPO_ROOT, 'db', 'aurelia.db')
RAG_STORE_PATH = os.path.join(REPO_ROOT, 'db', 'rag_store.db') 

class ToolToggle(BaseModel):
    agent_name: str
    tool_name: str
    is_active: bool

class RAGDocument(BaseModel):
    text: str
    metadata: dict = {}

class AdminDecision(BaseModel):
    decision: str  # "approve" | "reject" | "modify"
    payload: dict | None = None

@router.get("/tools")
def get_tools():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT agent_name, tool_name, is_active FROM agent_tools")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

@router.post("/tools/toggle")
def toggle_tool(toggle: ToolToggle):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE agent_tools SET is_active = ? WHERE agent_name = ? AND tool_name = ?",
        (1 if toggle.is_active else 0, toggle.agent_name, toggle.tool_name)
    )
    conn.commit()
    conn.close()
    return {"status": "success"}

@router.post("/rag/upload")
def upload_document(doc: RAGDocument):
    vector_store = VectorStore(store_path=RAG_STORE_PATH)
    doc_id = vector_store.add_document(text=doc.text, metadata=doc.metadata)
    return {"status": "success", "document_id": doc_id}

@router.delete("/rag/document/{doc_id}")
def delete_document(doc_id: str):
    vector_store = VectorStore(store_path=RAG_STORE_PATH)
    success = vector_store.delete_document(doc_id=doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success"}


@router.get("/tickets")
def get_tickets(status: str | None = None):
    """List admin_tickets rows, optionally filtered by status
    ('open', 'pending_admin', or 'resolved')."""
    return list_tickets(status=status, db_path=DB_PATH)


@router.get("/tickets/{ticket_id}")
def get_ticket_detail(ticket_id: str):
    ticket = get_ticket(ticket_id, db_path=DB_PATH)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@router.post("/tickets/{ticket_id}/decision")
async def post_ticket_decision(ticket_id: str, body: AdminDecision):
    """Submit an admin's Approve/Reject/Modify decision for a pending_admin
    ticket. Resumes the paused LangGraph thread the ticket represents and
    marks the ticket resolved."""
    try:
        result_state = await submit_admin_decision(
            ticket_id=ticket_id,
            decision=body.decision,
            payload=body.payload,
            db_path=DB_PATH,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="Ticket not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "resumed", "ticket_id": ticket_id, "graph_state": result_state}

@router.post("/tickets/{ticket_id}/resume")
async def post_ticket_resume(ticket_id: str):
    """Resume an unplanned failure ticket from its exact checkpoint."""
    try:
        result_state = await resume_from_ticket(ticket_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Ticket not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "resumed", "ticket_id": ticket_id, "graph_state": result_state}