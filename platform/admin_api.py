from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sqlite3
import os
import sys

# Import VectorStore from your existing RAG module
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from rag.vector_store import VectorStore

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