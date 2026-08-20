import os
import json
import logging
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.language_models.chat_models import BaseChatModel

# Import Aurelia components mapped to your existing project structure
from rag.vector_store import VectorStore
from rag.retrievers import NaiveRAG
from planning.planning_lab.algorithms.decomposition import decompose_goal, execute_plan
from state_graph.mcp_client import open_mcp_session
from state_graph.checkpointer import DEFAULT_DB_PATH
from state_graph.tickets import raise_ticket, open_hitl_ticket

logger = logging.getLogger(__name__)

class VendorLogisticsState(TypedDict):
    """The unified state dictionary for the vendor logistics graph."""
    thread_id: str
    event_id: str
    vendor_name: str
    logistics_goal: str
    budget: float
    
    # LLM-Call Additions: RAG and Task Decomposition
    vendor_policies: List[str]
    logistics_plan_outputs: Dict[str, str]
    
    # External Wait State properties
    vendor_reply: Optional[str]
    vendor_proposal_amount: Optional[float]
    
    # Human-In-The-Loop properties
    admin_approved: Optional[bool]
    
    # Failure and Recovery properties
    status: str
    error_message: Optional[str]
    ticket_id: Optional[str]


def research_and_plan(state: VendorLogisticsState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Node 1: Pulls vendor policies via RAG, then decomposes the logistics goal into a plan."""
    try:
        llm: BaseChatModel = config["configurable"]["llm"]
        
        # 1. RAG Addition: Fetch vendor policies
        vs = VectorStore(store_path=DEFAULT_DB_PATH)
        rag = NaiveRAG(vector_store=vs)
        retrieved_docs = rag.retrieve(state["vendor_name"], top_k=3)
        policies = [doc["text"] for doc in retrieved_docs]
        
        # 2. Task Decomposition Addition: Break down the logistics goal
        # Injecting policies into the goal context to ground the planner
        augmented_goal = f"{state['logistics_goal']}\nVendor Policies Context:\n{policies}"
        plan = decompose_goal(augmented_goal, llm)
        plan_outputs = execute_plan(plan, llm)
        
        return {
            "vendor_policies": policies,
            "logistics_plan_outputs": plan_outputs,
            "status": "planning_complete"
        }
    except Exception as e:
        ticket_id = raise_ticket(
            graph_id="vendor_logistics",
            thread_id=state.get("thread_id", "unknown"),
            error_message=f"Research & Plan failed: {str(e)}",
            state_snapshot=json.dumps(state)
        )
        return {"status": "failed", "error_message": str(e), "ticket_id": ticket_id}


async def draft_and_send(state: VendorLogisticsState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Node 2: Drafts the vendor request using MCP and sends it out."""
    try:
        # Example MCP integration to trigger external communication
        async with open_mcp_session() as mcp:
            # Assuming a generic notification/email tool exists in your actual MCP server
            # Alternatively, if drafting heavily relies on MCP lookup tools, call them here.
            pass
            
        return {"status": "waiting_for_vendor_reply"}
    except Exception as e:
        ticket_id = raise_ticket(
            graph_id="vendor_logistics",
            thread_id=state.get("thread_id", "unknown"),
            error_message=f"Draft & Send failed: {str(e)}",
            state_snapshot=json.dumps(state)
        )
        return {"status": "failed", "error_message": str(e), "ticket_id": ticket_id}


def wait_for_vendor_reply(state: VendorLogisticsState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Node 3: The Graph pauses BEFORE this node. It runs once a webhook resumes the graph."""
    try:
        if not state.get("vendor_reply"):
            raise ValueError("Graph resumed but vendor_reply is missing.")
            
        # HITL Condition Evaluation: Check if proposal exceeds budget
        proposal_amount = state.get("vendor_proposal_amount", 0.0)
        if proposal_amount > state["budget"]:
            # The graph is about to pause (interrupt_before=["hitl_approval"])
            # immediately after this node returns, so open the pending_admin
            # ticket here -- this node only runs once per pause (unlike a
            # dynamic `interrupt()` node, a static interrupt_before pause
            # does not replay the node that led into it).
            checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
            ticket_id = open_hitl_ticket(
                graph_id="vendor_logistics",
                thread_id=state.get("thread_id", "unknown"),
                reason=(
                    f"Vendor proposal ${proposal_amount:,.2f} for "
                    f"{state.get('vendor_name', 'vendor')} exceeds budget "
                    f"${state['budget']:,.2f} -- needs admin approval."
                ),
                state_snapshot=json.dumps(state),
                checkpoint_ns=checkpoint_ns,
            )
            return {"status": "hitl_approval_required", "ticket_id": ticket_id}
            
        return {"status": "ready_to_finalize"}
    except Exception as e:
        ticket_id = raise_ticket(
            graph_id="vendor_logistics",
            thread_id=state.get("thread_id", "unknown"),
            error_message=f"Processing Reply failed: {str(e)}",
            state_snapshot=json.dumps(state)
        )
        return {"status": "failed", "error_message": str(e), "ticket_id": ticket_id}


def hitl_approval(state: VendorLogisticsState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Node 4: The Graph pauses BEFORE this node. It processes the Admin's explicit decision."""
    try:
        if state.get("admin_approved") is True:
            return {"status": "ready_to_finalize"}
        return {"status": "rejected"}
    except Exception as e:
        ticket_id = raise_ticket(
            graph_id="vendor_logistics",
            thread_id=state.get("thread_id", "unknown"),
            error_message=f"HITL processing failed: {str(e)}",
            state_snapshot=json.dumps(state)
        )
        return {"status": "failed", "error_message": str(e), "ticket_id": ticket_id}


def finalize(state: VendorLogisticsState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Node 5: Finalizes the vendor logistics contract."""
    try:
        # Wrap up logic / Database writes here
        return {"status": "completed"}
    except Exception as e:
        ticket_id = raise_ticket(
            graph_id="vendor_logistics",
            thread_id=state.get("thread_id", "unknown"),
            error_message=f"Finalize failed: {str(e)}",
            state_snapshot=json.dumps(state)
        )
        return {"status": "failed", "error_message": str(e), "ticket_id": ticket_id}


# --- Routing Functions ---
def route_after_research(state: VendorLogisticsState) -> str:
    return "end" if state["status"] == "failed" else "continue"

def route_after_draft(state: VendorLogisticsState) -> str:
    return "end" if state["status"] == "failed" else "continue"

def route_after_wait(state: VendorLogisticsState) -> str:
    if state["status"] == "failed": return "end"
    if state["status"] == "hitl_approval_required": return "hitl"
    return "finalize"

def route_after_hitl(state: VendorLogisticsState) -> str:
    if state["status"] == "failed" or state["status"] == "rejected": return "end"
    return "finalize"


# --- Graph Construction ---
def build_vendor_logistics_graph():
    """Builds and returns the uncompiled LangGraph workflow."""
    workflow = StateGraph(VendorLogisticsState)
    
    workflow.add_node("research_and_plan", research_and_plan)
    workflow.add_node("draft_and_send", draft_and_send)
    workflow.add_node("wait_for_vendor_reply", wait_for_vendor_reply)
    workflow.add_node("hitl_approval", hitl_approval)
    workflow.add_node("finalize", finalize)
    
    workflow.set_entry_point("research_and_plan")
    
    workflow.add_conditional_edges("research_and_plan", route_after_research, {"continue": "draft_and_send", "end": END})
    workflow.add_conditional_edges("draft_and_send", route_after_draft, {"continue": "wait_for_vendor_reply", "end": END})
    workflow.add_conditional_edges("wait_for_vendor_reply", route_after_wait, {"hitl": "hitl_approval", "finalize": "finalize", "end": END})
    workflow.add_conditional_edges("hitl_approval", route_after_hitl, {"finalize": "finalize", "end": END})
    workflow.add_edge("finalize", END)
    
    return workflow

# --- Compilation with Checkpointer ---
def compile_vendor_logistics_graph(checkpointer=None):
    """Compile the vendor logistics workflow against the shared checkpointer.

    Pauses before `wait_for_vendor_reply` (an external wait on the vendor's
    reply, resumed by a webhook once that arrives) and before
    `hitl_approval` (the admin-facing HITL node, resumed via
    state_graph.hitl.submit_admin_decision once an admin acts on the
    pending_admin ticket opened in `wait_for_vendor_reply`).

    Expected to be called within the
    `async with get_checkpointer() as checkpointer:` block from
    `state_graph/checkpointer.py`, e.g.:

        async with get_checkpointer() as checkpointer:
            graph = compile_vendor_logistics_graph(checkpointer)
            await graph.ainvoke(initial_state, config=config)
    """
    return build_vendor_logistics_graph().compile(
        checkpointer=checkpointer,
        interrupt_before=["wait_for_vendor_reply", "hitl_approval"],
    )