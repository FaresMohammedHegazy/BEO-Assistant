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
from state_graph.tickets import open_hitl_ticket
from state_graph.recovery import with_error_handling

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

@with_error_handling("vendor_logistics", "research_and_plan")
def research_and_plan(state: VendorLogisticsState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Node 1: Pulls vendor policies via RAG, then decomposes the logistics goal into a plan."""
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

@with_error_handling("vendor_logistics", "draft_and_send")
async def draft_and_send(state: VendorLogisticsState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Node 2: Drafts the vendor request using MCP and sends it out."""
    # Example MCP integration to trigger external communication
    async with open_mcp_session() as mcp:
        # Assuming a generic notification/email tool exists in your actual MCP server
        pass
        
    return {"status": "waiting_for_vendor_reply"}

@with_error_handling("vendor_logistics", "wait_for_vendor_reply")
def wait_for_vendor_reply(state: VendorLogisticsState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Node 3: The Graph pauses BEFORE this node. It runs once a webhook resumes the graph."""
    if not state.get("vendor_reply"):
        raise ValueError("Graph resumed but vendor_reply is missing.")
        
    # HITL Condition Evaluation: Check if proposal exceeds budget
    proposal_amount = state.get("vendor_proposal_amount", 0.0)
    if proposal_amount > state["budget"]:
        # The graph is about to pause (interrupt_before=["hitl_approval"])
        # immediately after this node returns, so open the pending_admin ticket
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

@with_error_handling("vendor_logistics", "hitl_approval")
def hitl_approval(state: VendorLogisticsState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Node 4: The Graph pauses BEFORE this node. It processes the Admin's explicit decision."""
    if state.get("admin_approved") is True:
        return {"status": "ready_to_finalize"}
    return {"status": "rejected"}

@with_error_handling("vendor_logistics", "finalize")
def finalize(state: VendorLogisticsState, config: Dict[str, Any]) -> Dict[str, Any]:
    """Node 5: Finalizes the vendor logistics contract."""
    # Wrap up logic / Database writes here
    return {"status": "completed"}

# --- Routing Functions ---
def route_after_wait(state: VendorLogisticsState) -> str:
    if state["status"] == "hitl_approval_required": return "hitl"
    return "finalize"

def route_after_hitl(state: VendorLogisticsState) -> str:
    if state["status"] == "rejected": return "end"
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
    
    def route_entry(state: VendorLogisticsState) -> str:
        # If the graph is already finished, don't restart it!
        if state.get("status") in ["completed", "rejected"]:
            return "end"
        return "research_and_plan"

    workflow.set_conditional_entry_point(route_entry, {
        "end": END,
        "research_and_plan": "research_and_plan"
    })
    
    # We no longer need conditional failure edges because the exception decorator halts the graph
    workflow.add_edge("research_and_plan", "draft_and_send")
    workflow.add_edge("draft_and_send", "wait_for_vendor_reply")
    workflow.add_conditional_edges("wait_for_vendor_reply", route_after_wait, {"hitl": "hitl_approval", "finalize": "finalize"})
    workflow.add_conditional_edges("hitl_approval", route_after_hitl, {"finalize": "finalize", "end": END})
    workflow.add_edge("finalize", END)
    
    return workflow

# --- Compilation with Checkpointer ---
def compile_vendor_logistics_graph(checkpointer=None):
    return build_vendor_logistics_graph().compile(
        checkpointer=checkpointer,
        interrupt_before=["wait_for_vendor_reply", "hitl_approval"],
    )