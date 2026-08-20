import json
import functools
import traceback
from typing import Callable, Any

from state_graph.tickets import raise_ticket, get_ticket, resolve_ticket
from state_graph.checkpointer import get_checkpointer

def with_error_handling(graph_id: str, node_name: str):
    """
    Error-handling wrapper inside state_graph/ utilities.
    Catches unplanned exceptions (e.g., tool errors, LLM parsing failures),
    persists a failure ticket for admins, and halts execution so the 
    checkpointer preserves the pre-node state.
    """
    def decorator(func: Callable) -> Callable:
        import asyncio
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(state: dict, config: dict | None = None, **kwargs):
                try:
                    if config is not None:
                        return await func(state, config, **kwargs)
                    return await func(state, **kwargs)
                except Exception as e:
                    _record_failure(e, graph_id, node_name, state, config)
                    raise RuntimeError(f"Unplanned failure in {node_name}: {str(e)}") from e
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(state: dict, config: dict | None = None, **kwargs):
                try:
                    if config is not None:
                        return func(state, config, **kwargs)
                    return func(state, **kwargs)
                except Exception as e:
                    _record_failure(e, graph_id, node_name, state, config)
                    raise RuntimeError(f"Unplanned failure in {node_name}: {str(e)}") from e
            return sync_wrapper
    return decorator

def _record_failure(e: Exception, graph_id: str, node_name: str, state: dict, config: dict | None):
    thread_id = "unknown"
    if config and "configurable" in config:
        thread_id = config["configurable"].get("thread_id", "unknown")
    if thread_id == "unknown":
        thread_id = state.get("thread_id") or state.get("event_id") or state.get("_thread_id") or "unknown"
        
    error_msg = f"Exception in {node_name}: {str(e)}\n{traceback.format_exc()}"
    
    raise_ticket(
        graph_id=graph_id,
        thread_id=thread_id,
        error_message=error_msg,
        state_snapshot=json.dumps(state, default=str)
    )

async def resume_from_ticket(ticket_id: str) -> dict:
    """
    Retrieves the state of a failed ticket and re-invokes the graph.
    """
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise ValueError(f"Ticket {ticket_id} not found.")
        
    if ticket["status"] != "open":
        raise ValueError(f"Ticket {ticket_id} is not an open failure ticket.")
        
    graph_id = ticket["graph_id"]
    thread_id = ticket["thread_id"]
    
    # Resolve the failure ticket so it's cleared from the queue
    resolve_ticket(ticket_id, decision="Resumed after resolving unplanned failure")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Late imports to avoid circular dependencies
    from state_graph.billing_dispute import build_billing_dispute_graph
    from state_graph.vendor_logistics import compile_vendor_logistics_graph
    from state_graph.vip_dietary import build_vip_dietary_graph
    
    async with get_checkpointer() as checkpointer:
        if graph_id == "billing_dispute":
            graph = build_billing_dispute_graph() 
            # Use sync invoke() here since billing_dispute uses a sync SqliteSaver
            return graph.invoke(None, config) 
        elif graph_id == "vendor_logistics":
            graph = compile_vendor_logistics_graph(checkpointer)
            return await graph.ainvoke(None, config)
        elif graph_id == "vip_dietary_agent":
            graph = build_vip_dietary_graph(checkpointer=checkpointer)
            return await graph.ainvoke(None, config)
        else:
            raise ValueError(f"Unknown graph_id: {graph_id}")