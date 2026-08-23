import asyncio
import sys
import os
from dotenv import load_dotenv

# Ensure Python can find your project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from langchain_groq import ChatGroq
from state_graph.vendor_logistics import compile_vendor_logistics_graph
from state_graph.checkpointer import get_checkpointer

async def fire_webhook(event_id: str, proposed_amount: float):
    # 1. Attach the LLM to the config to satisfy LangGraph's strict state checks
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)
    config = {"configurable": {"thread_id": event_id, "llm": llm}}
    
    async with get_checkpointer() as checkpointer:
        graph = compile_vendor_logistics_graph(checkpointer)
        
        print(f"\n--- 🔍 Checking Graph State for Event: '{event_id}' ---")
        state = await graph.aget_state(config)
        print(f"Current Next Node: {state.next}")
        
        if not state.next or "wait_for_vendor_reply" not in state.next:
            print("\n❌ Error: The graph is NOT paused waiting for the vendor!")
            print("Start a FRESH event in the UI and wait for it to say 'waiting on their reply'.")
            return

        print(f"\n--- 🚀 Injecting Vendor Reply (${proposed_amount}) ---")
        
        # 2. Update the state and capture the NEW checkpoint configuration
        updated_config = await graph.aupdate_state(config, {
            "vendor_reply": "Yes, we can provide the linens.",
            "vendor_proposal_amount": proposed_amount
        })
        
        print("--- ⚡ Waking up the graph ---")
        
        # 3. Resume the graph flawlessly from the new checkpoint!
        await graph.ainvoke(None, updated_config)
        
        new_state = await graph.aget_state(config)
        print(f"\nNew Next Node: {new_state.next}")
        print(f"Graph Status: {new_state.values.get('status')}")
        
        if new_state.next and "hitl_approval" in new_state.next:
            print("\n✅ SUCCESS! The graph successfully routed to HITL Approval.")
            print("Go check your Admin UI -> Tickets tab!")
        else:
            print("\n⚠️ Hmm, it didn't route to the Admin Ticket.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simulate_vendor.py <EVENT_ID> <AMOUNT>")
    else:
        asyncio.run(fire_webhook(sys.argv[1], float(sys.argv[2])))