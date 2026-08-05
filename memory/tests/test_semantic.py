import asyncio
import os
from dotenv import load_dotenv

# Load environment variables before initializing the Groq client
load_dotenv()

from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from memory.consolidation import SemanticConsolidator
from memory.short_term import Message

async def demo_conflict_resolution():
    episodic = EpisodicStore()
    semantic = SemanticStore()
    consolidator = SemanticConsolidator(semantic, episodic)

    # 1. First event: Guest is identified as VEGAN
    episodic.add_entry(Message(role="user", content="Guest GUEST_VIP_1 requires a strictly VEGAN menu."), "Extracted from chat")
    
    # Run periodic pass
    await consolidator.run_consolidation_pass()

    # 2. Second event later on: Guest updates their restriction, causing a conflict
    episodic.add_entry(Message(role="user", content="GUEST_VIP_1 called back, they are no longer VEGAN, they are just PESCATARIAN now."), "Updated by user")
    
    # Run second periodic pass (this triggers the conflict resolution logic)
    await consolidator.run_consolidation_pass()

    # 3. Prove the old fact wasn't silently overwritten
    print("\n--- Final Semantic State ---")
    active = semantic.get_active_facts("GUEST_VIP_1")
    history = semantic.get_fact_history("GUEST_VIP_1", "dietary_restrictions")
    
    print(f"Active Restrictions: {active}")
    print(f"Total Versions Kept: {len(history)}")

if __name__ == "__main__":
    asyncio.run(demo_conflict_resolution())