import os
import json
from groq import AsyncGroq
from memory.semantic_store import SemanticStore
from memory.episodic_store import EpisodicStore

class SemanticConsolidator:
    def __init__(self, semantic_store: SemanticStore, episodic_store: EpisodicStore):
        self.semantic_store = semantic_store
        self.episodic_store = episodic_store
        self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
        
    async def run_consolidation_pass(self, max_age_days: int = 90):
        """
        Periodically scans the episodic store for un-consolidated entries
        and extracts established facts into the semantic store.
        """
        unconsolidated = [e for e in self.episodic_store.entries if not e.get("consolidated", False)]
        
        if not unconsolidated:
            print("[Consolidation] No new episodes to consolidate.")
            return
            
        print(f"[Consolidation] Running pass over {len(unconsolidated)} new episodes...")
        
        # Prepare the log text for the LLM
        text_to_process = ""
        for idx, entry in enumerate(unconsolidated):
            text_to_process += f"[{entry['timestamp']}] {entry['role'].upper()}: {entry['content']}\n"
            
        prompt = f"""
        Analyze the following recent episodic memory logs. Extract any established, durable facts 
        about entities (like specific Events, Guests, or Rooms).
        
        Return the results as a JSON object with a single key "facts" containing a list of objects.
        Each object must have:
        - "entity_id": The exact identifier (e.g., "EVT_999", "GUEST_VIP_1")
        - "attribute": The property being established (e.g., "headcount", "dietary_restrictions", "status")
        - "value": The new or updated value
        - "reasoning": Why this fact was extracted or how it changed based on the logs
        
        Logs to process:
        {text_to_process}
        """
        
        response = await self.groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            facts = result.get("facts", [])
            
            for fact in facts:
                self.semantic_store.update_fact(
                    entity_id=fact["entity_id"],
                    attribute=fact["attribute"],
                    value=fact["value"],
                    episode_ref=unconsolidated[-1]["timestamp"],
                    reasoning=fact["reasoning"]
                )
        except Exception as e:
            print(f"[Consolidation] Error parsing LLM output: {e}")
            
        # Mark as consolidated so they aren't processed again
        for entry in unconsolidated:
            entry["consolidated"] = True

        # Periodic expiration check: demote any active fact that hasn't been
        # reinforced or contradicted in a while, instead of trusting it forever.
        self.semantic_store.expire_stale_facts(max_age_days=max_age_days)

        print("[Consolidation] Pass complete.")