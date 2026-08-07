from datetime import datetime

class SemanticStore:
    def __init__(self):
        # Maps entity_id -> { attribute: [list of versioned facts] }
        # Example: {"GUEST_VIP_1": {"dietary_restrictions": [...]}}
        self.entities = {}

    def update_fact(self, entity_id: str, attribute: str, value: str, episode_ref: str, reasoning: str):
        """
        Adds a fact or creates a new version if it contradicts an existing fact.
        Guarantees no silent overwrites.
        """
        if entity_id not in self.entities:
            self.entities[entity_id] = {}
        
        if attribute not in self.entities[entity_id]:
            self.entities[entity_id][attribute] = []
        
        history = self.entities[entity_id][attribute]
        new_version = len(history) + 1
        
        fact_record = {
            "version": new_version,
            "value": value,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "episode_ref": episode_ref,
            "reasoning": reasoning,
            "status": "active"
        }
        
        # Conflict Resolution / Versioning
        if history:
            current_latest = history[-1]
            if current_latest["value"] != value:
                print(f"\n[SEMANTIC CONFLICT RESOLVED] Entity: {entity_id}")
                print(f"  Attribute '{attribute}' changed.")
                print(f"  Old Value (v{current_latest['version']}): {current_latest['value']} -> STATUS: superseded")
                print(f"  New Value (v{new_version}): {value} -> STATUS: active")
                current_latest["status"] = "superseded"
                self.entities[entity_id][attribute].append(fact_record)
            else:
                # Fact hasn't changed, no need to version
                pass
        else:
            self.entities[entity_id][attribute].append(fact_record)

    def get_active_facts(self, entity_id: str) -> dict:
        """Returns only the current active facts for an entity."""
        if entity_id not in self.entities:
            return {}
        
        active_facts = {}
        for attr, history in self.entities[entity_id].items():
            latest = history[-1]
            if latest["status"] == "active":
                active_facts[attr] = latest["value"]
        return active_facts

    def get_fact_history(self, entity_id: str, attribute: str) -> list:
        """Retrieves the full version history of a specific fact."""
        return self.entities.get(entity_id, {}).get(attribute, [])

    def expire_stale_facts(self, max_age_days: int = 90) -> list[dict]:
        """
        Walks every entity/attribute and demotes any 'active' fact whose
        timestamp is older than max_age_days to 'expired'. Nothing is deleted -
        the full version history (including the now-expired entry) stays intact,
        exactly like the 'superseded' status used for conflicts.
        Returns the list of facts that were just expired, for logging/demo purposes.
        """
        now = datetime.utcnow()
        newly_expired = []

        for entity_id, attributes in self.entities.items():
            for attribute, history in attributes.items():
                if not history:
                    continue
                latest = history[-1]
                if latest["status"] != "active":
                    continue  # already superseded or expired, nothing to do

                ts_str = latest["timestamp"]
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1]
                fact_time = datetime.fromisoformat(ts_str)
                age_days = (now - fact_time).total_seconds() / 86400

                if age_days > max_age_days:
                    latest["status"] = "expired"
                    newly_expired.append({
                        "entity_id": entity_id,
                        "attribute": attribute,
                        "value": latest["value"],
                        "age_days": round(age_days, 1),
                    })
                    print(f"\n[SEMANTIC EXPIRATION] Entity: {entity_id}")
                    print(f"  Attribute '{attribute}' (v{latest['version']}) is {age_days:.1f} days old "
                          f"(> {max_age_days} day limit) -> STATUS: expired")

        return newly_expired