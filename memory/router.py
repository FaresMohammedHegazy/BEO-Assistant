from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.short_term import Message


class MemoryRouter:
    def __init__(self, episodic_store):
        self.episodic_store = episodic_store
        self.decision_log = []

    def route_aging_item(self, item: "Message", context: dict) -> str:
        decision, reason = self._evaluate(item, context)
        self._log_decision(item, decision, reason, context)
        if decision == "promote":
            self.episodic_store.add_entry(item, reasoning=reason)
        return decision

    def _evaluate(self, item: "Message", context: dict) -> tuple[str, str]:
        scratchpad = context.get("scratchpad")
        content = item.content.lower()
        goal = (scratchpad.current_goal or "").lower()
        sub_goal = (scratchpad.active_sub_goal or "").lower()

        if item.metadata.get("persist"):
            return "promote", "Item explicitly marked to persist via metadata."

        if item.metadata.get("important"):
            return "promote", "Item metadata flagged as important."

        if item.role == "system":
            return "promote", "System message is treated as durable context."

        if any(key in item.metadata for key in ("event_id", "task_id", "step")):
            return "promote", "Item metadata contains structured event/task references."

        if goal and goal in content:
            return "promote", "Message content matches the current goal in scratchpad."

        if sub_goal and sub_goal in content:
            return "promote", "Message content matches the active sub-goal in scratchpad."

        if "decision" in content or "action item" in content or "next step" in content:
            return "promote", "Message appears to contain a decision or next step."

        return "forget", "Aging item is not judged relevant by the router."

    def _log_decision(self, item: "Message", decision: str, reason: str, context: dict):
        self.decision_log.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "item_preview": item.content[:120],
            "item_role": item.role,
            "decision": decision,
            "reason": reason,
            "scratchpad": {
                "current_goal": getattr(context.get("scratchpad"), "current_goal", None),
                "active_sub_goal": getattr(context.get("scratchpad"), "active_sub_goal", None),
            },
        })

    def get_decision_log(self) -> list[dict]:
        return list(self.decision_log)
