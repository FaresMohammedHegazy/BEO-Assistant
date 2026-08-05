from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.short_term import Message


class EpisodicStore:
    def __init__(self):
        self.entries = []

    def add_entry(self, message: "Message", reasoning: str):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "role": message.role,
            "content": message.content,
            "metadata": dict(message.metadata),
            "reasoning": reasoning,
            "consolidated": False
        }
        self.entries.append(entry)

    def get_entries(self) -> list[dict]:
        return list(self.entries)

    def clear(self):
        self.entries.clear()
