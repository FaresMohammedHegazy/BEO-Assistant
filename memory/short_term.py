from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from typing import Optional

from memory.router import MemoryRouter

@dataclass
class Message:
    role: str        
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)  

@dataclass
class ScratchpadState:
    current_goal: str = ""
    active_sub_goal: str = ""
    working_facts: dict = field(default_factory=dict)  # ex:{"event_id": "EVT_999", "step": 2}
    last_updated: datetime = field(default_factory=datetime.utcnow)

class ShortTermMemory:
    def __init__(self, buffer_size: int = 20, router: Optional[MemoryRouter] = None):
        self.buffer = deque()
        self.buffer_size = buffer_size
        self.scratchpad = ScratchpadState()
        self.router = router

    def add_message(self, message: Message):
        self.buffer.append(message)
        self._trim_overflowing_items()

    def _trim_overflowing_items(self):
        while len(self.buffer) > self.buffer_size:
            item = self.buffer.popleft()
            if self.router is not None:
                self.router.route_aging_item(item, self.get_context())

    def update_scratchpad(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.scratchpad, k, v)
        self.scratchpad.last_updated = datetime.utcnow()

    def get_context(self) -> dict:
        return {
            "recent_messages": list(self.buffer),
            "scratchpad": self.scratchpad  
        }

    def overflowing_items(self) -> list[Message]:
        return list(self.buffer)[self.buffer_size:]
