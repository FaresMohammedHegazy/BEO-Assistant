from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

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
    def __init__(self, buffer_size: int = 20):
        self.buffer = deque(maxlen=buffer_size)   # rolling buffer
        self.scratchpad = ScratchpadState()        

    def add_message(self, message: Message):
        self.buffer.append(message)  # when it's full, the last message is automatically deleted

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
        # 
        ...