from memory.short_term import ShortTermMemory, Message

def test_scratchpad_survives_buffer_overflow():
    stm = ShortTermMemory(buffer_size=5)
    stm.update_scratchpad(active_sub_goal="Confirm EVT_999 deposit")
    for i in range(20):
        stm.add_message(Message(role="user", content=f"unrelated msg {i}"))
    assert stm.scratchpad.active_sub_goal == "Confirm EVT_999 deposit"
    assert len(stm.buffer) == 5

def test_scratchpad_survives_manual_prune():
    stm = ShortTermMemory(buffer_size=20)
    stm.update_scratchpad(active_sub_goal="Confirm EVT_999 deposit")
    for i in range(10):
        stm.add_message(Message(role="user", content=f"msg {i}"))
    stm.buffer.clear()
    assert stm.scratchpad.active_sub_goal == "Confirm EVT_999 deposit"
    assert len(stm.buffer) == 0