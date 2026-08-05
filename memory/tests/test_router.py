import unittest

from memory.episodic_store import EpisodicStore
from memory.router import MemoryRouter
from memory.short_term import ShortTermMemory, Message


class MemoryRouterTests(unittest.TestCase):
    def test_router_promotes_relevant_item_to_episodic_store(self):
        store = EpisodicStore()
        router = MemoryRouter(episodic_store=store)
        stm = ShortTermMemory(buffer_size=3, router=router)
        stm.update_scratchpad(active_sub_goal="Confirm EVT_999 deposit")

        stm.add_message(Message(role="user", content="Confirm EVT_999 deposit"))
        stm.add_message(Message(role="user", content="second message"))
        stm.add_message(Message(role="user", content="third message"))

        # This message will overflow the buffer and should be evaluated by the router.
        stm.add_message(Message(role="user", content="unrelated overflowing message"))

        self.assertEqual(len(stm.buffer), 3)
        self.assertEqual(len(store.get_entries()), 1)
        self.assertEqual(store.get_entries()[0]["content"], "Confirm EVT_999 deposit")
        self.assertEqual(
            store.get_entries()[0]["reasoning"],
            "Message content matches the active sub-goal in scratchpad.",
        )
        self.assertEqual(router.get_decision_log()[0]["decision"], "promote")

    def test_router_drops_irrelevant_aging_items(self):
        store = EpisodicStore()
        router = MemoryRouter(episodic_store=store)
        stm = ShortTermMemory(buffer_size=2, router=router)

        stm.add_message(Message(role="user", content="keep me"))
        stm.add_message(Message(role="user", content="also keep me"))
        stm.add_message(Message(role="user", content="old irrelevant item"))

        self.assertEqual(len(stm.buffer), 2)
        self.assertEqual(len(store.get_entries()), 0)
        self.assertEqual(router.get_decision_log()[0]["decision"], "forget")
        self.assertIn("not judged relevant", router.get_decision_log()[0]["reason"].lower())
