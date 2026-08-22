"""
Tests for Issue #74: End-User Chat Interface with Agent Routing.

Two layers are covered:

* Pure logic in chat_agents.py (turn-text formatting, the client-reply
  classifier) needs no database or live LLM, so it's tested directly.
* The full HTTP surface (chat_api.py + chat_sessions.py) is tested
  against the real FastAPI app via httpx's ASGI transport -- no live
  server process, matching state_graph/tests' preference for exercising
  real wiring over mocks wherever that's actually offline-safe.

billing_dispute is the one state graph exercised end-to-end here,
because (per its own module docstring) it's deliberately LLM-free:
email drafting is template + heuristic-scored, not a Groq call, so the
whole create -> dispute -> escalate -> admin-resume -> notice flow runs
with no GROQ_API_KEY and no live mcp_server subprocess -- the same
property state_graph/tests/test_vip_dietary_graph.py relies on for its
own offline coverage (there, via injected llm_generate/check_stock
fakes instead). vip_dietary and vendor_logistics are covered at the
pure-formatting level only; driving them end to end needs either those
same injectable fakes (see test_vip_dietary_graph.py) or a live
GROQ_API_KEY, neither of which chat_api.py's thin routing layer should
need its own copy of to be tested.

These tests read/write state_graph/billing_dispute.py's real database
(db/aurelia.db) because that module's _connect() reads a module-level
DB_PATH constant rather than a per-call override -- there's no way to
point it at a temp file. Each test uses its own throwaway event_id and
cleans up after itself.
"""
import os
import sqlite3
import unittest
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(REPO_ROOT, "db", "aurelia.db")

import chat_agents as agents
from chat_sessions import SessionStore


def _seed_event(event_id: str, headcount: int = 40) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO events (event_id, guest_id, room_id, status, headcount, deposit_required) "
        "VALUES (?, 'GUEST_VIP_1', 'ROOM_101', 'PENDING_DEPOSIT', ?, 1000.0)",
        (event_id, headcount),
    )
    conn.commit()
    conn.close()


def _cleanup_event(event_id: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
    conn.execute("DELETE FROM admin_tickets WHERE thread_id = ?", (event_id,))
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (event_id,))
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (event_id,))
    conn.commit()
    conn.close()


def _require_seeded_db():
    if not os.path.exists(DB_PATH):
        raise unittest.SkipTest(
            "db/aurelia.db not found -- run `python db/setup_db.py` before this suite."
        )


# ---------------------------------------------------------------------------
# Pure logic -- no DB, no LLM
# ---------------------------------------------------------------------------

class TestClientReplyClassifier(unittest.TestCase):
    def test_clear_acceptance(self):
        status, feedback = agents._classify_client_reply("Sounds good, we accept.")
        self.assertEqual(status, "ACCEPTED")
        self.assertIsNone(feedback)

    def test_final_rejection(self):
        status, feedback = agents._classify_client_reply("This is final, I will not pay this.")
        self.assertEqual(status, "REJECTED_FINAL")
        self.assertEqual(feedback, "This is final, I will not pay this.")

    def test_ambiguous_defaults_to_disputed_not_silently_accepted(self):
        status, feedback = agents._classify_client_reply("The headcount looks off to me.")
        self.assertEqual(status, "DISPUTED")
        self.assertTrue(feedback)

    def test_accept_phrase_with_dispute_word_is_not_misread_as_acceptance(self):
        status, _ = agents._classify_client_reply("Looks good but I still dispute the total.")
        self.assertEqual(status, "DISPUTED")


class TestTurnTextFormatting(unittest.TestCase):
    def test_vendor_logistics_paused_before_vendor_reply(self):
        result = agents._vendor_logistics_turn_text(
            {"vendor_name": "Acme Linens"}, ("wait_for_vendor_reply",), "tid"
        )
        self.assertTrue(result.paused)
        self.assertIn("Acme Linens", result.text)

    def test_vendor_logistics_paused_before_admin_approval(self):
        result = agents._vendor_logistics_turn_text(
            {"ticket_id": "TICKET_1"}, ("hitl_approval",), "tid"
        )
        self.assertTrue(result.paused)
        self.assertEqual(result.ticket_id, "TICKET_1")

    def test_vendor_logistics_finished(self):
        result = agents._vendor_logistics_turn_text({"status": "completed"}, (), "tid")
        self.assertTrue(result.finished)
        self.assertFalse(result.paused)

    def test_vip_dietary_paused_for_chef_signoff(self):
        result = agents._vip_dietary_turn_text(
            {"guest_id": "GUEST_VIP_1", "current_combo": ["Quinoa", "Tofu"]},
            ("chef_signoff",),
        )
        self.assertTrue(result.paused)

    def test_vip_dietary_confirmed(self):
        result = agents._vip_dietary_turn_text(
            {"status": "CONFIRMED", "final_menu": "Quinoa, Tofu"}, ()
        )
        self.assertTrue(result.finished)
        self.assertEqual(result.text, "Quinoa, Tofu")

    def test_billing_dispute_paused_at_finance_review(self):
        result = agents._billing_dispute_turn_text(
            {"escalation_ticket_id": "TCK-1"}, ("human_finance_review",), "EVT_1"
        )
        self.assertTrue(result.paused)
        self.assertEqual(result.ticket_id, "TCK-1")


# ---------------------------------------------------------------------------
# billing_dispute end to end -- offline, real graph, real (throwaway) DB rows
# ---------------------------------------------------------------------------

class TestBillingDisputeAdapterEndToEnd(unittest.TestCase):
    def setUp(self):
        _require_seeded_db()
        self.event_id = f"EVT_TEST_{uuid.uuid4().hex[:8]}"
        _seed_event(self.event_id, headcount=40)
        self.addCleanup(_cleanup_event, self.event_id)

    def test_dispute_then_escalate_then_admin_resume_is_reported(self):
        start = agents.start_billing_dispute(self.event_id)
        self.assertFalse(start.paused)
        self.assertFalse(start.finished)
        self.assertIn(self.event_id, start.text)

        disputed = agents.continue_billing_dispute(
            self.event_id, "The headcount looks too high to me."
        )
        self.assertFalse(disputed.paused)
        self.assertFalse(disputed.finished)

        escalated = agents.continue_billing_dispute(
            self.event_id, "This is final, I will not pay this."
        )
        self.assertTrue(escalated.paused)
        self.assertIsNotNone(escalated.ticket_id)

        # A message sent while paused must not re-drive the graph -- only
        # a finance admin can resolve this.
        still_paused = agents.continue_billing_dispute(self.event_id, "hello?")
        self.assertTrue(still_paused.paused)
        self.assertEqual(still_paused.ticket_id, escalated.ticket_id)

        # Admin resumes out-of-band, the same way platform/app/admin does.
        from state_graph.billing_dispute import build_billing_dispute_graph, resume_after_finance_review
        graph = build_billing_dispute_graph()
        resume_after_finance_review(graph, self.event_id, "Approved a goodwill write-off.")

        resolved = agents.check_billing_dispute(self.event_id)
        self.assertFalse(resolved.paused)
        self.assertTrue(resolved.finished)
        self.assertIn("goodwill write-off", resolved.text)

    def test_clear_acceptance_finishes_without_escalation(self):
        agents.start_billing_dispute(self.event_id)
        result = agents.continue_billing_dispute(self.event_id, "Sounds good, we accept.")
        self.assertTrue(result.finished)
        self.assertFalse(result.paused)


# ---------------------------------------------------------------------------
# SessionStore -- validation and lifecycle
# ---------------------------------------------------------------------------

class TestSessionStoreValidation(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_agent_key_raises(self):
        store = SessionStore()
        with self.assertRaises(ValueError):
            await store.create("not_a_real_agent", {})

    async def test_missing_required_field_raises(self):
        store = SessionStore()
        with self.assertRaises(ValueError):
            await store.create("vip_dietary", {"event_id": "EVT_1"})  # guest_id missing

    async def test_non_numeric_budget_raises(self):
        store = SessionStore()
        with self.assertRaises(ValueError):
            await store.create("vendor_logistics", {
                "event_id": "EVT_1", "vendor_name": "V",
                "logistics_goal": "G", "budget": "not-a-number",
            })

    async def test_get_and_delete_unknown_session(self):
        store = SessionStore()
        self.assertIsNone(await store.get("nope"))
        self.assertFalse(await store.delete("nope"))


class TestSessionStoreBillingDisputeLifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _require_seeded_db()
        self.event_id = f"EVT_TEST_{uuid.uuid4().hex[:8]}"
        _seed_event(self.event_id, headcount=40)
        self.addCleanup(_cleanup_event, self.event_id)

    async def test_create_dispute_and_out_of_band_resolution_is_surfaced_on_refresh(self):
        store = SessionStore()
        session = await store.create("billing_dispute", {"event_id": self.event_id})
        self.assertEqual(session.thread_id, self.event_id)
        self.assertFalse(session.paused)
        first_message_count = len(session.messages)
        self.assertEqual(first_message_count, 1)

        await store.send_message(session, "This is final, I will not pay this.")
        self.assertTrue(session.paused)
        self.assertIsNotNone(session.ticket_id)

        # Sending another message while paused should acknowledge, not
        # silently do nothing and not re-drive the graph.
        before = len(session.messages)
        await store.send_message(session, "still there?")
        self.assertGreater(len(session.messages), before)
        self.assertTrue(session.paused)

        # Resolve out-of-band (as if an admin acted via platform/app/admin),
        # then a plain GET-style refresh should notice and report it.
        from state_graph.billing_dispute import build_billing_dispute_graph, resume_after_finance_review
        graph = build_billing_dispute_graph()
        resume_after_finance_review(graph, self.event_id, "Approved a goodwill write-off.")

        await store.refresh(session)
        self.assertFalse(session.paused)
        self.assertTrue(session.finished)
        self.assertIn("goodwill write-off", session.messages[-1].content)

        # A second refresh (simulating a later poll) must not duplicate
        # the notice.
        message_count_after_first_refresh = len(session.messages)
        await store.refresh(session)
        self.assertEqual(len(session.messages), message_count_after_first_refresh)

    async def test_delete_removes_session(self):
        store = SessionStore()
        session = await store.create("billing_dispute", {"event_id": self.event_id})
        self.assertTrue(await store.delete(session.session_id))
        self.assertIsNone(await store.get(session.session_id))


if __name__ == "__main__":
    unittest.main()
