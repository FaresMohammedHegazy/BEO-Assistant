"""
Tests for Issue #71: wiring vendor_logistics' HITL node to admin_tickets.

wait_for_vendor_reply is exercised directly as a plain async function
(rather than through the full compiled graph) so this runs without a live
RAG vector store or MCP server -- research_and_plan and draft_and_send
aren't involved in the HITL path this issue is about.
"""
import json
import os
import sqlite3
import tempfile
import unittest

from state_graph.vendor_logistics import wait_for_vendor_reply
from state_graph.tickets import list_tickets


def _make_temp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE admin_tickets (
        ticket_id TEXT PRIMARY KEY, graph_id TEXT, thread_id TEXT,
        status TEXT, state_snapshot TEXT, error_message TEXT,
        checkpoint_ns TEXT NOT NULL DEFAULT '', decision TEXT,
        decision_payload TEXT, created_at TEXT, resolved_at TEXT)""")
    conn.commit()
    conn.close()
    return path


class TestWaitForVendorReplyHitlTicket(unittest.TestCase):
    def setUp(self):
        self.db_path = _make_temp_db()
        # wait_for_vendor_reply calls open_hitl_ticket() with the module's
        # DEFAULT_DB_PATH unless we point it at our temp db.
        import state_graph.tickets as tickets_module
        self._orig_default = tickets_module.DEFAULT_DB_PATH
        tickets_module.DEFAULT_DB_PATH = self.db_path

    def tearDown(self):
        import state_graph.tickets as tickets_module
        tickets_module.DEFAULT_DB_PATH = self._orig_default
        os.remove(self.db_path)

    def test_proposal_over_budget_opens_pending_admin_ticket(self):
        state = {
            "thread_id": "vendor-thread-1",
            "vendor_name": "Acme Linens",
            "budget": 1000.0,
            "vendor_reply": "Here's our quote.",
            "vendor_proposal_amount": 5000.0,
        }
        config = {"configurable": {"thread_id": "vendor-thread-1", "checkpoint_ns": ""}}

        result = wait_for_vendor_reply(state, config)

        self.assertEqual(result["status"], "hitl_approval_required")
        self.assertIn("ticket_id", result)

        tickets = list_tickets(status="pending_admin", db_path=self.db_path)
        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets[0]["graph_id"], "vendor_logistics")
        self.assertEqual(tickets[0]["thread_id"], "vendor-thread-1")
        self.assertEqual(tickets[0]["ticket_id"], result["ticket_id"])

    def test_proposal_within_budget_does_not_open_a_ticket(self):
        state = {
            "thread_id": "vendor-thread-2",
            "vendor_name": "Acme Linens",
            "budget": 5000.0,
            "vendor_reply": "Here's our quote.",
            "vendor_proposal_amount": 1000.0,
        }
        config = {"configurable": {"thread_id": "vendor-thread-2", "checkpoint_ns": ""}}

        result = wait_for_vendor_reply(state, config)

        self.assertEqual(result["status"], "ready_to_finalize")
        self.assertEqual(list_tickets(db_path=self.db_path), [])

    def test_repeated_over_budget_call_reuses_same_ticket(self):
        state = {
            "thread_id": "vendor-thread-3",
            "vendor_name": "Acme Linens",
            "budget": 1000.0,
            "vendor_reply": "Here's our quote.",
            "vendor_proposal_amount": 5000.0,
        }
        config = {"configurable": {"thread_id": "vendor-thread-3", "checkpoint_ns": ""}}

        first = wait_for_vendor_reply(state, config)
        second = wait_for_vendor_reply(state, config)

        self.assertEqual(first["ticket_id"], second["ticket_id"])
        self.assertEqual(len(list_tickets(db_path=self.db_path)), 1)


if __name__ == "__main__":
    unittest.main()
