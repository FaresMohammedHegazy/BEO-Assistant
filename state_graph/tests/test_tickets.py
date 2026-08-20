"""
Tests for Issue #71: the admin_tickets lifecycle helpers in
state_graph/tickets.py -- raise_ticket (failures, pre-existing), plus the
new open_hitl_ticket / get_ticket / list_tickets / resolve_ticket used to
wire HITL pauses to the ticketing database and API.
"""
import os
import sqlite3
import tempfile
import unittest

from state_graph.tickets import (
    get_ticket,
    list_tickets,
    open_hitl_ticket,
    raise_ticket,
    resolve_ticket,
)


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


class TestTickets(unittest.TestCase):
    def setUp(self):
        self.db_path = _make_temp_db()

    def tearDown(self):
        os.remove(self.db_path)

    def test_raise_ticket_still_opens_a_failure_ticket(self):
        ticket_id = raise_ticket(
            graph_id="vendor_logistics",
            thread_id="thread-1",
            error_message="boom",
            state_snapshot="{}",
            db_path=self.db_path,
        )
        ticket = get_ticket(ticket_id, db_path=self.db_path)
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket["status"], "open")
        self.assertEqual(ticket["error_message"], "boom")

    def test_open_hitl_ticket_creates_pending_admin_row(self):
        ticket_id = open_hitl_ticket(
            graph_id="vip_dietary_agent",
            thread_id="thread-2",
            reason="chef sign-off needed",
            state_snapshot="{}",
            db_path=self.db_path,
        )
        ticket = get_ticket(ticket_id, db_path=self.db_path)
        self.assertEqual(ticket["status"], "pending_admin")
        self.assertEqual(ticket["graph_id"], "vip_dietary_agent")
        self.assertEqual(ticket["thread_id"], "thread-2")
        self.assertIsNotNone(ticket["created_at"])

    def test_open_hitl_ticket_is_idempotent_per_graph_and_thread(self):
        first_id = open_hitl_ticket(
            graph_id="vip_dietary_agent", thread_id="thread-3",
            reason="first call", state_snapshot="{}", db_path=self.db_path,
        )
        second_id = open_hitl_ticket(
            graph_id="vip_dietary_agent", thread_id="thread-3",
            reason="replayed call", state_snapshot="{}", db_path=self.db_path,
        )
        self.assertEqual(first_id, second_id)
        self.assertEqual(len(list_tickets(db_path=self.db_path)), 1)

    def test_open_hitl_ticket_after_resolution_opens_a_new_one(self):
        first_id = open_hitl_ticket(
            graph_id="vip_dietary_agent", thread_id="thread-4",
            reason="round 1", state_snapshot="{}", db_path=self.db_path,
        )
        resolve_ticket(first_id, decision="reject", db_path=self.db_path)

        second_id = open_hitl_ticket(
            graph_id="vip_dietary_agent", thread_id="thread-4",
            reason="round 2", state_snapshot="{}", db_path=self.db_path,
        )
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(len(list_tickets(db_path=self.db_path)), 2)

    def test_resolve_ticket_records_decision_and_timestamp(self):
        ticket_id = open_hitl_ticket(
            graph_id="vendor_logistics", thread_id="thread-5",
            reason="budget exceeded", state_snapshot="{}", db_path=self.db_path,
        )
        updated = resolve_ticket(
            ticket_id, decision="approve", decision_payload='{"note": "ok"}',
            db_path=self.db_path,
        )
        self.assertEqual(updated["status"], "resolved")
        self.assertEqual(updated["decision"], "approve")
        self.assertEqual(updated["decision_payload"], '{"note": "ok"}')
        self.assertIsNotNone(updated["resolved_at"])

    def test_list_tickets_filters_by_status(self):
        open_hitl_ticket(graph_id="vip_dietary_agent", thread_id="t-a",
                          reason="r", state_snapshot="{}", db_path=self.db_path)
        raise_ticket(graph_id="vendor_logistics", thread_id="t-b",
                      error_message="e", state_snapshot="{}", db_path=self.db_path)

        pending = list_tickets(status="pending_admin", db_path=self.db_path)
        failed = list_tickets(status="open", db_path=self.db_path)
        self.assertEqual(len(pending), 1)
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(list_tickets(db_path=self.db_path)), 2)

    def test_get_ticket_returns_none_for_unknown_id(self):
        self.assertIsNone(get_ticket("TICKET_doesnotexist", db_path=self.db_path))


if __name__ == "__main__":
    unittest.main()
