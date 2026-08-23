"""
Tests for Issue #1: billing_dispute wasn't wired into the shared HITL
admin-resolution path (state_graph/hitl.py), and wrote non-canonical
admin_tickets.status values ('PENDING_FINANCE_REVIEW' / 'RESOLVED')
instead of using open_hitl_ticket() / resolve_ticket(). Both meant an
admin could not actually resolve a billing_dispute finance escalation
through platform/app/admin/tickets/page.jsx.

These tests avoid compiling the real billing_dispute LangGraph graph
(which needs a real db/aurelia.db and the Groq-backed nodes) wherever
possible, the same way test_hitl.py avoids the real vip_dietary /
vendor_logistics graphs: escalate_to_finance and human_finance_review are
plain node functions and are exercised directly, and _resume_billing_dispute
/ submit_admin_decision are exercised against a minimal FakeGraph that just
records how it was called.
"""
import os
import sqlite3
import tempfile
import unittest

import state_graph.billing_dispute as billing_dispute
from state_graph.billing_dispute import escalate_to_finance, human_finance_review
from state_graph.hitl import GRAPH_REGISTRY, _resume_billing_dispute, submit_admin_decision
from state_graph.tickets import get_ticket, list_tickets, open_hitl_ticket


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


class FakeGraph:
    """Records how aupdate_state / ainvoke were called, now that
    billing_dispute compiles against the shared AsyncSqliteSaver like the
    other two graphs (see the checkpointer-unification commit)."""

    def __init__(self, result=None):
        self._result = result if result is not None else {"status": "done"}
        self.ainvoke_calls = []
        self.aupdate_state_calls = []

    async def aupdate_state(self, config, values):
        self.aupdate_state_calls.append((config, values))

    async def ainvoke(self, input_value, config):
        self.ainvoke_calls.append((input_value, config))
        return self._result


class TestEscalateToFinanceOpensCanonicalTicket(unittest.TestCase):
    """escalate_to_finance() must open a 'pending_admin' ticket via
    open_hitl_ticket(), not a hand-rolled 'PENDING_FINANCE_REVIEW' row."""

    def setUp(self):
        self.db_path = _make_temp_db()
        self._orig_db_path = billing_dispute.DB_PATH
        billing_dispute.DB_PATH = self.db_path

    def tearDown(self):
        billing_dispute.DB_PATH = self._orig_db_path
        os.remove(self.db_path)

    def test_opens_pending_admin_ticket(self):
        state = {
            "event_id": "EVT_TEST_1",
            "invoice": {"total": 500.0},
            "reconciliation": {"discrepancy": 50.0},
            "negotiation_round": 3,
            "candidate_emails": [],
            "client_feedback": "This is too high.",
        }

        result = escalate_to_finance(state)

        ticket = get_ticket(result["escalation_ticket_id"], db_path=self.db_path)
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket["status"], "pending_admin")
        self.assertEqual(ticket["graph_id"], "billing_dispute")
        self.assertEqual(ticket["thread_id"], "EVT_TEST_1")
        self.assertIn("This is too high.", ticket["error_message"])

    def test_is_idempotent_like_the_other_two_graphs(self):
        state = {"event_id": "EVT_TEST_2", "client_feedback": "no"}

        first = escalate_to_finance(state)
        second = escalate_to_finance(state)

        self.assertEqual(first["escalation_ticket_id"], second["escalation_ticket_id"])
        tickets = list_tickets(status="pending_admin", db_path=self.db_path)
        self.assertEqual(len([t for t in tickets if t["thread_id"] == "EVT_TEST_2"]), 1)


class TestHumanFinanceReview(unittest.TestCase):
    def test_raises_without_finance_decision(self):
        with self.assertRaises(RuntimeError):
            human_finance_review({"event_id": "EVT_TEST_3"})

    def test_no_longer_touches_admin_tickets_directly(self):
        # Ticket resolution is now submit_admin_decision()'s job; this node
        # must not write to admin_tickets itself (regression guard for the
        # old hand-rolled `UPDATE admin_tickets SET status = 'RESOLVED'`).
        result = human_finance_review({
            "event_id": "EVT_TEST_4",
            "finance_decision": "Approved a $200 goodwill write-off",
        })
        self.assertIn("Approved a $200 goodwill write-off", result["resolution"])


class TestResumeBillingDisputeAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_writes_finance_decision_and_invokes_async(self):
        graph = FakeGraph(result={"resolution": "done"})
        config = {"configurable": {"thread_id": "EVT_TEST_5", "checkpoint_ns": ""}}

        result = await _resume_billing_dispute(graph, config, "approve", None)

        self.assertEqual(result, {"resolution": "done"})
        self.assertEqual(graph.aupdate_state_calls, [(config, {"finance_decision": "approve"})])
        self.assertEqual(graph.ainvoke_calls, [(None, config)])

    async def test_modify_payload_overrides_bare_decision(self):
        graph = FakeGraph()
        config = {"configurable": {"thread_id": "EVT_TEST_6", "checkpoint_ns": ""}}

        await _resume_billing_dispute(
            graph, config, "modify", {"finance_decision": "Approved a partial refund"}
        )

        self.assertEqual(
            graph.aupdate_state_calls,
            [(config, {"finance_decision": "Approved a partial refund"})],
        )

class TestSubmitAdminDecisionEndToEndForBillingDispute(unittest.IsolatedAsyncioTestCase):
    """The same path platform/admin_api.py's POST /tickets/{id}/decision
    calls -- proves a billing_dispute ticket is resolvable through the
    shared entry point without compiling the real graph."""

    def setUp(self):
        self.db_path = _make_temp_db()

    def tearDown(self):
        os.remove(self.db_path)

    async def test_billing_dispute_is_registered(self):
        self.assertIn("billing_dispute", GRAPH_REGISTRY)

    async def test_happy_path_resumes_and_resolves(self):
        ticket_id = open_hitl_ticket(
            graph_id="billing_dispute", thread_id="EVT_TEST_7",
            reason="Client rejected the final invoice",
            state_snapshot="{}", db_path=self.db_path,
        )

        fake_graph = FakeGraph(result={"resolution": "Approved a $200 write-off"})
        calls = {}

        def fake_build(checkpointer, db_path=None):
            calls["built_with_checkpointer"] = checkpointer
            return fake_graph

        registry = {
            "billing_dispute": {"build": fake_build, "resume": _resume_billing_dispute},
        }

        result = await submit_admin_decision(
            ticket_id, "approve", payload={"finance_decision": "Approved a $200 write-off"},
            db_path=self.db_path, graph_registry=registry,
        )

        self.assertEqual(result, {"resolution": "Approved a $200 write-off"})
        self.assertIn("built_with_checkpointer", calls)
        ticket = get_ticket(ticket_id, db_path=self.db_path)
        self.assertEqual(ticket["status"], "resolved")
        self.assertEqual(ticket["decision"], "approve")
        self.assertIsNotNone(ticket["resolved_at"])


if __name__ == "__main__":
    unittest.main()