"""
Tests for Issue #71: state_graph/hitl.py, which resumes a paused LangGraph
thread from an admin's decision on a pending_admin ticket and marks the
ticket resolved.

submit_admin_decision is tested against a fake graph_registry so it never
needs to compile the real vip_dietary / vendor_logistics graphs (which
pull in Groq/MCP/sentence-transformers). The per-graph resume adapters
(_resume_vip_dietary, _resume_vendor_logistics) are tested separately
against a minimal fake graph object that just records how it was called,
so we're still exercising the real Command/aupdate_state wiring.
"""
import os
import sqlite3
import tempfile
import unittest

from langgraph.types import Command

from state_graph.hitl import _resume_vendor_logistics, _resume_vip_dietary, submit_admin_decision
from state_graph.tickets import get_ticket, open_hitl_ticket


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
    """Records how ainvoke / aupdate_state were called instead of running
    a real LangGraph graph."""

    def __init__(self, result=None):
        self._result = result if result is not None else {"status": "done"}
        self.ainvoke_calls = []
        self.aupdate_state_calls = []

    async def ainvoke(self, input_value, config):
        self.ainvoke_calls.append((input_value, config))
        return self._result

    async def aupdate_state(self, config, values):
        self.aupdate_state_calls.append((config, values))


class TestResumeAdapters(unittest.IsolatedAsyncioTestCase):
    async def test_resume_vip_dietary_sends_command_with_decision_and_payload(self):
        graph = FakeGraph(result={"status": "CONFIRMED"})
        config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}

        result = await _resume_vip_dietary(graph, config, "approve", {"note": "looks good"})

        self.assertEqual(result, {"status": "CONFIRMED"})
        self.assertEqual(len(graph.ainvoke_calls), 1)
        sent_command, sent_config = graph.ainvoke_calls[0]
        self.assertIsInstance(sent_command, Command)
        self.assertEqual(sent_command.resume, {"decision": "approve", "note": "looks good"})
        self.assertEqual(sent_config, config)

    async def test_resume_vendor_logistics_approve_sets_admin_approved_true(self):
        graph = FakeGraph(result={"status": "completed"})
        config = {"configurable": {"thread_id": "t2", "checkpoint_ns": ""}}

        result = await _resume_vendor_logistics(graph, config, "approve", None)

        self.assertEqual(result, {"status": "completed"})
        self.assertEqual(graph.aupdate_state_calls, [(config, {"admin_approved": True})])
        self.assertEqual(graph.ainvoke_calls, [(None, config)])

    async def test_resume_vendor_logistics_reject_sets_admin_approved_false(self):
        graph = FakeGraph()
        config = {"configurable": {"thread_id": "t3", "checkpoint_ns": ""}}

        await _resume_vendor_logistics(graph, config, "reject", None)

        self.assertEqual(graph.aupdate_state_calls, [(config, {"admin_approved": False})])

    async def test_resume_vendor_logistics_modify_merges_payload_and_approves(self):
        graph = FakeGraph()
        config = {"configurable": {"thread_id": "t4", "checkpoint_ns": ""}}

        await _resume_vendor_logistics(
            graph, config, "modify", {"vendor_proposal_amount": 4200.0}
        )

        self.assertEqual(
            graph.aupdate_state_calls,
            [(config, {"admin_approved": True, "vendor_proposal_amount": 4200.0})],
        )


class TestSubmitAdminDecision(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db_path = _make_temp_db()

    def tearDown(self):
        os.remove(self.db_path)

    async def test_unknown_decision_raises_value_error(self):
        with self.assertRaises(ValueError):
            await submit_admin_decision("TICKET_x", "maybe", db_path=self.db_path)

    async def test_missing_ticket_raises_lookup_error(self):
        with self.assertRaises(LookupError):
            await submit_admin_decision("TICKET_missing", "approve", db_path=self.db_path)

    async def test_ticket_not_pending_raises_value_error(self):
        ticket_id = open_hitl_ticket(
            graph_id="vip_dietary_agent", thread_id="t5",
            reason="r", state_snapshot="{}", db_path=self.db_path,
        )
        from state_graph.tickets import resolve_ticket
        resolve_ticket(ticket_id, decision="approve", db_path=self.db_path)

        with self.assertRaises(ValueError):
            await submit_admin_decision(ticket_id, "approve", db_path=self.db_path)

    async def test_unregistered_graph_id_raises_value_error(self):
        ticket_id = open_hitl_ticket(
            graph_id="some_future_graph", thread_id="t6",
            reason="r", state_snapshot="{}", db_path=self.db_path,
        )
        with self.assertRaises(ValueError):
            await submit_admin_decision(ticket_id, "approve", db_path=self.db_path,
                                         graph_registry={})

    async def test_happy_path_resumes_graph_and_resolves_ticket(self):
        ticket_id = open_hitl_ticket(
            graph_id="vendor_logistics", thread_id="t7",
            reason="budget exceeded", state_snapshot="{}",
            checkpoint_ns="", db_path=self.db_path,
        )

        fake_graph = FakeGraph(result={"status": "completed"})
        calls = {}

        def fake_build(checkpointer, db_path=None):
            calls["built_with_checkpointer"] = checkpointer
            calls["built_with_db_path"] = db_path
            return fake_graph

        async def fake_resume(graph, config, decision, payload):
            calls["resume_args"] = (graph, config, decision, payload)
            return {"status": "completed"}

        registry = {
            "vendor_logistics": {"build": fake_build, "resume": fake_resume},
        }

        result = await submit_admin_decision(
            ticket_id, "approve", payload={"note": "ok"},
            db_path=self.db_path, graph_registry=registry,
        )

        self.assertEqual(result, {"status": "completed"})
        self.assertIn("built_with_checkpointer", calls)
        graph_arg, config_arg, decision_arg, payload_arg = calls["resume_args"]
        self.assertIs(graph_arg, fake_graph)
        self.assertEqual(config_arg["configurable"]["thread_id"], "t7")
        self.assertEqual(decision_arg, "approve")
        self.assertEqual(payload_arg, {"note": "ok"})

        ticket = get_ticket(ticket_id, db_path=self.db_path)
        self.assertEqual(ticket["status"], "resolved")
        self.assertEqual(ticket["decision"], "approve")
        self.assertIsNotNone(ticket["resolved_at"])


if __name__ == "__main__":
    unittest.main()
