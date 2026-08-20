"""
Tests for Issue #5: VIP Dietary Handoff State Graph.

LLM calls and MCP tool calls are injected as fakes, so this runs offline
with no GROQ_API_KEY and no live mcp_server subprocess. Covers the three
paths the diagram promised: the internal retry loop (stock miss ->
backtrack), the mandatory HITL pause + resume, and the ticket exit when
the search space is exhausted.
"""
import os
import sqlite3
import tempfile
import unittest

from langgraph.types import Command

from state_graph.checkpointer import get_checkpointer
from state_graph.vip_dietary import build_vip_dietary_graph


def _make_temp_db(stock_overrides: dict[str, int]) -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE guests (
        guest_id TEXT PRIMARY KEY, name TEXT, vip_status BOOLEAN, dietary_restrictions TEXT)""")
    cur.execute("""CREATE TABLE safe_ingredients (
        ingredient_id TEXT PRIMARY KEY, name TEXT, is_nut_free BOOLEAN, is_vegan BOOLEAN)""")
    cur.execute("""CREATE TABLE admin_tickets (
        ticket_id TEXT PRIMARY KEY, graph_id TEXT, thread_id TEXT,
        status TEXT, state_snapshot TEXT, error_message TEXT,
        checkpoint_ns TEXT NOT NULL DEFAULT '', decision TEXT,
        decision_payload TEXT, created_at TEXT, resolved_at TEXT)""")

    cur.execute("INSERT INTO guests VALUES (?, ?, ?, ?)",
                ("GUEST_VIP_1", "Eleanor Vance", 1, "SEVERE NUT ALLERGY, VEGAN"))

    ingredients = [
        ("ING_1", "Quinoa", 1, 1),
        ("ING_4", "Chickpeas", 1, 1),
        ("ING_5", "Tofu", 1, 1),
    ]
    cur.executemany("INSERT INTO safe_ingredients VALUES (?, ?, ?, ?)", ingredients)
    conn.commit()
    conn.close()
    return path


async def _fake_llm_always_yes(prompt: str) -> str:
    return "YES"


def _fake_check_stock_factory(stock_by_name: dict[str, int]):
    async def _check_stock(name: str) -> int:
        return stock_by_name.get(name, 0)
    return _check_stock


class TestVipDietaryGraph(unittest.IsolatedAsyncioTestCase):

    async def test_retry_loop_then_hitl_pause_and_approve(self):
        db_path = _make_temp_db({})
        cp_path = db_path  # checkpoint tables live in the same file

        check_stock = _fake_check_stock_factory(
            {"Quinoa": 50, "Chickpeas": 0, "Tofu": 30}  # Chickpeas out of stock
        )

        async with get_checkpointer(cp_path) as checkpointer:
            graph = build_vip_dietary_graph(
                llm_generate=_fake_llm_always_yes,
                check_stock=check_stock,
                db_path=db_path,
                checkpointer=checkpointer,
            )
            config = {"configurable": {"thread_id": "test-thread-vip-1", "checkpoint_ns": ""}}

            await graph.ainvoke(
                {"event_id": "EVT_999", "guest_id": "GUEST_VIP_1", "_thread_id": "test-thread-vip-1"},
                config=config,
            )

            # __interrupt__ only shows up in invoke()/ainvoke() results on
            # LangGraph >=0.4.0. get_state() exposes pending interrupts on
            # every version, so we read it from there instead.
            state_snapshot = await graph.aget_state(config)
            pending_interrupts = [
                i for task in state_snapshot.tasks for i in task.interrupts
            ]

            # First candidate (Chickpeas, Quinoa) fails on stock, graph loops
            # internally, and only pauses once it lands on (Quinoa, Tofu).
            self.assertEqual(len(pending_interrupts), 1)
            payload = pending_interrupts[0].value
            self.assertEqual(payload["proposed_combo"], ["Quinoa", "Tofu"])

            # Issue #71: pausing on chef_signoff must write a pending_admin
            # ticket to admin_tickets so the admin dashboard has something
            # to show while the graph is paused.
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT status, graph_id, thread_id FROM admin_tickets"
            )
            rows = cur.fetchall()
            conn.close()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0], ("pending_admin", "vip_dietary_agent", "test-thread-vip-1"))

            resumed = await graph.ainvoke(Command(resume={"decision": "approve"}), config=config)
            self.assertEqual(resumed["status"], "CONFIRMED")
            self.assertIn("Quinoa", resumed["final_menu"])
            self.assertIn("Tofu", resumed["final_menu"])

            # chef_signoff's body replays on resume (LangGraph re-runs a
            # dynamic-interrupt node from the top), so open_hitl_ticket's
            # idempotency must have kept this at exactly one row rather than
            # inserting a second pending_admin ticket for the same pause.
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM admin_tickets")
            count = cur.fetchone()[0]
            conn.close()
            self.assertEqual(count, 1)

        os.remove(db_path)

    async def test_search_exhausted_raises_ticket(self):
        db_path = _make_temp_db({})
        check_stock = _fake_check_stock_factory({})  # everything reports 0 stock

        async with get_checkpointer(db_path) as checkpointer:
            graph = build_vip_dietary_graph(
                llm_generate=_fake_llm_always_yes,
                check_stock=check_stock,
                db_path=db_path,
                checkpointer=checkpointer,
            )
            config = {"configurable": {"thread_id": "test-thread-vip-2", "checkpoint_ns": ""}}

            result = await graph.ainvoke(
                {"event_id": "EVT_999", "guest_id": "GUEST_VIP_1", "_thread_id": "test-thread-vip-2"},
                config=config,
            )
            self.assertEqual(result["status"], "FAILED_TICKET")

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT status, graph_id FROM admin_tickets")
        row = cur.fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "open")
        self.assertEqual(row[1], "vip_dietary_agent")

        os.remove(db_path)


if __name__ == "__main__":
    unittest.main()