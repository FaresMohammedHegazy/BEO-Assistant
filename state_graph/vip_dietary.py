"""
VIP Dietary Handoff State Graph

A menu-change request for a severe-allergy VIP guest must never be
finalized by the agent alone. This graph:
  1. Searches safe ingredient pairings with LATS (propose -> LLM-score ->
     backtrack), all internal to the agent -- no human involved yet.
  2. Verifies real DB stock for the winning pairing with a Constrained
     ReAct node that can only ever call check_ingredient_stock.
  3. Pauses on a MANDATORY human-in-the-loop node (chef_signoff) before
     any confirmation -- every pairing goes through this, not just
     high-risk ones.
  4. Raises a ticket, not a HITL pause, if the search space is fully
     exhausted -- that's a real failure a retry cannot fix.
"""
import itertools
import json
import os
import sqlite3
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

from state_graph.mcp_client import open_mcp_session
from state_graph.tickets import raise_ticket

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(REPO_ROOT, "db", "aurelia.db")


class VipDietaryState(TypedDict, total=False):
    event_id: str
    guest_id: str
    dietary_restrictions: str
    candidate_pool: list[str]
    tried_combos: list[list[str]]
    current_combo: list[str] | None
    stock_ok: bool | None
    failing_ingredient: str | None
    chef_decision: str | None
    final_menu: str | None
    status: str
    ticket_reason: str | None
    _thread_id: str


# ---------------------------------------------------------------------
# Default (real) dependencies -- overridable for tests
# ---------------------------------------------------------------------
async def _default_llm_generate(prompt: str) -> str:
    from groq import AsyncGroq
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    client = AsyncGroq(api_key=api_key)
    model = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100,
    )
    return response.choices[0].message.content


async def _default_check_stock(ingredient_name: str) -> int:
    import re
    async with open_mcp_session() as session:
        result = await session.call_tool(
            "check_ingredient_stock", arguments={"ingredient_name": ingredient_name}
        )
        text = result.content[0].text
        match = re.search(r"stock=(\d+)", text)
        return int(match.group(1)) if match else 0


# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------
def make_fetch_guest_constraints(db_path: str):
    def fetch_guest_constraints(state: VipDietaryState) -> dict:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT dietary_restrictions FROM guests WHERE guest_id = ?",
            (state["guest_id"],),
        )
        row = cursor.fetchone()
        cursor.execute("SELECT name, is_nut_free, is_vegan FROM safe_ingredients")
        ingredients = cursor.fetchall()
        conn.close()

        if not row:
            raise ValueError(f"Guest {state['guest_id']} not found.")

        restrictions = row[0].upper()
        needs_nut_free = "NUT" in restrictions
        needs_vegan = "VEGAN" in restrictions

        pool = sorted(
            name for name, is_nut_free, is_vegan in ingredients
            if (not needs_nut_free or is_nut_free) and (not needs_vegan or is_vegan)
        )

        return {
            "dietary_restrictions": row[0],
            "candidate_pool": pool,
            "tried_combos": [],
            "status": "SEARCHING",
        }

    return fetch_guest_constraints


def _all_combos(pool: list[str]) -> list[list[str]]:
    return [list(c) for c in itertools.combinations(pool, 2)]


def make_lats_search(llm_generate):
    async def lats_search(state: VipDietaryState) -> dict:
        pool = state["candidate_pool"]
        tried = list(state.get("tried_combos", []))
        untried = [c for c in _all_combos(pool) if c not in tried]

        if not untried:
            return {"status": "EXHAUSTED", "current_combo": None}

        # LATS: propose each remaining branch, score it with the LLM, and
        # backtrack to the next branch on rejection -- all inside this one
        # node call, no external wait involved.
        for candidate in untried:
            prompt = (
                f"Guest dietary restrictions: {state['dietary_restrictions']}.\n"
                f"Proposed two-course pairing: {candidate[0]} and {candidate[1]}.\n"
                "Does this make a coherent two-course banquet menu? "
                "Reply with exactly YES or NO on the first line."
            )
            verdict = await llm_generate(prompt)
            if verdict.strip().upper().startswith("YES"):
                return {"current_combo": candidate, "status": "SEARCHING"}
            tried.append(candidate)

        return {"status": "EXHAUSTED", "current_combo": None, "tried_combos": tried}

    return lats_search


def make_inventory_check(check_stock):
    async def inventory_check(state: VipDietaryState) -> dict:
        combo = state["current_combo"]
        for ingredient in combo:
            stock = await check_stock(ingredient)
            if stock <= 0:
                return {
                    "stock_ok": False,
                    "failing_ingredient": ingredient,
                    "tried_combos": list(state.get("tried_combos", [])) + [combo],
                }
        return {"stock_ok": True, "failing_ingredient": None}

    return inventory_check


def chef_signoff(state: VipDietaryState) -> dict:
    decision = interrupt({
        "type": "hitl_menu_approval",
        "event_id": state["event_id"],
        "guest_id": state["guest_id"],
        "proposed_combo": state["current_combo"],
        "reason": "Mandatory executive chef sign-off before an irreversible "
                  "VIP allergy-safe menu change.",
    })
    return {"chef_decision": decision.get("decision", "reject")}


def record_chef_decision(state: VipDietaryState) -> dict:
    if state.get("chef_decision") == "approve":
        return {}
    combo = state.get("current_combo")
    tried = list(state.get("tried_combos", []))
    return {"tried_combos": tried + [combo] if combo else tried}


def confirmed(state: VipDietaryState) -> dict:
    combo = state["current_combo"]
    return {
        "status": "CONFIRMED",
        "final_menu": f"Confirmed dishes for {state['guest_id']}: {', '.join(combo)}",
    }


def make_ticket_exhausted(db_path: str):
    def ticket_exhausted(state: VipDietaryState) -> dict:
        reason = (
            f"No safe, in-stock, chef-approved combination found for guest "
            f"{state['guest_id']} (event {state['event_id']}) from candidate "
            f"pool {state['candidate_pool']}."
        )
        raise_ticket(
            graph_id="vip_dietary_agent",
            thread_id=state.get("_thread_id", "unknown"),
            error_message=reason,
            state_snapshot=json.dumps(state, default=str),
            db_path=db_path,
        )
        return {"status": "FAILED_TICKET", "ticket_reason": reason}

    return ticket_exhausted


# ---------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------
def route_after_search(state: VipDietaryState) -> str:
    return "ticket_exhausted" if state.get("status") == "EXHAUSTED" else "inventory_check"


def _combos_exhausted(state: VipDietaryState) -> bool:
    all_combos = _all_combos(state["candidate_pool"])
    return len(state.get("tried_combos", [])) >= len(all_combos)


def route_after_inventory(state: VipDietaryState) -> str:
    if state.get("stock_ok"):
        return "chef_signoff"
    return "ticket_exhausted" if _combos_exhausted(state) else "lats_search"


def route_after_signoff(state: VipDietaryState) -> str:
    if state.get("chef_decision") == "approve":
        return "confirmed"
    return "ticket_exhausted" if _combos_exhausted(state) else "lats_search"


# ---------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------
def build_vip_dietary_graph(llm_generate=None, check_stock=None,
                             db_path: str | None = None, checkpointer=None):
    resolved_db_path = db_path or DEFAULT_DB_PATH
    llm_generate = llm_generate or _default_llm_generate
    check_stock = check_stock or _default_check_stock

    builder = StateGraph(VipDietaryState)
    builder.add_node("fetch_guest_constraints", make_fetch_guest_constraints(resolved_db_path))
    builder.add_node("lats_search", make_lats_search(llm_generate))
    builder.add_node("inventory_check", make_inventory_check(check_stock))
    builder.add_node("chef_signoff", chef_signoff)
    builder.add_node("record_chef_decision", record_chef_decision)
    builder.add_node("confirmed", confirmed)
    builder.add_node("ticket_exhausted", make_ticket_exhausted(resolved_db_path))

    builder.set_entry_point("fetch_guest_constraints")
    builder.add_edge("fetch_guest_constraints", "lats_search")
    builder.add_conditional_edges("lats_search", route_after_search, {
        "inventory_check": "inventory_check",
        "ticket_exhausted": "ticket_exhausted",
    })
    builder.add_conditional_edges("inventory_check", route_after_inventory, {
        "chef_signoff": "chef_signoff",
        "lats_search": "lats_search",
        "ticket_exhausted": "ticket_exhausted",
    })
    builder.add_edge("chef_signoff", "record_chef_decision")
    builder.add_conditional_edges("record_chef_decision", route_after_signoff, {
        "confirmed": "confirmed",
        "lats_search": "lats_search",
        "ticket_exhausted": "ticket_exhausted",
    })
    builder.add_edge("confirmed", END)
    builder.add_edge("ticket_exhausted", END)

    return builder.compile(checkpointer=checkpointer)