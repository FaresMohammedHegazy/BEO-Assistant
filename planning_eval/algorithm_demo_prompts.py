from __future__ import annotations

import asyncio
import json
import os
from dataclasses import fields, is_dataclass

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

from planning.planning_lab.algorithms.plan_and_solve import (
    plan_and_solve,
    plan_and_solve_against_mcp,
)
from planning.planning_lab.algorithms.tree_of_thoughts import tree_of_thoughts_grounded
from planning.planning_lab.algorithms.lats import lats, flatten_lats_tree
from planning.planning_lab.algorithms.reflexion import reflexion
from planning.planning_lab.algorithms.self_refine import draft_and_refine_beo_summary
from planning.planning_lab.algorithms.environment import Environment

from agent.planning_agent_executor import PlanningAgentExecutor
from agent.planning_router import PlanningRouter

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "algorithm_demo_prompts.json")


def _to_serializable(value):
    
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_serializable(getattr(value, f.name)) for f in fields(value)}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value


# ── Plan-and-Solve demo prompts ──
PS_PROMPTS = [
    {
        "instruction": "Check whether event EVT_CONF_01's current deposit is under a $15,000 budget cap.",
        "tool_name": "view_event_deposit_status",
        "tool_arguments": {"event_id": "EVT_CONF_01"},
    },
    {
        "instruction": "Check the deposit status for event EVT_999 and note whether it needs follow-up.",
        "tool_name": "view_event_deposit_status",
        "tool_arguments": {"event_id": "EVT_999"},
    },
    {
        "instruction": "If a deposit of $4,000 has been paid against a required $15,000 total, compute the outstanding balance.",
        "tool_name": None,
    },
]

# ── Tree of Thoughts demo prompts ──
TOT_PROMPTS = [
    "Select 2 distinct rooms (ROOM_XXX ids) that can each hold 40 guests without violating fire code.",
    "Propose a combination of 3 rooms for parallel breakout tracks, avoiding any room whose STRICT_ENFORCEMENT fire code cap is under 40.",
    "Choose an alternate room combination if ROOM_101 cannot support 40 guests due to its fire code limit.",
]

# ── LATS demo prompts ──
LATS_PROMPTS = [
    "Propose a VIP dinner menu for GUEST_VIP_1 (severe nut allergy, vegan) using only ingredients confirmed nut-free in the safe_ingredients table.",
    "Book the VIP guest's event EVT_999 into a room for 500 guests -- a financial-penalty risk if it violates fire code capacity.",
    "Finalize the room booking proposal for EVT_CONF_01 with no fire-code violations.",
]

# ── Reflexion demo prompts ──
REFLEXION_PROMPTS = [
    "Book 350 guests into ROOM_101, adjusting the plan if this exceeds fire code capacity.",
    "Attempt to confirm EVT_999's booking at 500 guests, retrying with a compliant headcount after the first attempt is rejected for going over capacity.",
    "Propose a room for a 40-person breakout track, correcting the plan if the first suggested room's fire code cap is exceeded.",
]

# ── Self-Refine demo prompts ──
SELF_REFINE_PROMPTS = [
    {
        "goal": "Write the BEO summary for EVT_CONF_01's 3-day leadership summit setup.",
        "context": "Main hall available 2026-09-10 to 2026-09-12. Deposit status: $4,000 of $15,000 paid.",
    },
    {
        "goal": "Draft the VIP dinner menu description for GUEST_VIP_1, noting allergy accommodations.",
        "context": "GUEST_VIP_1 has a severe nut allergy and is vegan. Confirmed nut-free ingredients: quinoa, chickpeas, roasted vegetables.",
    },
    {
        "goal": "Summarize the fire-safety policy relevant to EVT_999's room booking for the event log.",
        "context": "ROOM_101 fire code status: STRICT_ENFORCEMENT, max_capacity 300. Requested headcount: 500 (rejected).",
    },
]


async def run_all_demos() -> dict:
    llm = ChatGroq(model=MODEL_NAME, temperature=0.2)
    router = PlanningRouter()
    environment = Environment()
    results: dict[str, list[dict]] = {
        "plan_and_solve": [],
        "tree_of_thoughts": [],
        "lats": [],
        "reflexion": [],
        "self_refine": [],
    }

    async with PlanningAgentExecutor(model_name=MODEL_NAME) as executor:
        session = executor._session  

        # ── Plan-and-Solve ──
        for prompt in PS_PROMPTS:
            decision = router.route_subtask(prompt["instruction"])
            if prompt["tool_name"]:
                grounded = await plan_and_solve_against_mcp(
                    prompt["instruction"], prompt["tool_name"], prompt["tool_arguments"], llm, session
                )
                output, tool_output = grounded.reasoning, grounded.tool_output
            else:
                output, tool_output = plan_and_solve(prompt["instruction"], llm), None
            results["plan_and_solve"].append({
                "instruction": prompt["instruction"],
                "routed_algorithm": decision.algorithm.value,
                "tool_output": tool_output,
                "output": output,
            })

        # ── Tree of Thoughts ──
        for problem in TOT_PROMPTS:
            decision = router.route_subtask(problem)
            frontier = tree_of_thoughts_grounded(problem, llm, depth=2, beam_width=2)
            results["tree_of_thoughts"].append({
                "problem": problem,
                "routed_algorithm": decision.algorithm.value,
                "final_frontier": [_to_serializable(t) for t in frontier],
            })

        # ── LATS ──
        for task in LATS_PROMPTS:
            decision = router.route_subtask(task)
            outcome = lats(task, llm, environment, iterations=2, n_actions=2)
            results["lats"].append({
                "task": task,
                "routed_algorithm": decision.algorithm.value,
                "success": outcome.success,
                "output": outcome.output,
                "best_score": outcome.best_score,
                "iterations": outcome.iterations,
                "tree": flatten_lats_tree(outcome.root),
            })

        # ── Reflexion ──
        for task in REFLEXION_PROMPTS:
            decision = router.route_subtask(task)
            outcome = reflexion(task, llm, environment, max_trials=3, memory_size=2)
            results["reflexion"].append({
                "task": task,
                "routed_algorithm": decision.algorithm.value,
                "success": outcome.success,
                "output": outcome.output,
                "trials": [_to_serializable(t) for t in outcome.trials],
                "final_memory": outcome.memory,
            })

        # ── Self-Refine ──
        for item in SELF_REFINE_PROMPTS:
            decision = router.route_subtask(item["goal"])
            outcome = draft_and_refine_beo_summary(item["goal"], item["context"], llm)
            results["self_refine"].append({
                "goal": item["goal"],
                "routed_algorithm": decision.algorithm.value,
                "draft": outcome.draft,
                "critique": outcome.critique,
                "revised": outcome.revised,
                "grounded_issues": outcome.grounded_issues,
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    output = asyncio.run(run_all_demos())
    for algo, runs in output.items():
        print(f"\n=== {algo} ({len(runs)} prompts) ===")
        for run in runs:
            label = run.get("instruction") or run.get("problem") or run.get("task") or run.get("goal")
            print(f"  - {label}  (router chose: {run['routed_algorithm']})")
    print(f"\nSaved full evidence to {OUTPUT_PATH}")