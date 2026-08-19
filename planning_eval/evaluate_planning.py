from __future__ import annotations
import argparse
import asyncio
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
SUITE_PATH = ROOT / "planning_eval" / "test_suite.json"
ARTIFACT_DIR = ROOT / "planning" / "artifacts"
COMPARISON_PATH = ROOT / "planning_eval" / "comparison_table.md"
README_PATH = ROOT / "README.md"
TABLE_START = "<!-- PLANNING_EVAL_TABLE_START -->"
TABLE_END = "<!-- PLANNING_EVAL_TABLE_END -->"
ALGORITHMS = ("dag", "dynamic", "ps", "tot", "reflexion", "lats", "self_refine")
ALGORITHM_LABELS = {
    "dag": "Static DAG",
    "dynamic": "Dynamic Decomposition",
    "ps": "Plan-and-Solve",
    "tot": "Tree of Thoughts",
    "reflexion": "Reflexion",
    "lats": "LATS",
    "self_refine": "Self-Refine",
}
# Approximate Groq openai/gpt-oss-120b blended token rate (USD / token).
COST_PER_TOKEN = 0.00000069
def _estimate_tokens(messages_or_text) -> int:
    if isinstance(messages_or_text, list):
        text = json.dumps(messages_or_text, ensure_ascii=False)
    else:
        text = str(messages_or_text)
    return max(1, len(text.split()))
class InstrumentedStructuredRunnable:
    def __init__(self, tracker: "InstrumentedLLM", runnable):
        self._tracker = tracker
        self._runnable = runnable
    def invoke(self, messages, **kwargs):
        self._tracker.llm_calls += 1
        self._tracker.input_tokens += _estimate_tokens(messages)
        result = self._runnable.invoke(messages, **kwargs)
        self._tracker.output_tokens += _estimate_tokens(result)
        return result
class InstrumentedLLM:
    """Wrap a chat model to collect LLM-call and token metrics."""
    def __init__(self, llm):
        self._llm = llm
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    def reset_metrics(self) -> None:
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
    def invoke(self, messages, **kwargs):
        self.llm_calls += 1
        self.input_tokens += _estimate_tokens(messages)
        response = self._llm.invoke(messages, **kwargs)
        content = getattr(response, "content", response)
        self.output_tokens += _estimate_tokens(content)
        return response
    def with_structured_output(self, schema, **kwargs):
        return InstrumentedStructuredRunnable(
            self,
            self._llm.with_structured_output(schema, **kwargs),
        )
def _db_path() -> Path:
    return ROOT / "db" / "aurelia.db"
def _extract_room_ids(text: str) -> list[str]:
    matches = re.findall(r"\bROOM_\d+\b", text or "", flags=re.IGNORECASE)
    return [match.upper() for match in matches]
def _extract_event_ids(text: str) -> list[str]:
    matches = re.findall(r"\bEVT_\d+\b", text or "", flags=re.IGNORECASE)
    return [match.upper() for match in matches]
def _room_capacity_ok(room_id: str, min_capacity: int) -> bool:
    db_path = _db_path()
    if not db_path.exists():
        return False
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT max_capacity FROM rooms WHERE room_id = ?",
            (room_id.upper(),),
        ).fetchone()
    return row is not None and row[0] >= min_capacity
def _score_scenario_1_static(result_text: str) -> bool:
    lowered = (result_text or "").lower()
    required_rooms = ("room_102", "room_103", "room_104")
    if not all(room in lowered for room in required_rooms):
        return False
    room_ids = _extract_room_ids(result_text)
    return len(set(room_ids)) >= 3 and all(_room_capacity_ok(room, 1) for room in set(room_ids))
def _score_scenario_2_dynamic(result_text: str) -> bool:
    lowered = (result_text or "").lower()
    if "evt_999" not in lowered:
        return False
    if "evt_conf_01" in lowered and "not found" not in lowered and "missing" not in lowered:
        # Treat a confident EVT_CONF_01 answer with no pivot as a failure.
        if "evt_999" not in _extract_event_ids(result_text):
            return False
    return any(token in lowered for token in ("deposit", "pending", "room_101", "room"))
def _score_scenario_3_search(result_text: str) -> bool:
    room_ids = _extract_room_ids(result_text)
    unique_rooms = list(dict.fromkeys(room_ids))
    if len(unique_rooms) < 4:
        return False
    return all(_room_capacity_ok(room, 40) for room in unique_rooms[:4])
def _score_scenario_4_reflexion(result_text: str) -> bool:
    from planning.planning_lab.algorithms.environment import Environment
    feedback = Environment().evaluate(result_text)
    if not feedback.success:
        return False
    lowered = (result_text or "").lower()
    if "evt_999" not in lowered or "room_101" not in lowered:
        return False
    headcount = Environment._extract_headcount(result_text)
    return headcount is not None and headcount <= 300
SCENARIO_SCORERS = {
    "scenario_1_static": _score_scenario_1_static,
    "scenario_2_dynamic": _score_scenario_2_dynamic,
    "scenario_3_search": _score_scenario_3_search,
    "scenario_4_reflexion": _score_scenario_4_reflexion,
}
def score_scenario(scenario_id: str, result_text: str) -> bool:
    scorer = SCENARIO_SCORERS.get(scenario_id)
    if scorer is None:
        return False
    return scorer(result_text)

# ═══════════════════════════════════════════════════════════════
# Runner: executes every applicable algorithm against every test
# scenario, aggregates metrics, and writes the comparison table.
# Everything above this line is the original scaffolding.
# ═══════════════════════════════════════════════════════════════

SCENARIO_ALGORITHMS = {
    "scenario_1_static": ("dag", "dynamic"),
    "scenario_2_dynamic": ("dag", "dynamic"),
    "scenario_3_search": ("ps", "tot", "lats"),
    "scenario_4_reflexion": ("self_refine", "reflexion"),
}


async def _run_decomposition_first(scenario: dict, tracked_llm, session) -> str:
    from planning.planning_lab.algorithms.decomposition import (
        decompose_goal_grounded, execute_plan_against_mcp, final_output,
    )
    plan, tool_bindings = decompose_goal_grounded(scenario["goal"], tracked_llm)
    outputs = await execute_plan_against_mcp(plan, tool_bindings, tracked_llm, session)
    return final_output(plan, outputs)


async def _run_dynamic(scenario: dict, tracked_llm, session) -> str:
    from planning.planning_lab.algorithms.dynamic_decomposition import dynamic_decomposition_grounded
    history = await dynamic_decomposition_grounded(scenario["goal"], tracked_llm, session, max_steps=8)
    return history[-1][1] if history else ""


def _run_plan_and_solve(scenario: dict, tracked_llm) -> str:
    from planning.planning_lab.algorithms.plan_and_solve import plan_and_solve
    return plan_and_solve(scenario["goal"], tracked_llm)


def _run_tree_of_thoughts(scenario: dict, tracked_llm) -> str:
    from planning.planning_lab.algorithms.tree_of_thoughts import tree_of_thoughts_grounded
    frontier = tree_of_thoughts_grounded(scenario["goal"], tracked_llm, depth=2, beam_width=2)
    return frontier[0].state if frontier else ""


def _run_lats(scenario: dict, tracked_llm, environment) -> str:
    from planning.planning_lab.algorithms.lats import lats
    outcome = lats(scenario["goal"], tracked_llm, environment, iterations=2, n_actions=2)
    return outcome.output


def _run_self_refine(scenario: dict, tracked_llm) -> str:
    from planning.planning_lab.algorithms.self_refine import draft_and_refine_beo_summary
    outcome = draft_and_refine_beo_summary(scenario["goal"], context="", llm=tracked_llm)
    return outcome.revised


def _run_reflexion(scenario: dict, tracked_llm, environment) -> str:
    from planning.planning_lab.algorithms.reflexion import reflexion
    outcome = reflexion(scenario["goal"], tracked_llm, environment, max_trials=3, memory_size=2)
    return outcome.output


async def _run_case(algorithm: str, scenario: dict, tracked_llm, session, environment) -> dict:
    start = time.perf_counter()
    if algorithm == "dag":
        text = await _run_decomposition_first(scenario, tracked_llm, session)
    elif algorithm == "dynamic":
        text = await _run_dynamic(scenario, tracked_llm, session)
    elif algorithm == "ps":
        text = _run_plan_and_solve(scenario, tracked_llm)
    elif algorithm == "tot":
        text = _run_tree_of_thoughts(scenario, tracked_llm)
    elif algorithm == "lats":
        text = _run_lats(scenario, tracked_llm, environment)
    elif algorithm == "self_refine":
        text = _run_self_refine(scenario, tracked_llm)
    elif algorithm == "reflexion":
        text = _run_reflexion(scenario, tracked_llm, environment)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return {
        "scenario_id": scenario["id"],
        "algorithm": algorithm,
        "output": text,
        "success": score_scenario(scenario["id"], text),
        "latency_seconds": round(time.perf_counter() - start, 2),
    }


async def run_evaluation() -> list[dict]:
    from langchain_groq import ChatGroq
    from agent.planning_agent_executor import PlanningAgentExecutor
    from planning.planning_lab.algorithms.environment import Environment

    load_dotenv()
    model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

    with open(SUITE_PATH, "r", encoding="utf-8") as f:
        scenarios = {item["id"]: item for item in json.load(f)}

    environment = Environment()
    executor = PlanningAgentExecutor(model_name=model_name)
    await executor.connect()  # one live MCP session, reused for every dag/dynamic run

    records: list[dict] = []
    try:
        for scenario_id, algorithms in SCENARIO_ALGORITHMS.items():
            scenario = scenarios[scenario_id]
            for algorithm in algorithms:
                tracked_llm = InstrumentedLLM(ChatGroq(model=model_name, temperature=0.2))
                record = await _run_case(algorithm, scenario, tracked_llm, executor.session, environment)
                record.update({
                    "llm_calls": tracked_llm.llm_calls,
                    "total_tokens": tracked_llm.total_tokens,
                    "estimated_cost_usd": round(tracked_llm.total_tokens * COST_PER_TOKEN, 6),
                })
                records.append(record)
                print(f"[{ALGORITHM_LABELS[algorithm]}] {scenario_id}: "
                      f"success={record['success']} calls={record['llm_calls']} "
                      f"tokens={record['total_tokens']} latency={record['latency_seconds']}s")
    finally:
        await executor.close()

    return records


def _aggregate(records: list[dict]) -> dict[str, dict]:
    aggregated: dict[str, dict] = {}
    for algorithm in ALGORITHMS:
        rows = [r for r in records if r["algorithm"] == algorithm]
        if not rows:
            continue
        aggregated[algorithm] = {
            "cases": len(rows),
            "success_rate": f"{sum(r['success'] for r in rows)}/{len(rows)}",
            "avg_llm_calls": round(statistics.mean(r["llm_calls"] for r in rows), 1),
            "avg_tokens": round(statistics.mean(r["total_tokens"] for r in rows), 0),
            "avg_latency_s": round(statistics.mean(r["latency_seconds"] for r in rows), 2),
            "avg_cost_usd": round(statistics.mean(r["estimated_cost_usd"] for r in rows), 5),
        }
    return aggregated


def _render_table(aggregated: dict[str, dict]) -> str:
    rows = [
        "| Method | Cases | Success | Avg LLM calls | Avg tokens | Avg latency | Avg cost/run |",
        "|---|---|---|---|---|---|---|",
    ]
    for algorithm in ALGORITHMS:
        if algorithm not in aggregated:
            continue
        s = aggregated[algorithm]
        rows.append(
            f"| {ALGORITHM_LABELS[algorithm]} | {s['cases']} | {s['success_rate']} | "
            f"{s['avg_llm_calls']} | {s['avg_tokens']:.0f} | {s['avg_latency_s']}s | ${s['avg_cost_usd']} |"
        )
    return "\n".join(rows)


def _write_comparison_table(table_md: str) -> None:
    COMPARISON_PATH.write_text(table_md + "\n", encoding="utf-8")
    if not README_PATH.exists():
        return
    readme_text = README_PATH.read_text(encoding="utf-8")
    if TABLE_START not in readme_text or TABLE_END not in readme_text:
        readme_text += f"\n\n## Planning & Decomposition Comparison Table\n\n{TABLE_START}\n{table_md}\n{TABLE_END}\n"
    else:
        before = readme_text.split(TABLE_START)[0]
        after = readme_text.split(TABLE_END)[1]
        readme_text = f"{before}{TABLE_START}\n{table_md}\n{TABLE_END}{after}"
    README_PATH.write_text(readme_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the planning/decomposition comparison suite.")
    parser.parse_args()

    records = asyncio.run(run_evaluation())

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_path = ARTIFACT_DIR / f"planning_eval_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
    run_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    table_md = _render_table(_aggregate(records))
    _write_comparison_table(table_md)

    print("\n" + table_md)
    print(f"\nSaved raw run to {run_path}")
    print(f"Saved comparison table to {COMPARISON_PATH}")
    print(f"Embedded table in {README_PATH}")


if __name__ == "__main__":
    main()