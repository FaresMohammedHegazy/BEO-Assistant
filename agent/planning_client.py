"""
Standalone entry point for the Planning Agent (Issue 6.3).

This module is a deliberately separate entry point from `agent/client.py`.
It boots its own CLI, opens its own MCP stdio session, and drives goal
decomposition/execution through `PlanningAgentExecutor` and `PlanningRouter`.

It shares resources with the existing Memory/RAG agent -- the MCP server in
`mcp_server/` (spawned the same way, via `python -m mcp_server.server`) and
the SQLite database at `db/aurelia.db` (opened by the server itself) -- but
it does not import, call, or otherwise touch anything in `agent/client.py`.
The two agents are independent processes that can be run separately or at
the same time; each spawns its own short-lived MCP server subprocess and
each subprocess reads/writes the same on-disk `db/aurelia.db`.

Usage
-----
    python agent/planning_client.py
    python agent/planning_client.py "<goal>"
    python agent/planning_client.py "<goal>" --mode dynamic --max-steps 8

See README.md, section "Run the planning agent", for full setup and
execution instructions.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Allow `python agent/planning_client.py` to be run directly (i.e. as a
# script, not with `-m`) by putting the repo root on sys.path before any
# repo-internal package is imported. Mirrors the same guard already used by
# mcp_server/server.py, agent/tests/test_planning_router.py, and the
# planning/aurelia_adapters/*.py demo scripts.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv

from agent.planning_agent_executor import DAGRunResult, PlanningAgentExecutor
from agent.planning_router import PlanningRouter

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

DEFAULT_GOAL = (
    "We're planning our annual leadership summit: 3 days, ~150 attendees "
    "for the general session on 2026-09-10, 4 parallel breakout tracks "
    "(25-40 people each) across the same 3 days, and we need everything "
    "confirmed with the deposit under budget. First check the main hall's "
    "availability, then check the deposit status for event EVT_CONF_01."
)


def _check_groq_api_key() -> None:
    """Fail fast with an actionable message, same contract as agent/client.py."""
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. export GROQ_API_KEY=<your Groq key> "
            "before running agent/planning_client.py."
        )
    if "\n" in api_key or "\r" in api_key or api_key.startswith("export ") or api_key.startswith("cd "):
        raise RuntimeError(
            "GROQ_API_KEY looks malformed. It appears to contain shell text instead of a "
            "real Groq API key. Unset the bad environment variable and "
            "export GROQ_API_KEY=<your Groq key> only."
        )


async def run(goal: str, mode: str, max_steps: int) -> DAGRunResult:
    """Connect to the shared MCP server and execute `goal` with the requested strategy.

    Opens/closes its own MCP stdio session via `PlanningAgentExecutor`
    (spawned the same way `agent/client.py` spawns its own session), so it
    never shares a live session with the RAG agent -- only the on-disk
    `db/aurelia.db` and the `mcp_server/` code are shared.
    """
    async with PlanningAgentExecutor(model_name=MODEL_NAME) as executor:
        if mode == "dynamic":
            result = await executor.run_dynamic(goal, max_steps=max_steps)
        else:
            result = await executor.run_decomposition_first(goal)

    # Post-hoc routing audit: for each sub-task actually executed, record
    # which of the four planning algorithms (PS/ToT/Reflexion/LATS) the
    # Centralized Planning Router would recommend, with its rationale. This
    # is advisory/audit output; it does not change which code path executed
    # the step above.
    router = PlanningRouter()
    print("\n[Planning Router] Per-subtask algorithm audit:")
    for step in result.steps:
        decision = router.route_subtask(step.instruction)
        print(f"  {step.task_id}: [{decision.algorithm.value}] (confidence={decision.confidence:.2f})")
        print(f"     instruction: {step.instruction}")
        print(f"     topology={decision.topology.value} signals={decision.detected_signals}")

    return result


def _print_result(result: DAGRunResult) -> None:
    print(f"\n[{result.method}] goal: {result.goal}")
    print(f"[{result.method}] {len(result.steps)} step(s) executed:")
    for step in result.steps:
        tag = f"-> {step.tool_name}" if step.tool_name else "(reasoning)"
        print(f"  {step.task_id}: {step.instruction} {tag}")
        print(f"     output: {step.output}")
    print("\n[Final Answer]")
    print(result.final_answer)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="planning_client.py",
        description=(
            "Planning Agent entry point. Runs independently of agent/client.py "
            "while sharing mcp_server/ and db/aurelia.db."
        ),
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default=DEFAULT_GOAL,
        help="Natural-language goal to decompose and execute. Defaults to a demo conference-planning goal.",
    )
    parser.add_argument(
        "--mode",
        choices=["decomposition", "dynamic"],
        default="decomposition",
        help=(
            "Planning strategy. 'decomposition' builds the full task DAG up front and then "
            "executes it against the MCP server. 'dynamic' interleaves planning and "
            "observation one step at a time. Default: decomposition."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum steps for --mode dynamic. Ignored in decomposition mode. Default: 8.",
    )
    return parser


def main() -> None:
    # Parse args first so `--help`/`-h` works without requiring GROQ_API_KEY.
    args = build_arg_parser().parse_args()
    _check_groq_api_key()
    result = asyncio.run(run(args.goal, args.mode, args.max_steps))
    _print_result(result)


if __name__ == "__main__":
    main()
