from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from planning.planning_lab.algorithms.decomposition import (
    decompose_goal_grounded,
    execute_plan_against_mcp,
    final_output,
)
from planning.planning_lab.algorithms.dynamic_decomposition import (
    dynamic_decomposition_grounded,
)

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")


@dataclass
class DAGStep:
    task_id: str
    instruction: str
    tool_name: str | None   # None => this step was a reasoning node, not a real tool call
    output: str


@dataclass
class DAGRunResult:
    method: str              # "decomposition_first" | "dynamic"
    goal: str
    steps: list[DAGStep] = field(default_factory=list)
    final_answer: str = ""


class PlanningAgentExecutor:

    def __init__(self, model_name: str = MODEL_NAME):
        self.llm = ChatGroq(model=model_name, temperature=0.1)
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    async def connect(self) -> None:
        mcp_url = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8765/mcp")
        self._exit_stack = AsyncExitStack()
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            streamable_http_client(mcp_url)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._session = None
        self._exit_stack = None

    @property
    def session(self) -> ClientSession | None:
        """Read-only access to the live MCP session, for callers (like
        the evaluation harness) that need to pass it directly into
        algorithm functions instead of going through run_*()."""
        return self._session
    
    async def __aenter__(self) -> "PlanningAgentExecutor":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def run_decomposition_first(self, goal: str) -> DAGRunResult:
        
        if self._session is None:
            raise RuntimeError("Call connect() (or use 'async with') before running a DAG.")

        plan, tool_bindings = decompose_goal_grounded(goal, self.llm)
        outputs = await execute_plan_against_mcp(plan, tool_bindings, self.llm, self._session)

        steps = [
            DAGStep(
                task_id=task.id,
                instruction=task.instruction,
                tool_name=tool_bindings[task.id].tool_name if task.id in tool_bindings else None,
                output=outputs[task.id],
            )
            for task in plan.tasks
        ]
        return DAGRunResult(
            method="decomposition_first",
            goal=goal,
            steps=steps,
            final_answer=final_output(plan, outputs),
        )

    async def run_dynamic(self, goal: str, max_steps: int = 8) -> DAGRunResult:
       
        if self._session is None:
            raise RuntimeError("Call connect() (or use 'async with') before running a DAG.")

        history = await dynamic_decomposition_grounded(
            goal, self.llm, self._session, max_steps=max_steps
        )
       
        steps = [
            DAGStep(task_id=f"d{i+1}", instruction=task, tool_name=None, output=result)
            for i, (task, result) in enumerate(history)
        ]
        return DAGRunResult(
            method="dynamic",
            goal=goal,
            steps=steps,
            final_answer=history[-1][1] if history else "",
        )


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        goal = (
            "We're planning our annual leadership summit: 3 days, ~150 attendees "
            "for the general session on 2026-09-10, 4 parallel breakout tracks "
            "(25-40 people each) across the same 3 days, and we need everything "
            "confirmed with the deposit under budget. First check the main hall's "
            "availability, then check the deposit status for event EVT_CONF_01."
        )
        async with PlanningAgentExecutor() as executor:
            result = await executor.run_decomposition_first(goal)
            print(f"[{result.method}] {len(result.steps)} steps")
            for step in result.steps:
                tag = f"-> {step.tool_name}" if step.tool_name else "(reasoning)"
                print(f"  {step.task_id}: {step.instruction} {tag}")
            print("\n[Final]:", result.final_answer)

    asyncio.run(_demo())