import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv() 

from langchain_groq import ChatGroq
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from planning.planning_lab.algorithms.decomposition import (
    decompose_goal_grounded,
    execute_plan_against_mcp,
)
from planning.planning_lab.algorithms.decomposition import final_output  # reused unchanged

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

CONFERENCE_REQUEST = (
    "We're planning our annual leadership summit: 3 days, ~150 attendees "
    "for the general session on 2026-09-10, 4 parallel breakout tracks "
    "(25-40 people each) across the same 3 days, and we need everything "
    "confirmed with the deposit under budget. First check the main hall's "
    "availability, then check the deposit status for event EVT_CONF_01."
)


async def run_decomposition_first_demo():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()  # same handshake pattern as agent/client.py

            plan, tool_bindings = decompose_goal_grounded(CONFERENCE_REQUEST, llm)
            print(f"[Plan] {len(plan.tasks)} tasks, "
                  f"{len(tool_bindings)} bound to real MCP tools")
            for task in plan.tasks:
                binding = tool_bindings.get(task.id)
                tag = f"-> {binding.tool_name}" if binding else "(reasoning)"
                print(f"  {task.id}: {task.instruction}  {tag}")

            outputs = await execute_plan_against_mcp(plan, tool_bindings, llm, session)
            print("\n[Final synthesis]:")
            print(final_output(plan, outputs))


if __name__ == "__main__":
    asyncio.run(run_decomposition_first_demo())