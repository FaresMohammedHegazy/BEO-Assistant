import asyncio
import os
import sys

# 1. Add the root repository directory to Python's path FIRST
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv
load_dotenv() 

from langchain_groq import ChatGroq
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# Now Python can successfully find the 'planning' module
from planning.planning_lab.algorithms.dynamic_decomposition import dynamic_decomposition_grounded

# Exact same scenario as Issue 2.1 for benchmarking purposes
CONFERENCE_REQUEST = (
    "We're planning our annual leadership summit: 3 days, ~150 attendees "
    "We're planning our annual leadership summit: 3 days, ~150 attendees "
    "for the general session on 2026-09-10, 4 parallel breakout tracks "
    "(25-40 people each) across the same 3 days, and we need everything "
    "confirmed with the deposit under budget. First check the main hall's "
    "availability, then check the deposit status for event EVT_CONF_01."
)

async def run_dynamic_demo():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env=env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            print("==========================================================")
            print("DYNAMIC DECOMPOSITION (INTERLEAVED MCP EXECUTION)")
            print("==========================================================")
            print(f"[Goal]: {CONFERENCE_REQUEST}\n")
            
            history = await dynamic_decomposition_grounded(
                CONFERENCE_REQUEST, llm, session, max_steps=8
            )
            
            print("==========================================================")
            print("[Execution History & Live Feedback]:")
            for i, (task, result) in enumerate(history, 1):
                # Print the executed task
                print(f"\nStep {i} Task: {task}")
                # Print the tool or reasoning response received
                print(f"Step {i} Result: {result}")
                
            if history:
                print("\n==========================================================")
                print("[Final Adaptive Synthesis]:")
                print(history[-1][1])
            else:
                print("\n[Final Status]: No actions were taken by the planner.")

if __name__ == "__main__":
    asyncio.run(run_dynamic_demo())