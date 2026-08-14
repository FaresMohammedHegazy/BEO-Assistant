import asyncio

from agent.planning_agent_executor import PlanningAgentExecutor

CONFERENCE_REQUEST = (
    "We're planning our annual leadership summit: 3 days, ~150 attendees "
    "for the general session on 2026-09-10, 4 parallel breakout tracks "
    "(25-40 people each) across the same 3 days, and we need everything "
    "confirmed with the deposit under budget. First check the main hall's "
    "availability, then check the deposit status for event EVT_CONF_01."
)


async def run_dynamic_demo():
    print("==========================================================")
    print("DYNAMIC DECOMPOSITION (INTERLEAVED MCP EXECUTION)")
    print("==========================================================")
    print(f"[Goal]: {CONFERENCE_REQUEST}\n")

    async with PlanningAgentExecutor() as executor:
        result = await executor.run_dynamic(CONFERENCE_REQUEST, max_steps=8)

        print("==========================================================")
        print("[Execution History & Live Feedback]:")
        for i, step in enumerate(result.steps, 1):
            print(f"\nStep {i} Task: {step.instruction}")
            print(f"Step {i} Result: {step.output}")

        if result.steps:
            print("\n==========================================================")
            print("[Final Adaptive Synthesis]:")
            print(result.final_answer)
        else:
            print("\n[Final Status]: No actions were taken by the planner.")


if __name__ == "__main__":
    asyncio.run(run_dynamic_demo())