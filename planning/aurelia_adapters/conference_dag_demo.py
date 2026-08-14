import asyncio

from agent.planning_agent_executor import PlanningAgentExecutor

CONFERENCE_REQUEST = (
    "We're planning our annual leadership summit: 3 days, ~150 attendees "
    "for the general session on 2026-09-10, 4 parallel breakout tracks "
    "(25-40 people each) across the same 3 days, and we need everything "
    "confirmed with the deposit under budget. First check the main hall's "
    "availability, then check the deposit status for event EVT_CONF_01."
)


async def run_decomposition_first_demo():
    async with PlanningAgentExecutor() as executor:
        result = await executor.run_decomposition_first(CONFERENCE_REQUEST)
        print(f"[Plan] {len(result.steps)} tasks")
        for step in result.steps:
            tag = f"-> {step.tool_name}" if step.tool_name else "(reasoning)"
            print(f"  {step.task_id}: {step.instruction}  {tag}")
        print("\n[Final synthesis]:")
        print(result.final_answer)


if __name__ == "__main__":
    asyncio.run(run_decomposition_first_demo())