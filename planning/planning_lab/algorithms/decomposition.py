from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

from ..models import Plan


PLANNER_SYSTEM = """You are a careful task-decomposition planner.
Produce a small executable DAG, not a prose checklist. Every task must make a concrete
contribution to the goal. Independent research or analysis tasks should be parallel.
The plan must end with exactly one synthesis task depending on every necessary branch."""


class PlannedTask(BaseModel):
    """Wire schema; richer semantic constraints are applied by the Task domain model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    depends_on: list[str]


class GeneratedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask]


def decompose_goal(goal: str, llm: BaseChatModel) -> Plan:
    generated = llm.with_structured_output(
        GeneratedPlan,
        method="json_schema",
    ).invoke([
        ("system", PLANNER_SYSTEM),
        ("human", f"""Decompose this goal into 3-6 tasks: {goal!r}
Use short task ids such as t1. Dependencies may refer only to tasks in the plan.
Preserve the supplied goal exactly in the plan's goal field."""),
    ], temperature=0.1)
    # The caller's goal remains authoritative even if the model paraphrases it.
    payload = generated.model_dump()
    payload["goal"] = goal
    return Plan.model_validate(payload)


def execute_plan(plan: Plan, llm: BaseChatModel, max_workers: int = 4) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for batch in plan.execution_batches():
        prompts: dict[str, str] = {}
        for task_id in batch:
            task = plan.task(task_id)
            context = "\n\n".join(
                f"OUTPUT FROM {dependency}:\n{outputs[dependency]}"
                for dependency in task.depends_on
            ) or "No prerequisite outputs."
            prompts[task_id] = f"""Overall goal: {plan.goal}
                Current task: {task.instruction}
                Prerequisite outputs:
                {context}
                Complete only the current task. Be concrete and concise. Do not invent sources."""
        # unnecessary but nice to have
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as pool:
            futures = {
                pool.submit(
                    llm.invoke,
                    [
                        ("system", "You execute one node in a validated task DAG."),
                        ("human", prompt),
                    ],
                    temperature=0.2,
                ): task_id
                for task_id, prompt in prompts.items()
            }
            for future in as_completed(futures):
                content = future.result().content
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("The chat model returned an empty or unsupported response")
                outputs[futures[future]] = content.strip()
    return outputs


def final_output(plan: Plan, outputs: dict[str, str]) -> str:
    terminals = plan.terminal_tasks()
    if len(terminals) != 1:
        raise ValueError(f"Expected exactly one terminal synthesis task, found {terminals}")
    return outputs[terminals[0]]

# ---------------------------------------------------------------
# Aurelia extension: tool-grounded decomposition-first (Issue 2.1)
# ---------------------------------------------------------------

class ToolCall(BaseModel):
    """A concrete MCP tool binding for one DAG node. Lives OUTSIDE the
    shared Task/Plan schema in models.py on purpose, so ToT/LATS/etc.
    that reuse Plan are never affected by this."""
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, str | int]


class PlannedToolTask(PlannedTask):
    """Same wire shape as PlannedTask, plus an optional tool binding
    the planner LLM fills in when a step maps directly to a real,
    deterministic MCP tool instead of free-text reasoning."""
    tool_call: ToolCall | None = None


class GeneratedToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str
    tasks: list[PlannedToolTask]


AURELIA_TOOL_CATALOG = """Available real tools you may bind a task to
(use the EXACT tool_name and EXACT argument names):

- audit_chain_wide_availability(audit_date: str) -> checks room availability for a date
- book_event_room(event_id: str, room_id: str, requested_headcount: int) -> books a room, enforces fire-code capacity
- view_event_deposit_status(event_id: str) -> returns current deposit status for an event
- draft_custom_menu(guest_id: str) -> drafts a safe menu for one VIP guest

Only bind a task to a tool if the task is a single, deterministic,
mechanical lookup or write. Reasoning/synthesis tasks (e.g. "propose
the final combined plan") should have tool_call = null."""


def decompose_goal_grounded(
    goal: str, llm: BaseChatModel
) -> tuple[Plan, dict[str, ToolCall]]:
    """Decomposition-first, grounded in real Aurelia MCP tools.

    Returns the same `Plan` object the rest of the toolkit already
    understands (so execution_batches(), terminal_tasks(), the cycle
    check in models.py, etc. all work unmodified), plus a side-dict of
    tool bindings the executor below uses to know which nodes are
    real tool calls vs. reasoning calls.
    """
    generated = llm.with_structured_output(
        GeneratedToolPlan,
        method="json_schema",
    ).invoke([
        ("system", PLANNER_SYSTEM + "\n\n" + AURELIA_TOOL_CATALOG),
        ("human", f"""Decompose this Aurelia Hotels conference-booking
request into 3-8 tasks: {goal!r}
Use short task ids such as t1. Dependencies may refer only to tasks
in the plan. Preserve the supplied goal exactly in the plan's goal
field. Bind every mechanical lookup/write task to a real tool from
the catalog above."""),
    ], temperature=0.1)

    payload = generated.model_dump()
    payload["goal"] = goal

    # Strip tool_call before validating against the SHARED Plan/Task
    # schema in models.py -- that schema is untouched and used by
    # every other algorithm in this repo, so we never pass it a field
    # it doesn't know about.
    plan_payload = {
        "goal": payload["goal"],
        "tasks": [
            {"id": t["id"], "instruction": t["instruction"], "depends_on": t["depends_on"]}
            for t in payload["tasks"]
        ],
    }
    plan = Plan.model_validate(plan_payload)  # cycle-check happens here (Issue 2.3 depends on this)

    tool_bindings = {
        t["id"]: ToolCall.model_validate(t["tool_call"])
        for t in payload["tasks"]
        if t.get("tool_call")
    }
    return plan, tool_bindings


async def execute_plan_against_mcp(
    plan: Plan,
    tool_bindings: dict[str, ToolCall],
    llm: BaseChatModel,
    mcp_session,          # an already-connected mcp ClientSession (see file 3 below)
    max_workers: int = 4,
) -> dict[str, str]:
    """Same batch-by-batch execution shape as execute_plan(), but a
    node bound to a real tool calls mcp_session.call_tool(...) instead
    of asking the LLM to imagine a result."""
    outputs: dict[str, str] = {}
    for batch in plan.execution_batches():
        for task_id in batch:  # kept sequential per batch: MCP stdio session is single-connection
            task = plan.task(task_id)
            if task_id in tool_bindings:
                call = tool_bindings[task_id]
                result = await mcp_session.call_tool(call.tool_name, arguments=call.arguments)
                outputs[task_id] = "\n".join(
                    block.text for block in result.content if hasattr(block, "text")
                )
            else:
                context = "\n\n".join(
                    f"OUTPUT FROM {dep}:\n{outputs[dep]}" for dep in task.depends_on
                ) or "No prerequisite outputs."
                response = llm.invoke([
                    ("system", "You execute one reasoning node in a validated task DAG."),
                    ("human", f"""Overall goal: {plan.goal}
                        Current task: {task.instruction}
                        Prerequisite outputs:
                        {context}
                        Complete only the current task. Be concrete and concise."""),
                ], temperature=0.2)
                outputs[task_id] = response.content.strip()
    return outputs