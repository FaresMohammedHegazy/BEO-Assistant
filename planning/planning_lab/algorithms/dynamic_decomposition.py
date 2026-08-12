from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict

# Import tool bindings from the decomposition module
from .decomposition import AURELIA_TOOL_CATALOG, ToolCall


class DynamicDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    next_task: str


class GroundedDynamicDecision(BaseModel):
    """Augments the dynamic decision with real MCP tool bindings."""
    model_config = ConfigDict(extra="forbid")

    done: bool
    next_task: str
    tool_call: ToolCall | None = None


def dynamic_decomposition(goal: str, llm: BaseChatModel, max_steps: int = 4) -> list[tuple[str, str]]:
    """Original non-grounded dynamic decomposition loop."""
    history: list[tuple[str, str]] = []
    for step in range(max_steps):
        observation = "\n".join(f"{task}: {result}" for task, result in history) or "None"
        
        # FIX applied here: removed method="json_schema"
        decision = llm.with_structured_output(
            DynamicDecision
        ).invoke([
            ("system", "You are an adaptive planner. Use prior observations before deciding what comes next."),
            ("human", f"""Goal: {goal}
Completed work and observations:
{observation}

Decide the single best next task. Set done to true only when the goal is met.
When done is true, use an empty string for next_task."""),
        ], temperature=0.1)
        
        if decision.done:
            break
        task = decision.next_task.strip()
        if not task:
            raise ValueError(f"Dynamic planner omitted next_task at step {step + 1}")
        response = llm.invoke([
            ("system", "Execute the next adaptive sub-task using the observations provided."),
            ("human", f"Goal: {goal}\nNext task: {task}\nPrior observations:\n{observation}"),
        ], temperature=0.2)
        result = response.content
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        result = result.strip()
        history.append((task, result))
    return history


async def dynamic_decomposition_grounded(
    goal: str, llm: BaseChatModel, mcp_session, max_steps: int = 8
) -> list[tuple[str, str]]:
    """
    Dynamic, interleaved decomposition grounded in real MCP tools.
    Decides one step -> Executes tool/reasoning -> Observes -> Decides next step.
    """
    history: list[tuple[str, str]] = []
    
    for step in range(max_steps):
        observation = "\n".join(f"{task}: {result}" for task, result in history) or "None"
        
        # 1. Decide the next task and optionally bind to a tool
        # FIX applied here: removed method="json_schema"
        decision = llm.with_structured_output(
            GroundedDynamicDecision
        ).invoke([
            ("system", f"You are an adaptive planner working with real tools.\n\n{AURELIA_TOOL_CATALOG}"),
            ("human", f"""Goal: {goal}
Completed work and observations:
{observation}

Decide the single best next task based on the observations. 
Bind to a tool if it is a single, deterministic, mechanical lookup or write.
If the goal is fully met or it is impossible to proceed due to a hard blocker, set done to true and next_task to empty."""),
        ], temperature=0.1)

        if decision.done:
            break
            
        task = decision.next_task.strip()
        if not task:
            break
            
        # 2. Interleaved Execution: Call the MCP Tool or use LLM Reasoning
        if decision.tool_call:
            try:
                result = await mcp_session.call_tool(
                    decision.tool_call.tool_name, 
                    arguments=decision.tool_call.arguments
                )
                output_text = "\n".join(
                    block.text for block in result.content if hasattr(block, "text")
                )
            except Exception as e:
                output_text = f"Tool execution failed: {str(e)}"
        else:
            response = llm.invoke([
                ("system", "Execute the next adaptive sub-task using the observations provided. Synthesize or reason over the data."),
                ("human", f"Goal: {goal}\nNext task: {task}\nPrior observations:\n{observation}"),
            ], temperature=0.2)
            
            output = response.content
            if not isinstance(output, str) or not output.strip():
                raise RuntimeError("The chat model returned an empty or unsupported response")
            output_text = output.strip()
            
        # 3. Store in history (which acts as episodic memory for the next loop)
        history.append((task, output_text))
        
    return history