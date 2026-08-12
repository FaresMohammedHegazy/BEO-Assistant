from langchain_core.language_models.chat_models import BaseChatModel
from dataclasses import dataclass

def plan_and_solve(question: str, llm: BaseChatModel) -> str:
    response = llm.invoke([
        ("system", "You use Plan-and-Solve prompting. Clearly separate PLAN from SOLUTION."),
        ("human", f"""{question}

First understand the problem and devise a plan to solve it. Then carry out the
plan step by step. Check calculations and common-sense assumptions."""),
    ], temperature=0.2)
    if not isinstance(response.content, str) or not response.content.strip():
        raise RuntimeError("The chat model returned an empty or unsupported response")
    return response.content.strip()

@dataclass
class GroundedPSResult:
    tool_output: str
    reasoning: str


async def plan_and_solve_against_mcp(
    instruction: str,
    tool_name: str,
    tool_arguments: dict,
    llm: BaseChatModel,
    mcp_session,
) -> GroundedPSResult:
    """Plan-and-Solve for single-pass, no-branching DAG nodes that need
    exactly one real fact from Aurelia's systems before reasoning over
    it (e.g. 'is the deposit under budget?'). Reuses plan_and_solve()
    unmodified -- this only adds the real MCP fact-fetch in front of it.
    """
    result = await mcp_session.call_tool(tool_name, arguments=tool_arguments)
    tool_output = "\n".join(
        block.text for block in result.content if hasattr(block, "text")
    )

    reasoning = plan_and_solve(
        f"""{instruction}

Real data retrieved from the Aurelia system:
{tool_output}

Use this data exactly as given -- do not invent or round numbers.""",
        llm,
    )
    return GroundedPSResult(tool_output=tool_output, reasoning=reasoning)