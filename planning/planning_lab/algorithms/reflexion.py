from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from ..models import EnvironmentFeedback
from .environment import Environment


@dataclass
class ReflexionTrial:
    number: int
    attempt: str
    feedback: EnvironmentFeedback
    reflection: str | None = None


@dataclass
class ReflexionResult:
    success: bool
    output: str
    trials: list[ReflexionTrial]
    memory: list[str]


def reflexion(
    task: str,
    llm: BaseChatModel,
    environment: Environment,
    max_trials: int = 3,
    memory_size: int = 3,
    memory: list[str] | None = None,
) -> ReflexionResult:
    if max_trials < 1 or memory_size < 1:
        raise ValueError("max_trials and memory_size must be positive")
        
    # 1. Persist memory across calls if provided, otherwise initialize
    if memory is None:
        memory = []
        
    trials: list[ReflexionTrial] = []
    best_attempt = ""
    best_score = -1.0
    
    for number in range(1, max_trials + 1):
        # 2. Ensure memory is explicitly capped to memory_size IN-PLACE
        # Using memory[:] ensures we don't break the reference to the caller's list
        memory[:] = memory[-memory_size:] if memory else []
        
        recalled = "\n".join(f"- {item}" for item in memory) or "- No prior trials."
        
        response = llm.invoke([
            ("system", "You are the acting agent in a Reflexion loop. Attempt the entire task again."),
            ("human", f"""Task: {task}
Episodic memory from previous failed trials:
{recalled}

Produce the complete deliverable. Apply remembered lessons without discussing them."""),
        ], temperature=0.2)
        
        attempt = response.content
        if not isinstance(attempt, str) or not attempt.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        attempt = attempt.strip()
        
        # 3. Uses Grounded Environment (SQLite DB) for Evaluation
        feedback = environment.evaluate(attempt)
        trial = ReflexionTrial(number=number, attempt=attempt, feedback=feedback)
        
        if feedback.score > best_score:
            best_attempt, best_score = attempt, feedback.score
            
        if feedback.success:
            trials.append(trial)
            return ReflexionResult(True, attempt, trials, memory)
            
        response = llm.invoke([
            ("system", "Generate a concise first-person Reflexion memory, not a revised answer."),
            ("human", f"""Task: {task}
Failed attempt:
{attempt}

External environment feedback (score {feedback.score}):
{chr(10).join('- ' + item for item in feedback.details)}

State what I did wrong and the specific strategy I should use next trial. Start with 'I'."""),
        ], temperature=0.2)
        
        reflection = response.content
        if not isinstance(reflection, str) or not reflection.strip():
            raise RuntimeError("The chat model returned an empty or unsupported response")
        reflection = reflection.strip()
        
        trial.reflection = reflection
        trials.append(trial)
        
        # 4. Add to memory and cap it IN-PLACE so the caller retains the items
        memory.append(reflection)
        memory[:] = memory[-memory_size:]
        
    return ReflexionResult(False, best_attempt, trials, memory)