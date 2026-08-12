import os
import re
import sqlite3
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from ..models import Thought


class ThoughtCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # FIX: Increased max_length to 8 to prevent Groq API validation crashes 
    # if the LLM accidentally generates too many array items.
    candidates: list[str] = Field(min_length=1, max_length=8)


class ThoughtEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    rationale: str


def tree_of_thoughts(
    problem: str,
    llm: BaseChatModel,
    depth: int = 2,
    beam_width: int = 2,
) -> list[Thought]:
    """Original non-grounded Tree of Thoughts loop."""
    frontier = [Thought(state="Start", score=0.5, rationale="root")]
    for _ in range(depth):
        candidates: list[Thought] = []
        for parent in frontier:
            generated = llm.with_structured_output(
                ThoughtCandidates
            ).invoke([
                ("system", "Generate distinct candidate next steps for Tree-of-Thoughts search."),
                ("human", f"Problem: {problem}\nPartial path: {parent.state}\nPropose two distinct promising continuations."),
            ], temperature=0.5)
            for state in generated.candidates[:2]:
                judged = llm.with_structured_output(
                    ThoughtEvaluation
                ).invoke([
                    ("system", "Independently evaluate a partial solution."),
                    ("human", f"Problem: {problem}\nCandidate path: {state}\nScore correctness, feasibility, and progress. Do not reward confident wording."),
                ], temperature=0.1)
                candidates.append(
                    Thought(state=state, score=judged.score, rationale=judged.rationale)
                )
        frontier = sorted(candidates, key=lambda item: item.score, reverse=True)[:beam_width]
        if not frontier:
            break
    return frontier


def evaluate_thought_grounded(state: str) -> ThoughtEvaluation:
    """
    Evaluate a candidate thought by querying the real database constraints
    rather than relying on LLM heuristics.
    """
    # FIX: Corrected path resolution to accurately target the root directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    db_path = os.path.join(repo_root, "db", "aurelia.db")
    
    # Extract proposed room IDs and assume a standard breakout headcount of 40
    room_matches = re.findall(r"\bROOM_\d+\b", state, flags=re.IGNORECASE)
    headcount = 40 
    
    if not room_matches:
        return ThoughtEvaluation(score=0.2, rationale="No specific room IDs proposed.")
        
    unique_rooms = list(set(room_matches))
    if len(unique_rooms) < len(room_matches):
        return ThoughtEvaluation(score=0.3, rationale="Duplicate rooms proposed for parallel tracks.")
        
    violations = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for room_id in unique_rooms:
                room_id = room_id.upper()
                cursor.execute("SELECT max_capacity, fire_code_status FROM rooms WHERE room_id = ?", (room_id,))
                room = cursor.fetchone()
                
                if not room:
                    violations.append(f"{room_id} does not exist in the database.")
                    continue
                    
                max_cap, fire_status = room
                
                if headcount > max_cap:
                    if fire_status == "STRICT_ENFORCEMENT":
                        violations.append(f"{room_id} has a STRICT fire code limit of {max_cap}. {headcount} violates this.")
                    else:
                        violations.append(f"{room_id} max capacity {max_cap} is too small for {headcount}.")
    except Exception as e:
        return ThoughtEvaluation(score=0.0, rationale=f"Database error: {str(e)}")
        
    if violations:
        penalty = 0.3 * len(violations)
        return ThoughtEvaluation(
            score=max(0.1, 1.0 - penalty),
            rationale="Constraints failed: " + " | ".join(violations)
        )
        
    return ThoughtEvaluation(score=1.0, rationale="Valid room combination. All DB constraints met.")


def tree_of_thoughts_grounded(
    problem: str,
    llm: BaseChatModel,
    depth: int = 2,
    beam_width: int = 2,
) -> list[Thought]:
    """Grounded ToT loop: Replaces LLM evaluation with hard database checks."""
    frontier = [Thought(state="Start", score=0.5, rationale="root")]
    for _ in range(depth):
        candidates: list[Thought] = []
        for parent in frontier:
            # 1. Generation Phase (LLM proposes room combinations)
            generated = llm.with_structured_output(
                ThoughtCandidates
            ).invoke([
                # FIX: Explicitly instruct the LLM on what a "candidate" should look like
                ("system", "Generate distinct candidate next steps for Tree-of-Thoughts search. Each candidate must be a COMPLETE proposed combination of rooms (e.g. a single string containing 4 exact ROOM_XXX identifiers)."),
                ("human", f"Problem: {problem}\nPartial path: {parent.state}\nPropose two distinct promising continuations (combinations)."),
            ], temperature=0.5)
            
            for state in generated.candidates[:2]:
                # 2. Grounded Evaluation Phase (Direct DB lookup replaces LLM guesswork)
                judged = evaluate_thought_grounded(state)
                candidates.append(
                    Thought(state=state, score=judged.score, rationale=judged.rationale)
                )
        frontier = sorted(candidates, key=lambda item: item.score, reverse=True)[:beam_width]
        if not frontier:
            break
    return frontier