import os
import sys

# Add the root repository directory to Python's path FIRST
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv
load_dotenv() 

from langchain_groq import ChatGroq
from planning.planning_lab.algorithms.environment import Environment
from planning.planning_lab.algorithms.reflexion import reflexion

# Scenario: Booking an event but hitting a strict fire-code constraint.
# The DB limits ROOM_101 to exactly 300.
BOOKING_REQUEST = (
    "Draft a booking summary for EVT_999 into ROOM_101. "
    "We have a requested_headcount of 350 attendees. "
    "Output the room ID, event ID, and requested_headcount. Do NOT ask clarifying questions."
)

def run_reflexion_demo():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
    
    # The grounded environment automatically checks the SQLite database constraints
    environment = Environment(success_threshold=1.0)
    
    print("==========================================================")
    print("REFLEXION (GROUNDED FULL-PLAN RETRIES)")
    print("==========================================================")
    print(f"[Goal]: {BOOKING_REQUEST}\n")
    
    # Shared persistent memory across potential multiple top-level runs
    persistent_memory = []
    
    print("Running Reflexion Planner...")
    result = reflexion(
        task=BOOKING_REQUEST,
        llm=llm,
        environment=environment,
        max_trials=3,
        memory_size=2,
        memory=persistent_memory
    )
    
    print(f"\n[Final Success]: {result.success}")
    
    print("\n[Trial Breakdown]:")
    for trial in result.trials:
        print(f"\n  Trial {trial.number}:")
        print(f"  Attempt Snippet: {trial.attempt[:100]}...")
        print(f"  Feedback Score : {trial.feedback.score}")
        if not trial.feedback.success:
            print(f"  DB Feedback    : {trial.feedback.details[0]}")
            print(f"  Reflection     : {trial.reflection}")
            
    print("\n==========================================================")
    print(f"[Final Output]:\n{result.output}")
    print("\n[Final Persistent Memory State]:")
    for mem in persistent_memory:
        print(f"- {mem}")

if __name__ == "__main__":
    run_reflexion_demo()