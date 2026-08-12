import os
import sys

# Add the root repository directory to Python's path FIRST
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv
load_dotenv() 

from langchain_groq import ChatGroq
from planning.planning_lab.algorithms.tree_of_thoughts import tree_of_thoughts_grounded

# Focus exclusively on the breakout room selection 
BREAKOUT_REQUEST = (
    "We need 4 parallel breakout tracks for the summit (40 people each). "
    "Select a combination of 4 distinct rooms. Standard Conference Rooms "
    "usually start at ROOM_102. Output the exact ROOM_ IDs you propose."
)

def run_tot_demo():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

    print("==========================================================")
    print("TREE OF THOUGHTS (GROUNDED DB EVALUATION)")
    print("==========================================================")
    print(f"[Goal]: {BREAKOUT_REQUEST}\n")
    print("Generating and evaluating room combinations using real SQLite constraints...")
    
    thoughts = tree_of_thoughts_grounded(
        problem=BREAKOUT_REQUEST,
        llm=llm,
        depth=2,
        beam_width=2
    )
    
    print("\n==========================================================")
    print("[Final Surviving Thoughts]:")
    for i, thought in enumerate(thoughts, 1):
        print(f"\nOption {i} (Score: {thought.score}):")
        print(f"State: {thought.state}")
        print(f"Rationale: {thought.rationale}")

if __name__ == "__main__":
    run_tot_demo()