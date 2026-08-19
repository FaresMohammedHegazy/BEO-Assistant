import json
import time
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

# 1. Force python to find the .env file in the root directory
root_dir = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(f"GROQ_API_KEY not found. Please check your .env file at {env_path}")

# Import the Self-RAG safeguard
from rag.self_rag import SelfRAG

# Import the three RAG architectures
try:
    from rag.retrievers import NaiveRAG, HybridRAG, AgenticRAG
except ImportError:
    print("Warning: Ensure NaiveRAG, HybridRAG, and AgenticRAG are properly defined in rag.retrievers")

def generate_answer(query: str, docs: List[Dict[str, Any]], client: Groq, model="openai/gpt-oss-120b") -> str:
    """Generate an answer based on the retrieved documents."""
    if not docs:
        return "No information found."
    
    context = "\n".join([doc.get("text", "") for doc in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer based ONLY on the context above."
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

def check_accuracy(answer: str, expected_concepts: List[str]) -> bool:
    """Check if the generated answer contains a reasonable match of expected concepts."""
    ans_lower = answer.lower()
    if not expected_concepts:
        return True
    matches = sum(1 for concept in expected_concepts if concept.lower() in ans_lower)
    return (matches / len(expected_concepts)) >= 0.5

def run_evaluation():
    # 1. Load the test questions
    questions_path = os.path.join(os.path.dirname(__file__), 'test_questions.json')
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # 2. Setup the LLM client using the verified API key
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    self_rag = SelfRAG()
    
    # Initialize the VectorStore first
    from rag.vector_store import VectorStore
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'aurelia.db')
    vector_store = VectorStore(db_path)

    # Initialize the retrieval systems using the vector_store
    systems = {
        "NaiveRAG": NaiveRAG(vector_store) if 'NaiveRAG' in globals() else None,
        "HybridRAG": HybridRAG(vector_store) if 'HybridRAG' in globals() else None,
        "AgenticRAG": AgenticRAG(vector_store) if 'AgenticRAG' in globals() else None
    }

    results = []

    print("\nStarting RAG Evaluation...\n" + "="*50)

    for q in questions:
        query = q['query']
        expected = q['expected_answer_concepts']
        
        for sys_name, system in systems.items():
            if system is None:
                continue
                
            print(f"Testing {sys_name} on question: {q['question_id']}...")
            
            # Measure retrieval latency
            start_time = time.time()
            
            # Retrieve documents
            try:
                docs = system.retrieve(query) if hasattr(system, 'retrieve') else system.search(query)
            except Exception as e:
                print(f"  [Error] Failed to retrieve: {e}")
                docs = []
            # Retrieve documents
            try:
                docs = system.retrieve(query) if hasattr(system, 'retrieve') else system.search(query)
            except Exception as e:
                print(f"  [Error] Failed to retrieve: {e}")
                docs = []
                
            print(f"--> Retrieved docs for query '{query}': {docs}")
            latency = time.time() - start_time
            
            # Check 1: Relevance
            is_relevant, rel_msg = self_rag.evaluate_retrieval(query, docs)
            
            # Generate Answer
            answer = generate_answer(query, docs, client)
            
            # Check 2: Support
            is_supported, sup_msg = self_rag.evaluate_support(query, answer, docs)
            
            # Check Accuracy
            is_accurate = check_accuracy(answer, expected)
            
            results.append({
                "System": sys_name,
                "Question": q['question_id'],
                "Accuracy": "Pass" if is_accurate else "Fail",
                "Latency (s)": round(latency, 3),
                "Self-RAG Rel.": "Pass" if is_relevant else "Fail",
                "Self-RAG Sup.": "Pass" if is_supported else "Fail"
            })

    # 3. Print the final comparison table
    print("\n" + "="*85)
    print(f"{'System':<15} | {'Question':<20} | {'Accuracy':<10} | {'Latency(s)':<10} | {'Self-RAG (Rel/Sup)'}")
    print("-" * 85)
    for r in results:
        self_rag_status = f"{r['Self-RAG Rel.']} / {r['Self-RAG Sup.']}"
        print(f"{r['System']:<15} | {r['Question']:<20} | {r['Accuracy']:<10} | {r['Latency (s)']:<10} | {self_rag_status}")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_evaluation()