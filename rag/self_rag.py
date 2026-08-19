import os
from typing import List, Dict, Any, Tuple
from groq import Groq

class SelfRAG:
    """Explicit safeguard to prevent the model from trusting bad returns."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = Groq(api_key=api_key) if api_key else None
        self.model = model_name

    def evaluate_retrieval(self, query: str, documents: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Check 1: Relevance (Are the retrieved documents relevant to the query?)
        """
        if not self.llm or not documents:
            return False, "No documents retrieved or LLM unavailable."
            
        context = "\n".join([doc.get("text", "") for doc in documents])
        prompt = (
            f"You are a strict evaluator. The user asked: '{query}'\n"
            f"Retrieved Documents:\n{context}\n\n"
            "Does the retrieved context contain information relevant to answering the user's query? "
            "Answer ONLY with 'YES' or 'NO'."
        )
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip().upper()
            is_relevant = "YES" in answer
            return is_relevant, "Relevant" if is_relevant else "Irrelevant context."
        except Exception as e:
            return False, f"Error checking relevance: {str(e)}"

    def evaluate_support(self, query: str, generated_answer: str, documents: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Check 2: Support/Hallucination (Is the answer fully supported by the text?)
        """
        if not self.llm or not documents:
            return False, "Cannot verify support without documents."
            
        context = "\n".join([doc.get("text", "") for doc in documents])
        prompt = (
            f"You are a strict fact-checker. The user asked: '{query}'\n"
            f"The proposed answer is: '{generated_answer}'\n"
            f"The retrieved facts are:\n{context}\n\n"
            "Is the proposed answer fully supported by the retrieved facts without adding any outside, unverified information? "
            "Answer ONLY with 'YES' (if fully supported) or 'NO' (if hallucinated or unsupported)."
        )
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip().upper()
            is_supported = "YES" in answer
            return is_supported, "Fully supported" if is_supported else "Hallucination detected."
        except Exception as e:
            return False, f"Error checking support: {str(e)}"