import os
from typing import List, Dict, Any, Optional
from groq import Groq

from .vector_store import VectorStore
from .hybrid_scorer import BM25Scorer

class NaiveRAG:
    """Baseline RAG: Direct vector similarity search."""
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, filters=filters, top_k=top_k)


class HybridRAG:
    """Hybrid RAG: Combines ANN Vector similarity with BM25 keyword scoring."""
    def __init__(self, vector_store: VectorStore, alpha: float = 0.5):
        self.vector_store = vector_store
        self.alpha = alpha  # 1.0 = Pure Vector, 0.0 = Pure BM25

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        # 1. Broad fetch from Vector Store (fetch more to rerank)
        candidates = self.vector_store.search(query, filters=filters, top_k=top_k * 3)
        if not candidates:
            return []

        # 2. Fit BM25 on the candidate subset
        scorer = BM25Scorer()
        scorer.fit(candidates)

        results = []
        
        # 3. Get score distributions to normalize them (Min-Max Scaling)
        v_scores = [c.get("score", 0.0) for c in candidates]
        max_v, min_v = max(v_scores), min(v_scores)

        for doc in candidates:
            # Normalize Vector Score
            v_raw = doc.get("score", 0.0)
            v_norm = (v_raw - min_v) / (max_v - min_v + 1e-9) if max_v != min_v else 1.0
            
            # Calculate and append BM25 Score
            doc["bm25_score"] = scorer.score(query, doc["id"])
            results.append(doc)
            
        b_scores = [c["bm25_score"] for c in results]
        max_b, min_b = max(b_scores), min(b_scores)
        
        # 4. Compute Final Hybrid Score
        for doc in results:
            b_raw = doc["bm25_score"]
            b_norm = (b_raw - min_b) / (max_b - min_b + 1e-9) if max_b != min_b else 1.0
            
            doc["hybrid_score"] = (self.alpha * v_norm) + ((1 - self.alpha) * b_norm)
            
        # 5. Sort by aggregated score and return top_k
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]


class AgenticRAG:
    """Agentic RAG: Uses LLM reasoning to expand queries and intelligently fetch documents."""
    def __init__(self, vector_store: VectorStore, hybrid_rag: Optional[HybridRAG] = None):
        self.vector_store = vector_store
        self.hybrid_rag = hybrid_rag or HybridRAG(vector_store)
        
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = Groq(api_key=api_key) if api_key else None
        self.model = os.getenv("MODEL_NAME", "llama3-8b-8192")

    def _generate_queries(self, user_query: str) -> List[str]:
        """Reasoning loop: Expand the original query into multiple semantic angles."""
        if not self.llm:
            return [user_query]
            
        prompt = (
            f"You are a retrieval optimization agent. The user is asking: '{user_query}'\n"
            "Generate 3 distinct, highly targeted search queries to extract the necessary context from a vector database. "
            "Return ONLY the queries, separated by newlines, with no additional text."
        )
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            text = response.choices[0].message.content.strip()
            return [q.strip() for q in text.split('\n') if q.strip()]
        except Exception as e:
            print(f"Agentic reasoning failed: {e}")
            return [user_query]

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        # 1. Expand query space
        queries = self._generate_queries(query)
        if query not in queries:
            queries.append(query)
            
        # 2. Retrieve across all queries using Hybrid RAG
        all_results = {}
        for q in queries:
            res = self.hybrid_rag.retrieve(q, filters=filters, top_k=top_k)
            for doc in res:
                doc_id = doc["id"]
                # Boost documents that appear across multiple query angles
                if doc_id in all_results:
                    all_results[doc_id]["agentic_score"] += doc.get("hybrid_score", 0.5)
                else:
                    doc["agentic_score"] = doc.get("hybrid_score", 0.5)
                    all_results[doc_id] = doc
                    
        # 3. Sort by consolidated score
        final_docs = list(all_results.values())
        final_docs.sort(key=lambda x: x.get("agentic_score", 0.0), reverse=True)
        
        return final_docs[:top_k]