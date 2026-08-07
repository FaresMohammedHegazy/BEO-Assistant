import os
import tempfile
import unittest
from unittest.mock import patch

from rag.embedder import SentenceEmbeddingModel
from rag.vector_store import VectorStore
from rag.hybrid_scorer import BM25Scorer
from rag.retrievers import NaiveRAG, HybridRAG, AgenticRAG

class TestRetrievers(unittest.TestCase):
    def setUp(self):
        """Set up a temporary VectorStore and populate it with test documents."""
        self.temp_dir = tempfile.mkdtemp(prefix="rag-test-")
        self.store_path = os.path.join(self.temp_dir, "rag.sqlite")
        
        # Initialize the embedder and store
        self.embedder = SentenceEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.store = VectorStore(store_path=self.store_path, embedder=self.embedder)
        self.store.initialize()
        
        # Add sample documents to test retrieval
        documents = [
            {
                "id": "doc-1", 
                "text": "The quick brown fox jumps over the lazy dog.", 
                "metadata": {"category": "animal"}
            },
            {
                "id": "doc-2", 
                "text": "Aurelia hotel strict fire safety policy and compliance.", 
                "metadata": {"category": "policy"}
            },
            {
                "id": "doc-3", 
                "text": "Banquet event ordering and VIP guest menu planning guidelines.", 
                "metadata": {"category": "event"}
            },
        ]
        self.store.add_documents(documents)

    def tearDown(self):
        """Clean up the temporary database file after tests."""
        # 1. Close the database connection to release the Windows file lock
        if getattr(self.store, '_conn', None):
            self.store._conn.close()
            
        # 2. Use shutil to delete the entire temp folder (including hidden WAL files)
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bm25_scorer(self):
        """Test that the BM25 algorithm correctly scores exact keyword matches."""
        docs = [
            {"id": "d1", "text": "apple banana apple"},
            {"id": "d2", "text": "banana cherry"},
        ]
        scorer = BM25Scorer()
        scorer.fit(docs)
        
        # 'apple' appears twice in d1 and zero times in d2.
        score_d1 = scorer.score("apple", "d1")
        score_d2 = scorer.score("apple", "d2")
        
        self.assertGreater(score_d1, score_d2)
        self.assertEqual(score_d2, 0.0)

    def test_naive_rag(self):
        """Test direct vector similarity search."""
        rag = NaiveRAG(self.store)
        results = rag.retrieve("hotel fire policy", top_k=1)
        
        self.assertEqual(len(results), 1)
        self.assertIn("score", results[0], "NaiveRAG must return a standard vector score.")

    def test_hybrid_rag(self):
        """Test the combination of vector similarity and BM25 scoring."""
        rag = HybridRAG(self.store, alpha=0.5)
        results = rag.retrieve("VIP event menu", top_k=2)
        
        self.assertTrue(len(results) > 0)
        self.assertIn("bm25_score", results[0], "HybridRAG must append BM25 scores.")
        self.assertIn("hybrid_score", results[0], "HybridRAG must calculate a final hybrid score.")

    def test_agentic_rag(self):
        """Test the multi-query expansion and document aggregation."""
        rag = AgenticRAG(self.store)
        
        # Mock the LLM query generation to avoid hitting the actual Groq API
        mocked_queries = ["banquet event", "VIP menu restrictions"]
        
        with patch.object(rag, '_generate_queries', return_value=mocked_queries):
            results = rag.retrieve("VIP event", top_k=2)
            
            self.assertTrue(len(results) > 0)
            self.assertIn("agentic_score", results[0], "AgenticRAG must aggregate scores into an agentic_score.")

if __name__ == "__main__":
    unittest.main()