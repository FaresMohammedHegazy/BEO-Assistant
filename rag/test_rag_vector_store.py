import os
import tempfile
import unittest

from rag.embedder import SentenceEmbeddingModel
from rag.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="rag-test-")
        self.store_path = os.path.join(self.temp_dir, "rag.sqlite")

    def tearDown(self):
        if os.path.exists(self.store_path):
            os.remove(self.store_path)

    def test_initialization_and_metadata_filtering(self):
        embedder = SentenceEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        store = VectorStore(store_path=self.store_path, embedder=embedder)
        store.initialize()

        documents = [
            {"id": "doc-1", "text": "Aurelia hotel policy for fire safety", "metadata": {"source": "policy", "category": "safety"}},
            {"id": "doc-2", "text": "Banquet event ordering guidelines", "metadata": {"source": "operations", "category": "planning"}},
        ]

        store.add_documents(documents)

        results = store.search("hotel policy", filters={"source": "policy"}, top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "doc-1")

        filtered = store.search("event ordering", filters={"category": "planning"}, top_k=3)
        self.assertTrue(filtered)
        self.assertEqual(filtered[0]["id"], "doc-2")


if __name__ == "__main__":
    unittest.main()
