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

    def test_add_document_returns_id_and_is_searchable(self):
        embedder = SentenceEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        store = VectorStore(store_path=self.store_path, embedder=embedder)
        store.initialize()

        doc_id = store.add_document("Aurelia deposit refund policy", metadata={"source": "policy"})
        self.assertTrue(doc_id.startswith("doc-"))

        results = store.search("deposit refund policy", top_k=3)
        self.assertTrue(any(r["id"] == doc_id for r in results))

    def test_delete_document_removes_from_next_search_without_restart(self):
        embedder = SentenceEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        store = VectorStore(store_path=self.store_path, embedder=embedder)
        store.initialize()

        doc_id = store.add_document("Aurelia vendor cancellation policy", metadata={"source": "policy"})
        self.assertTrue(any(r["id"] == doc_id for r in store.search("vendor cancellation", top_k=3)))

        deleted = store.delete_document(doc_id)
        self.assertTrue(deleted)

        # Same store instance, no re-initialize() / restart -- the very next
        # query must no longer surface the deleted document.
        results_after = store.search("vendor cancellation", top_k=3)
        self.assertFalse(any(r["id"] == doc_id for r in results_after))

        # Deleting an id that no longer exists reports False, not an error.
        self.assertFalse(store.delete_document(doc_id))

    def test_readd_document_clears_stale_metadata(self):
        embedder = SentenceEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        store = VectorStore(store_path=self.store_path, embedder=embedder)
        store.initialize()

        store.add_documents([
            {"id": "doc-stale", "text": "Original text", "metadata": {"category": "old"}}
        ])
        self.assertTrue(store.search("original text", filters={"category": "old"}, top_k=3))

        # Re-add the SAME id with a completely different metadata key.
        store.add_documents([
            {"id": "doc-stale", "text": "Updated text", "metadata": {"topic": "new"}}
        ])

        # The stale "category: old" filter must no longer match this document.
        stale_match = store.search("updated text", filters={"category": "old"}, top_k=3)
        self.assertFalse(any(r["id"] == "doc-stale" for r in stale_match))

        fresh_match = store.search("updated text", filters={"topic": "new"}, top_k=3)
        self.assertTrue(any(r["id"] == "doc-stale" for r in fresh_match))


if __name__ == "__main__":
    unittest.main()
