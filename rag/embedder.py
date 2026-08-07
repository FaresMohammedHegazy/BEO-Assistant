import os
from typing import List, Optional

from sentence_transformers import SentenceTransformer


class SentenceEmbeddingModel:
    """Wraps a real sentence-transformers model to produce semantic embeddings."""

    # Cache loaded models by name so repeated instantiation (e.g. VectorStore()
    # created per-request) doesn't reload multi-hundred-MB weights every time.
    _model_cache: dict[str, SentenceTransformer] = {}

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        if self.model_name not in self._model_cache:
            self._model_cache[self.model_name] = SentenceTransformer(self.model_name)
        self._model = self._model_cache[self.model_name]

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # normalize_embeddings=True -> vectors are unit-length, so a plain dot
        # product (as used in vector_store.search) is equivalent to cosine similarity.
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]