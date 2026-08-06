import os
from typing import List, Optional

import numpy as np


class SentenceEmbeddingModel:
    """Simple embedding wrapper with a deterministic fallback for tests."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings = []
        for text in texts:
            tokens = [token.lower() for token in text.replace("\n", " ").split() if token]
            if not tokens:
                embeddings.append([0.0 for _ in range(8)])
                continue

            vector = []
            for token in tokens:
                value = sum(ord(char) for char in token) / max(len(token), 1)
                vector.append(float(value % 17) / 17.0)
            if len(vector) < 8:
                vector.extend([0.0] * (8 - len(vector)))
            else:
                vector = vector[:8]
            embeddings.append(vector)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]
