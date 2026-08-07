import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embedder import SentenceEmbeddingModel


class VectorStore:
    """A lightweight vector store backed by SQLite with ANN-style cosine search."""

    def __init__(self, store_path: str, embedder: Optional[SentenceEmbeddingModel] = None):
        self.store_path = store_path
        self.embedder = embedder or SentenceEmbeddingModel()
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.store_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                document_id TEXT PRIMARY KEY,
                vector_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_index (
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                document_id TEXT NOT NULL,
                PRIMARY KEY (key, value, document_id)
            )
            """
        )
        self._conn.commit()

    def add_documents(self, documents: Sequence[Dict[str, Any]]) -> None:
        if self._conn is None:
            self.initialize()

        for item in documents:
            document_id = str(item["id"])
            text = str(item["text"])
            metadata = item.get("metadata") or {}
            chunk_index = int(item.get("chunk_index", 0))
            self._conn.execute(
                "INSERT OR REPLACE INTO documents (id, text, chunk_index, metadata_json) VALUES (?, ?, ?, ?)",
                (document_id, text, chunk_index, json.dumps(metadata)),
            )

            embedding = self.embedder.embed_query(text)
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (document_id, vector_json) VALUES (?, ?)",
                (document_id, json.dumps(embedding)),
            )

            for key, value in metadata.items():
                self._conn.execute(
                    "INSERT OR REPLACE INTO metadata_index (key, value, document_id) VALUES (?, ?, ?)",
                    (str(key), str(value), document_id),
                )

        self._conn.commit()

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        
        if not words:
            return [""]
            
        for idx in range(0, len(words), max(1, chunk_size - overlap)):
            chunk = " ".join(words[idx : idx + chunk_size])
            chunks.append(chunk)
            if idx + chunk_size >= len(words):
                break
                
        return chunks

    def add_texts(self, texts: Sequence[str], metadata: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        if metadata is None:
            metadata = [{} for _ in range(len(texts))]

        documents = []
        for index, text in enumerate(texts):
            item_metadata = metadata[index] if index < len(metadata) else {}
            for chunk_index, chunk in enumerate(self.chunk_text(text)):
                documents.append(
                    {
                        "id": f"chunk-{index}-{chunk_index}",
                        "text": chunk,
                        "metadata": {**item_metadata, "chunk_index": chunk_index},
                    }
                )
        self.add_documents(documents)

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        if self._conn is None:
            self.initialize()

        query_vector = np.array(self.embedder.embed_query(query), dtype=float)
        
        filtered_ids = None
        if filters:
            filtered_ids = []
            for key, value in filters.items():
                matching_ids = [
                    row["document_id"]
                    for row in self._conn.execute(
                        "SELECT document_id FROM metadata_index WHERE key = ? AND value = ?",
                        (str(key), str(value)),
                    ).fetchall()
                ]
                if not matching_ids:
                    filtered_ids = []
                    break
                if not filtered_ids:
                    filtered_ids = matching_ids
                else:
                    filtered_ids = [doc_id for doc_id in filtered_ids if doc_id in matching_ids]

        if filters is not None and not filtered_ids:
            rows = []
        elif filtered_ids:
            placeholders = ", ".join("?" for _ in filtered_ids)
            query_sql = f"""
                SELECT d.id, d.text, d.metadata_json, e.vector_json 
                FROM documents d
                JOIN embeddings e ON d.id = e.document_id
                WHERE d.id IN ({placeholders})
            """
            rows = self._conn.execute(query_sql, filtered_ids).fetchall()
        else:
            query_sql = """
                SELECT d.id, d.text, d.metadata_json, e.vector_json 
                FROM documents d
                JOIN embeddings e ON d.id = e.document_id
            """
            rows = self._conn.execute(query_sql).fetchall()

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            if row["vector_json"] is None:
                continue
            
            doc_vector = np.array(json.loads(row["vector_json"]), dtype=float)
            
            similarity = float(np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector) + 1e-9))
            
            scored.append((
                similarity, 
                {
                    "id": row["id"], 
                    "text": row["text"], 
                    "metadata": json.loads(row["metadata_json"]), 
                    "score": similarity
                }
            ))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]