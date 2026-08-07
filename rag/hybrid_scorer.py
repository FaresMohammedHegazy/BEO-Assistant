import math
from collections import Counter
from typing import List, Dict, Any

class BM25Scorer:
    """A lightweight BM25 implementation for hybrid keyword scoring."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.doc_lengths = {}
        self.avgdl = 0
        self.num_docs = 0
        self.corpus_words = {}

    def fit(self, documents: List[Dict[str, Any]]) -> None:
        """Fits the scorer on a corpus of retrieved document candidates."""
        self.num_docs = len(documents)
        if self.num_docs == 0:
            return

        total_length = 0
        for doc in documents:
            doc_id = doc["id"]
            # Simple tokenization by splitting on whitespace
            text = doc.get("text", "").lower().split()
            self.doc_lengths[doc_id] = len(text)
            total_length += len(text)
            
            freqs = Counter(text)
            self.doc_freqs[doc_id] = freqs
            
            for word in freqs.keys():
                self.corpus_words[word] = self.corpus_words.get(word, 0) + 1
                
        self.avgdl = total_length / max(self.num_docs, 1)

    def score(self, query: str, doc_id: str) -> float:
        """Calculates the BM25 score for a specific query and document."""
        if doc_id not in self.doc_lengths:
            return 0.0
            
        query_words = query.lower().split()
        score = 0.0
        doc_len = self.doc_lengths[doc_id]
        freqs = self.doc_freqs[doc_id]
        
        for word in query_words:
            if word not in self.corpus_words:
                continue
                
            # Inverse Document Frequency
            idf = math.log(1 + (self.num_docs - self.corpus_words[word] + 0.5) / (self.corpus_words[word] + 0.5))
            # Term Frequency
            tf = freqs.get(word, 0)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avgdl, 1)))
            score += idf * (numerator / denominator)
            
        return score