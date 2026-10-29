import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Store:
    """Sparse keyword index using BM25Okapi algorithm."""

    def __init__(self):
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi = None

    def tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())

    def update_index(self, chunks: List[Dict[str, Any]]):
        """Rebuild or update BM25 index with current chunks."""
        self.corpus_chunks = chunks
        self.tokenized_corpus = [self.tokenize(c.get("content", "")) for c in chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform BM25 keyword similarity search."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair chunk with score and sort descending
        scored_pairs = []
        for idx, score in enumerate(scores):
            if score > 0:
                chunk = self.corpus_chunks[idx]
                scored_pairs.append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "content": chunk.get("content", ""),
                    "metadata": chunk.get("metadata", {}),
                    "score": float(score),
                    "retriever": "sparse_bm25"
                })

        scored_pairs.sort(key=lambda x: x["score"], reverse=True)
        return scored_pairs[:k]
