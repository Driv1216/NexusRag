import os
from typing import List, Dict, Any
from backend.config import RERANKER_MODEL

class CrossEncoderReranker:
    """Re-ranks retrieved candidates using a dedicated Cross-Encoder model or RRF scoring."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CrossEncoderReranker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = None
        if os.getenv("ENABLE_RERANKER_DOWNLOAD", "false").lower() == "true":
            try:
                from sentence_transformers import CrossEncoder
                print(f"Loading Cross-Encoder model: {RERANKER_MODEL}...")
                self.model = CrossEncoder(RERANKER_MODEL, max_length=512)
                print("Cross-Encoder re-ranker loaded successfully.")
            except Exception as e:
                print(f"Note: Cross-Encoder model not loaded ({e}). Using weighted hybrid RRF scoring.")
                self.model = None
        else:
            self.model = None
        self._initialized = True

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Re-rank candidate chunks based on cross-attention or RRF scores."""
        if not candidate_chunks:
            return []

        if self.model is None:
            # High-precision RRF fallback: sort by existing rrf_score or score
            candidate_chunks.sort(
                key=lambda x: x.get("rrf_score") or x.get("score") or 0.0,
                reverse=True
            )
            for idx, c in enumerate(candidate_chunks):
                if "rerank_score" not in c:
                    c["rerank_score"] = round(c.get("rrf_score") or c.get("score") or (0.95 - idx * 0.05), 4)
            return candidate_chunks[:top_k]

        pairs = [[query, c.get("content", "")[:1000]] for c in candidate_chunks]
        try:
            scores = self.model.predict(pairs)
            for idx, s in enumerate(scores):
                prob = float(1.0 / (1.0 + (2.71828 ** (-s))))
                candidate_chunks[idx]["rerank_score"] = round(prob, 4)

            candidate_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return candidate_chunks[:top_k]
        except Exception as e:
            print(f"Reranking error: {e}. Falling back to initial order.")
            return candidate_chunks[:top_k]
