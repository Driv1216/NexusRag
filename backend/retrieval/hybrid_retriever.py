from typing import List, Dict, Any, Optional
from backend.retrieval.vector_store import ChromaVectorStore
from backend.retrieval.bm25_store import BM25Store
from backend.retrieval.reranker import CrossEncoderReranker
from backend.config import (
    DEFAULT_TOP_K,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
    RRF_K_CONSTANT
)

class HybridRetriever:
    """
    Hybrid Retriever with:
    1. Dense similarity search (ChromaDB + SentenceTransformers / CLIP)
    2. Sparse BM25 keyword search (rank_bm25)
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-Encoder neural re-ranking
    5. Parent context expansion
    """

    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store
        self.bm25_store = BM25Store()
        self.reranker = CrossEncoderReranker()
        self._refresh_bm25()

    def _refresh_bm25(self):
        """Sync BM25 index with current ChromaDB records."""
        all_chunks = self.vector_store.get_all_chunks()
        self.bm25_store.update_index(all_chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        modality_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        enable_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute hybrid search pipeline."""
        # 1. Dense retrieval (fetch 2x top_k)
        candidate_k = max(top_k * 3, 15)
        dense_results = self.vector_store.similarity_search(
            query_text=query,
            k=candidate_k,
            modality_filter=modality_filter,
            source_filter=source_filter
        )

        # 2. Sparse BM25 retrieval
        sparse_results = self.bm25_store.search(query=query, k=candidate_k)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunks_map: Dict[str, Dict[str, Any]] = {}

        # Process dense rankings
        for rank, item in enumerate(dense_results, start=1):
            c_id = item["chunk_id"]
            chunks_map[c_id] = item
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (
                HYBRID_DENSE_WEIGHT / (RRF_K_CONSTANT + rank)
            )

        # Process sparse rankings
        for rank, item in enumerate(sparse_results, start=1):
            c_id = item["chunk_id"]
            if c_id not in chunks_map:
                chunks_map[c_id] = item
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (
                HYBRID_SPARSE_WEIGHT / (RRF_K_CONSTANT + rank)
            )

        # Sort combined items by RRF score
        fused_candidates = []
        for c_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            item = chunks_map[c_id]
            item["rrf_score"] = round(score, 6)
            fused_candidates.append(item)

        candidates_pool = fused_candidates[:candidate_k]

        # 4. Neural Cross-Encoder Re-ranking
        if enable_rerank and candidates_pool:
            final_results = self.reranker.rerank(query=query, candidate_chunks=candidates_pool, top_k=top_k)
        else:
            final_results = candidates_pool[:top_k]

        return final_results
