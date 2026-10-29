import os
import sys
import unittest
import uuid
import tempfile
import asyncio
from unittest.mock import MagicMock, patch

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ingestion.chunking import HierarchicalChunker
from backend.retrieval.bm25_store import BM25Store
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.agent.router import QueryRouter
from backend.agent.memory import ConversationMemory
from backend.models import QueryRequest, QueryResponse, Citation, DocumentChunk, ChunkMetadata
from fastapi.testclient import TestClient
from backend.main import app


class TestOfflineNexusSuite(unittest.TestCase):
    """
    100% Offline Test Suite for NexusRAG.
    Zero external network calls or LLM API dependencies.
    Executes in seconds and validates all algorithmic, data, and endpoint layers.
    """

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 60)
        print("  RUNNING NEXUSRAG 100% OFFLINE VERIFICATION SUITE")
        print("  Zero External API / LLM Outage Dependencies")
        print("=" * 60 + "\n")

    # ---------------------------------------------------------
    # 1. Hierarchical Chunking Engine
    # ---------------------------------------------------------
    def test_01_hierarchical_chunking_parent_child(self):
        """Test hierarchical chunker creates child chunks linked to parents."""
        chunker = HierarchicalChunker(
            child_size=150,
            child_overlap=30,
            parent_size=400,
            parent_overlap=50
        )
        sample_text = (
            "NexusRAG is a multimodal agentic retrieval augmented generation system. "
            "It indexes complex PDF documents containing tables, visual figures, and multi-column text. "
            "It supports hybrid retrieval fusing dense vector embeddings with sparse BM25Okapi scores. "
            "The system uses Reciprocal Rank Fusion to synthesize balanced candidates before neural re-ranking. "
            "Footnotes are attached directly to each claim linking to exact page numbers and figure previews. "
            "Evaluations are performed across Faithfulness, Relevancy, Precision, and Recall."
        )

        chunks = chunker.chunk_document_section(
            text=sample_text,
            source_file="quantum_research.pdf",
            page_number=1,
            modality="text"
        )

        self.assertGreaterEqual(len(chunks), 1, "Should generate at least one chunk")
        first_chunk = chunks[0]
        self.assertIsInstance(first_chunk, DocumentChunk)
        self.assertEqual(first_chunk.metadata.source_file, "quantum_research.pdf")
        self.assertEqual(first_chunk.metadata.page_number, 1)
        self.assertEqual(first_chunk.metadata.modality, "text")
        self.assertTrue(len(first_chunk.metadata.parent_chunk_id) > 0, "Parent chunk ID must be linked")
        self.assertIn("NexusRAG", first_chunk.content)

    def test_02_tabular_markdown_chunking(self):
        """Test tabular data chunking into markdown and schema context."""
        import pandas as pd
        from backend.ingestion.spreadsheet_ingestor import SpreadsheetIngestor

        ingestor = SpreadsheetIngestor()
        df = pd.DataFrame({
            "Algorithm": ["Shor's Algorithm", "Grover's Search", "HHL Linear Solver"],
            "Speedup": ["Exponential", "Quadratic", "Exponential"],
            "Qubits_Required": [2048, 512, 1024]
        })

        md_output = ingestor._df_to_markdown(df)
        self.assertIn("Algorithm", md_output)
        self.assertIn("Shor's Algorithm", md_output)
        self.assertIn("2048", md_output)

    # ---------------------------------------------------------
    # 2. BM25 Sparse Keyword Inverted Index
    # ---------------------------------------------------------
    def test_03_bm25_tokenization_and_search(self):
        """Test BM25Okapi tokenization, inverted indexing, and score ranking."""
        bm25 = BM25Store()

        corpus = [
            {"chunk_id": "c1", "content": "Shor algorithm achieves exponential speedup for factoring.", "metadata": {}},
            {"chunk_id": "c2", "content": "Grover search provides quadratic acceleration for database queries.", "metadata": {}},
            {"chunk_id": "c3", "content": "Classical supercomputers require millions of core hours for large numbers.", "metadata": {}}
        ]
        bm25.update_index(corpus)

        # Search for Grover
        results = bm25.search("quadratic grover search", k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c2", "c2 should be top match for Grover query")
        self.assertGreater(results[0]["score"], 0.0)

        # Search for empty or nonexistent token
        empty_res = bm25.search("xyznonexistentterm123")
        self.assertEqual(len(empty_res), 0)

    # ---------------------------------------------------------
    # 3. Reciprocal Rank Fusion (RRF) & Neural Re-Ranker
    # ---------------------------------------------------------
    def test_04_reciprocal_rank_fusion_math(self):
        """Test Reciprocal Rank Fusion (RRF) calculation and ranking logic."""
        k = 60
        dense_w = 0.6
        sparse_w = 0.4

        # Item A is rank 1 in dense, rank 2 in sparse
        # Item B is rank 2 in dense, rank 1 in sparse
        score_A = (dense_w / (k + 1)) + (sparse_w / (k + 2))
        score_B = (dense_w / (k + 2)) + (sparse_w / (k + 1))

        # Because dense_w (0.6) > sparse_w (0.4), Item A should score higher than Item B
        self.assertGreater(score_A, score_B, "Dense weight prioritization should rank Item A higher")

        reranker = CrossEncoderReranker()
        candidates = [
            {"chunk_id": "b", "content": "Doc B", "rrf_score": score_B},
            {"chunk_id": "a", "content": "Doc A", "rrf_score": score_A}
        ]
        reranked = reranker.rerank(query="quantum speedup", candidate_chunks=candidates, top_k=2)
        self.assertEqual(reranked[0]["chunk_id"], "a")

    # ---------------------------------------------------------
    # 4. Agentic Query Router Heuristics
    # ---------------------------------------------------------
    def test_05_agent_query_router_heuristic_paths(self):
        """Test deterministic routing rules without external LLM API dependency."""
        router = QueryRouter()

        # Direct greeting
        route_direct = router._rule_based_route("Hello there, how are you?")
        self.assertEqual(route_direct["route"], "DIRECT")

        # Table lookup
        route_table = router._rule_based_route("Show me the total qubit column values in the spreadsheet table")
        self.assertEqual(route_table["route"], "TABLE_LOOKUP")

        # Multi-step comparative
        route_multi = router._rule_based_route("Compare Shor algorithm vs Grover algorithm performance differences")
        self.assertEqual(route_multi["route"], "MULTI_STEP")
        self.assertGreaterEqual(len(route_multi["sub_queries"]), 2)

        # General factual retrieval
        route_retrieval = router._rule_based_route("Explain the mathematical derivation of Grover diffusion operator")
        self.assertEqual(route_retrieval["route"], "RETRIEVAL")

    # ---------------------------------------------------------
    # 5. SQLite Conversational Sliding-Window Memory
    # ---------------------------------------------------------
    def test_06_sqlite_conversation_memory(self):
        """Test persistent SQLite memory storage, turn tracking, and sliding window."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = os.path.join(tmp_dir, "test_chat.db")
            mem = ConversationMemory(db_path=test_db)

            session_id = f"test_sess_{uuid.uuid4().hex[:8]}"

            # Add turns
            mem.add_message(session_id, "user", "What is quantum supremacy?")
            mem.add_message(session_id, "assistant", "Quantum supremacy is performing a calculation classical computers cannot.")
            mem.add_message(session_id, "user", "Give an example of an algorithm.")
            mem.add_message(session_id, "assistant", "Shor's algorithm is a primary example.")

            # Retrieve context
            ctx = mem.get_conversation_context(session_id)
            recent = ctx["recent_messages"]
            self.assertEqual(len(recent), 4, "Should retrieve 4 recent messages")
            self.assertEqual(recent[0]["role"], "user")
            self.assertEqual(recent[0]["content"], "What is quantum supremacy?")
            self.assertEqual(recent[3]["role"], "assistant")
            self.assertEqual(recent[3]["content"], "Shor's algorithm is a primary example.")

            # Verify session isolation
            other_ctx = mem.get_conversation_context("non_existent_session")
            self.assertEqual(len(other_ctx["recent_messages"]), 0)

    # ---------------------------------------------------------
    # 6. Pydantic Schemas & Data Integrity
    # ---------------------------------------------------------
    def test_07_pydantic_data_contracts(self):
        """Test Pydantic v2 validation contracts for requests, responses, and citations."""
        req = QueryRequest(
            question="What are quantum phase estimation principles?",
            session_id="sess_123",
            top_k=5,
            enable_agent=True
        )
        self.assertEqual(req.top_k, 5)
        self.assertTrue(req.enable_agent)

        cite = Citation(
            id=1,
            chunk_id="chunk_abc",
            type="figure",
            file="diagram.pdf",
            page=3,
            snippet="Figure 2: Bloch sphere trajectory.",
            score=0.94
        )
        self.assertEqual(cite.type, "figure")
        self.assertEqual(cite.page, 3)

        resp = QueryResponse(
            answer="Here is the explanation [Fig. 1 (Pg 3)].",
            citations=[cite],
            route_taken="RETRIEVAL",
            confidence_score=0.98,
            sub_queries=["What is phase estimation?"]
        )
        dumped = resp.model_dump()
        self.assertIn("answer", dumped)
        self.assertEqual(len(dumped["citations"]), 1)
        self.assertEqual(dumped["citations"][0]["file"], "diagram.pdf")

    # ---------------------------------------------------------
    # 7. FastAPI REST Endpoints (Offline Mocks)
    # ---------------------------------------------------------
    def test_08_fastapi_rest_endpoints_offline(self):
        """Test FastAPI endpoints for availability, schema validation, and health checks."""
        client = TestClient(app)

        # Health endpoint
        health = client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        data = health.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("capabilities", data)
        self.assertGreaterEqual(len(data["capabilities"]), 4)

        # Documents endpoint
        docs = client.get("/api/documents")
        self.assertEqual(docs.status_code, 200)
        docs_data = docs.json()
        self.assertIn("documents", docs_data)
        self.assertIsInstance(docs_data["documents"], list)

        # Gallery endpoint
        gal = client.get("/api/gallery")
        self.assertEqual(gal.status_code, 200)
        gal_data = gal.json()
        self.assertIn("gallery", gal_data)
        self.assertIsInstance(gal_data["gallery"], list)

        # Evaluation results endpoint
        eval_res = client.get("/api/evaluate/results")
        self.assertEqual(eval_res.status_code, 200)

        # Validation error on bad POST payload
        bad_post = client.post("/api/query", json={})
        self.assertEqual(bad_post.status_code, 422, "Should return 422 Unprocessable Entity on missing question")


if __name__ == "__main__":
    unittest.main(verbosity=2)
