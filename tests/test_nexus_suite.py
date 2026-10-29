import os
import sys
import unittest
import uuid
from PIL import Image, ImageDraw
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_sample_chart(filepath: str):
    img = Image.new("RGB", (300, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([30, 160, 270, 162], fill=(40, 40, 40))
    d.rectangle([50, 80, 80, 160], fill=(59, 130, 246))
    d.rectangle([110, 50, 140, 160], fill=(16, 185, 129))
    d.rectangle([170, 100, 200, 160], fill=(245, 158, 11))
    img.save(filepath)
    return filepath

def create_sample_pdf(filepath: str, chart_path: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(filepath, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 720, "Multimodal Quantum Computing Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 680, "Shor algorithm factors integers in polynomial time O((log N)^3).")
    c.drawString(50, 650, "Grover search provides quadratic speedup for unstructured databases.")
    c.drawImage(chart_path, 50, 380, width=280, height=180)
    c.drawString(50, 350, "Figure 1: Benchmark execution time comparison between Shor and Grover.")
    c.showPage()
    c.save()

def create_sample_csv(filepath: str):
    df = pd.DataFrame({
        "Algorithm": ["Shor", "Grover", "HHL", "VQE"],
        "Speedup": ["Exponential", "Quadratic", "Exponential", "Variational"],
        "Qubits_Required": [2048, 512, 1024, 128],
        "Error_Tolerance": [0.0001, 0.001, 0.0005, 0.01]
    })
    df.to_csv(filepath, index=False)

class TestNexusRAGSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n==================================================")
        print("  STARTING NEXUSRAG MULTIMODAL AGENTIC VERIFICATION")
        print("==================================================\n")
        os.makedirs("uploads", exist_ok=True)
        cls.chart_path = os.path.join("uploads", "test_chart.png")
        cls.pdf_path = os.path.join("uploads", "test_quantum.pdf")
        cls.csv_path = os.path.join("uploads", "test_benchmarks.csv")

        create_sample_chart(cls.chart_path)
        create_sample_pdf(cls.pdf_path, cls.chart_path)
        create_sample_csv(cls.csv_path)

    def test_01_backend_modules_and_models(self):
        print("[TEST 01] Validating backend models and config...")
        from backend.config import BASE_DIR, GOOGLE_API_KEY
        self.assertIsNotNone(GOOGLE_API_KEY, "GOOGLE_API_KEY must be set in .env")
        print("  PASSED: Core configuration and models verified.")

    def test_02_multimodal_ingestion_pipeline(self):
        print("[TEST 02] Testing Multi-Format Ingestors (PDF, Image, CSV)...")
        from backend.agent.engine import AgenticRAGEngine
        engine = AgenticRAGEngine()

        # Ingest PDF
        pdf_res = engine.ingest_file(self.pdf_path, os.path.basename(self.pdf_path))
        self.assertGreaterEqual(pdf_res.total_chunks, 1)
        self.assertGreaterEqual(pdf_res.figures_extracted, 1)
        print(f"  PASSED: PDF ingested ({pdf_res.total_chunks} chunks, {pdf_res.figures_extracted} figure).")

        # Ingest CSV
        csv_res = engine.ingest_file(self.csv_path, os.path.basename(self.csv_path))
        self.assertGreaterEqual(csv_res.tables_extracted, 1)
        print(f"  PASSED: CSV ingested ({csv_res.tables_extracted} table chunk).")

    def test_03_hybrid_retrieval_and_reranking(self):
        print("[TEST 03] Testing Hybrid Retrieval (Dense + BM25 + Cross-Encoder)...")
        from backend.agent.engine import AgenticRAGEngine
        engine = AgenticRAGEngine()

        results = engine.retriever.retrieve("What speedup does Shor algorithm achieve?", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        print(f"  PASSED: Retrieved {len(results)} chunks with hybrid fusion & reranking.")

    def test_04_agent_router_and_synthesis(self):
        print("[TEST 04] Testing Agentic Router & Multimodal Synthesis...")
        from backend.agent.engine import AgenticRAGEngine
        from backend.models import QueryRequest
        engine = AgenticRAGEngine()

        # Query testing
        req = QueryRequest(
            question="What is the speedup and qubit requirement for Shor according to the document and table?",
            session_id=f"test_session_{uuid.uuid4().hex[:6]}",
            top_k=4,
            enable_agent=True
        )
        res = engine.query(req)
        self.assertIn("answer", res.model_dump())
        self.assertGreaterEqual(len(res.citations), 1)
        print(f"  Route: {res.route_taken}")
        print(f"  Answer Preview: {res.answer[:160]}...")
        print(f"  Citations: {len(res.citations)}")
        print("  PASSED: Agentic query synthesized with citations.")

    def test_05_conversation_memory_sliding_window(self):
        print("[TEST 05] Testing Multi-turn Conversation Memory & SQLite...")
        from backend.agent.memory import ConversationMemory
        mem = ConversationMemory()
        test_sess_id = f"test_session_{uuid.uuid4().hex[:8]}"
        mem.add_message(test_sess_id, "user", "What is quantum computing?")
        mem.add_message(test_sess_id, "assistant", "Quantum computing uses superposition.")
        
        ctx = mem.get_conversation_context(test_sess_id)
        self.assertEqual(len(ctx["recent_messages"]), 2)
        print("  PASSED: Conversation memory preserved turns.")

    def test_06_rag_evaluation_framework(self):
        print("[TEST 06] Testing RAG Evaluation Framework (Faithfulness, Relevancy, Precision, Recall)...")
        from backend.agent.engine import AgenticRAGEngine
        from backend.evaluation.evaluator import RAGEvaluator
        engine = AgenticRAGEngine()
        evaluator = RAGEvaluator(engine)

        summary = evaluator.run_evaluation(test_questions=["What is Shor algorithm?"])
        self.assertGreaterEqual(summary.total_evaluated, 1)
        self.assertGreaterEqual(summary.overall_rag_score, 0.0)
        print(f"  Overall RAG Score: {summary.overall_rag_score}")
        print(f"  Faithfulness: {summary.average_faithfulness}")
        print(f"  Relevancy: {summary.average_relevancy}")
        print(f"  Precision: {summary.average_context_precision}")
        print(f"  Recall: {summary.average_context_recall}")
        print("  PASSED: Evaluation computed.")

    def test_07_fastapi_endpoints(self):
        print("[TEST 07] Testing FastAPI Endpoints...")
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

        health = client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "healthy")

        docs = client.get("/api/documents")
        self.assertEqual(docs.status_code, 200)

        gal = client.get("/api/gallery")
        self.assertEqual(gal.status_code, 200)

        eval_res = client.get("/api/evaluate/results")
        self.assertEqual(eval_res.status_code, 200)
        print("  PASSED: All FastAPI endpoints responsive.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
