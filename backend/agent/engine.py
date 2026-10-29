import os
import glob
import shutil
from typing import Dict, Any, List, Optional, AsyncGenerator
from backend.models import QueryRequest, QueryResponse, IngestResult, Citation
from backend.config import UPLOADS_DIR, IMAGES_DIR, AUDIO_DIR, VIDEO_DIR, GOOGLE_API_KEY, LLM_MODELS_FALLBACK
from backend.retrieval.vector_store import ChromaVectorStore
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.agent.router import QueryRouter
from backend.agent.query_rewriter import QueryRewriter
from backend.agent.memory import ConversationMemory
from backend.agent.synthesizer import MultimodalSynthesizer
from backend.ingestion import (
    PDFIngestor,
    ImageIngestor,
    DocxPptxIngestor,
    SpreadsheetIngestor,
    AudioIngestor,
    VideoIngestor
)
from langchain_google_genai import ChatGoogleGenerativeAI

class AgenticRAGEngine:
    """Master Orchestrator for NexusRAG Multimodal Agentic System."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgenticRAGEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print("Initializing NexusRAG Agentic Engine...")
        self.vector_store = ChromaVectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.router = QueryRouter()
        self.rewriter = QueryRewriter()
        self.memory = ConversationMemory()
        self.synthesizer = MultimodalSynthesizer()

        # Initialize ingestors
        self.pdf_ingestor = PDFIngestor()
        self.image_ingestor = ImageIngestor()
        self.docx_ingestor = DocxPptxIngestor()
        self.sheet_ingestor = SpreadsheetIngestor()
        self.audio_ingestor = AudioIngestor()
        self.video_ingestor = VideoIngestor()

        # LLM for summaries
        self.summary_llm = None
        for model in LLM_MODELS_FALLBACK:
            try:
                self.summary_llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                    max_output_tokens=800
                )
                self.active_model = model
                break
            except Exception:
                pass

        self._initialized = True
        print("NexusRAG Agentic Engine initialized successfully!")

    def ingest_file(self, file_path: str, filename: str) -> IngestResult:
        """Route file to appropriate ingestor, index chunks, and generate document summary."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            chunks, meta = self.pdf_ingestor.ingest(file_path, filename)
            file_type = "PDF Document"
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            chunks, meta = self.image_ingestor.ingest(file_path, filename)
            file_type = "Image Asset"
        elif ext in [".docx", ".pptx", ".ppt"]:
            chunks, meta = self.docx_ingestor.ingest(file_path, filename)
            file_type = "Office Document"
        elif ext in [".csv", ".xlsx", ".xls"]:
            chunks, meta = self.sheet_ingestor.ingest(file_path, filename)
            file_type = "Spreadsheet Table"
        elif ext in [".mp3", ".wav", ".m4a", ".ogg", ".flac"]:
            chunks, meta = self.audio_ingestor.ingest(file_path, filename)
            file_type = "Audio Track"
        elif ext in [".mp4", ".webm", ".mov", ".mkv"]:
            chunks, meta = self.video_ingestor.ingest(file_path, filename)
            file_type = "Video Recording"
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Store in ChromaDB vector store
        if chunks:
            self.vector_store.add_chunks(chunks)
            self.retriever._refresh_bm25()

        # Generate summary & suggested questions
        sample_text = "\n\n".join([c.content for c in chunks[:5]])[:3500]
        summary_info = self._generate_doc_summary(sample_text, filename)

        return IngestResult(
            filename=filename,
            file_type=file_type,
            total_pages_or_duration=float(meta.get("total_pages", 1)),
            total_chunks=len(chunks),
            figures_extracted=meta.get("figures_extracted", 0),
            tables_extracted=meta.get("tables_extracted", 0),
            summary=summary_info.get("summary", f"{filename} uploaded and indexed."),
            suggested_questions=summary_info.get("suggested_questions", [])
        )

    def _generate_doc_summary(self, sample_text: str, filename: str) -> Dict[str, Any]:
        """Generate concise summary and sample inquiries."""
        if not self.summary_llm or not sample_text.strip():
            return {
                "summary": f"Document '{filename}' successfully indexed with multimodal capabilities.",
                "suggested_questions": [f"What are the main topics in {filename}?", f"Summarize key data points in {filename}."]
            }

        prompt = f"""
Analyze this document excerpt from "{filename}":
{sample_text}

Provide:
SUMMARY: A 2-3 sentence overview of content, methods, or findings.
QUESTIONS:
1. First insightful question
2. Second insightful question
3. Third insightful question
4. Fourth insightful question
"""
        try:
            res = self.summary_llm.invoke(prompt)
            raw = res.content if hasattr(res, 'content') else str(res)
            if isinstance(raw, list):
                raw = "".join([x.get("text", str(x)) for x in raw])

            summary = ""
            questions = []
            if "SUMMARY:" in raw:
                parts = raw.split("QUESTIONS:")
                summary = parts[0].replace("SUMMARY:", "").strip()
                if len(parts) > 1:
                    questions = [q.strip().lstrip("0123456789.- ") for q in parts[1].strip().split("\n") if q.strip()]
            else:
                summary = str(raw)[:300]
                questions = [f"What are the key points in {filename}?"]

            return {"summary": summary, "suggested_questions": questions[:4]}
        except Exception as e:
            return {
                "summary": f"Document '{filename}' indexed.",
                "suggested_questions": [f"What is discussed in {filename}?"]
            }

    def query(self, req: QueryRequest) -> QueryResponse:
        """Full synchronous execution pipeline with agent routing & hybrid retrieval."""
        has_img = bool(req.query_image_b64)
        route_decision = self.router.route_query(req.question, has_image=has_img) if req.enable_agent else {"route": "RETRIEVAL", "sub_queries": [req.question]}
        route = route_decision.get("route", "RETRIEVAL")
        sub_queries = route_decision.get("sub_queries", [req.question])

        # Fetch conversation memory
        conv_context = self.memory.get_conversation_context(req.session_id) if req.session_id else None

        if route == "DIRECT":
            # Direct response without retrieval
            answer_text = "Hello! I am NexusRAG, your multimodal research assistant. You can upload PDFs, research papers, diagrams, spreadsheets, audio recordings, or videos and ask deep questions with verified citations."
            if req.session_id:
                self.memory.add_message(req.session_id, "user", req.question)
                self.memory.add_message(req.session_id, "assistant", answer_text)
            return QueryResponse(
                answer=answer_text,
                citations=[],
                route_taken="DIRECT",
                confidence_score=1.0,
                sub_queries=[]
            )

        # Multi-Step or standard retrieval
        all_retrieved = []
        seen_ids = set()

        for sq in sub_queries:
            # HyDE expansion
            search_query = self.rewriter.generate_hyde_passage(sq) if len(sq.split()) > 3 else sq
            retrieved = self.retriever.retrieve(
                query=search_query,
                top_k=req.top_k,
                modality_filter=req.filter_modality,
                enable_rerank=req.enable_rerank
            )
            for item in retrieved:
                c_id = item.get("chunk_id")
                if c_id not in seen_ids:
                    seen_ids.add(c_id)
                    all_retrieved.append(item)

        # Multimodal synthesis
        res = self.synthesizer.synthesize(
            question=req.question,
            retrieved_chunks=all_retrieved[:req.top_k],
            conversation_context=conv_context,
            query_image_b64=req.query_image_b64
        )

        # Save to memory
        if req.session_id:
            self.memory.add_message(req.session_id, "user", req.question)
            self.memory.add_message(req.session_id, "assistant", res["answer"])

        return QueryResponse(
            answer=res["answer"],
            citations=[Citation(**c) for c in res["citations"]],
            route_taken=route,
            confidence_score=0.96,
            sub_queries=sub_queries
        )

    async def stream_query(self, req: QueryRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """SSE streaming generator executing agent routing and token streaming."""
        has_img = bool(req.query_image_b64)
        route_decision = self.router.route_query(req.question, has_image=has_img) if req.enable_agent else {"route": "RETRIEVAL", "sub_queries": [req.question]}
        route = route_decision.get("route", "RETRIEVAL")
        sub_queries = route_decision.get("sub_queries", [req.question])

        yield {
            "type": "route",
            "route": route,
            "sub_queries": sub_queries
        }

        conv_context = self.memory.get_conversation_context(req.session_id) if req.session_id else None

        if route == "DIRECT":
            direct_ans = "Hello! I am NexusRAG, your multimodal research assistant. You can upload PDFs, research papers, diagrams, spreadsheets, audio recordings, or videos and ask deep questions with verified citations."
            for word in direct_ans.split(" "):
                yield {"type": "token", "content": word + " "}
            if req.session_id:
                self.memory.add_message(req.session_id, "user", req.question)
                self.memory.add_message(req.session_id, "assistant", direct_ans)
            yield {"type": "done", "citations": []}
            return

        # Retrieve
        all_retrieved = []
        seen_ids = set()
        for sq in sub_queries:
            search_query = self.rewriter.generate_hyde_passage(sq) if len(sq.split()) > 3 else sq
            retrieved = self.retriever.retrieve(
                query=search_query,
                top_k=req.top_k,
                modality_filter=req.filter_modality,
                enable_rerank=req.enable_rerank
            )
            for item in retrieved:
                c_id = item.get("chunk_id")
                if c_id not in seen_ids:
                    seen_ids.add(c_id)
                    all_retrieved.append(item)

        # Stream synthesis
        full_answer_parts = []
        async for chunk in self.synthesizer.stream_synthesize(
            question=req.question,
            retrieved_chunks=all_retrieved[:req.top_k],
            conversation_context=conv_context,
            query_image_b64=req.query_image_b64
        ):
            if chunk.get("type") == "token":
                full_answer_parts.append(chunk.get("content", ""))
            yield chunk

        # Save to memory on completion
        full_answer = "".join(full_answer_parts)
        if req.session_id and full_answer:
            self.memory.add_message(req.session_id, "user", req.question)
            self.memory.add_message(req.session_id, "assistant", full_answer)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents across formats."""
        pattern = os.path.join(UPLOADS_DIR, "*.*")
        all_files = glob.glob(pattern)
        docs = []

        for p in all_files:
            fname = os.path.basename(p)
            ext = os.path.splitext(fname)[1].lower()
            size_mb = round(os.path.getsize(p) / (1024 * 1024), 2)

            is_pdf = ext == ".pdf"
            is_image = ext in [".png", ".jpg", ".jpeg", ".webp"]
            is_sheet = ext in [".csv", ".xlsx", ".xls"]
            is_audio = ext in [".mp3", ".wav", ".m4a", ".ogg"]
            is_video = ext in [".mp4", ".webm", ".mov"]

            clean_name = os.path.splitext(fname)[0]
            fig_count = len(glob.glob(os.path.join(IMAGES_DIR, f"{clean_name}_*.*")))

            docs.append({
                "filename": fname,
                "size_mb": size_mb,
                "path": p,
                "is_pdf": is_pdf,
                "is_image": is_image,
                "is_sheet": is_sheet,
                "is_audio": is_audio,
                "is_video": is_video,
                "figures_count": fig_count if is_pdf else (1 if is_image else 0)
            })

        return docs

    def get_gallery(self) -> List[Dict[str, Any]]:
        """Fetch visual gallery images and diagrams."""
        image_files = glob.glob(os.path.join(IMAGES_DIR, "*.*"))
        gallery = []
        for img_path in sorted(image_files, key=os.path.getmtime, reverse=True):
            fname = os.path.basename(img_path)
            size_kb = round(os.path.getsize(img_path) / 1024, 1)
            rel_url = f"/uploads/extracted_images/{fname}"
            gallery.append({
                "filename": fname,
                "url": rel_url,
                "size_kb": size_kb
            })
        return gallery

    def clear_all(self):
        """Wipe vector DB, memory, uploaded assets, and extracted media."""
        self.vector_store.clear()
        self.retriever._refresh_bm25()
        for d in [UPLOADS_DIR, IMAGES_DIR, AUDIO_DIR, VIDEO_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)
        # Clear sqlite db
        db_path = os.path.join(DATA_DIR, "chat_history.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        self.memory._init_db()
        print("NexusRAG completely reset.")
