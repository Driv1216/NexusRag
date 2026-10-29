import os
import shutil
import json
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from backend.config import UPLOADS_DIR, IMAGES_DIR, AUDIO_DIR, VIDEO_DIR, BASE_DIR
from backend.models import QueryRequest, QueryResponse, EvaluationRequest
from backend.agent.engine import AgenticRAGEngine
from backend.evaluation.evaluator import RAGEvaluator

app = FastAPI(
    title="NexusRAG — Multimodal Agentic Document Intelligence API",
    version="3.0.0",
    description="Enterprise-grade Multimodal Agentic RAG Platform with Hybrid Retrieval, SSE Streaming, and Evaluation."
)

# Enable CORS for React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize master engines
engine = AgenticRAGEngine()
evaluator = RAGEvaluator(engine)


@app.get("/api/health")
def health_check():
    """System health check and engine stats."""
    docs = engine.list_documents()
    gallery = engine.get_gallery()
    return {
        "status": "healthy",
        "platform": "NexusRAG Multimodal Agentic Engine",
        "active_model": getattr(engine, "active_model", "gemini"),
        "total_documents": len(docs),
        "total_visual_figures": len(gallery),
        "vector_chunks": engine.vector_store.collection.count(),
        "capabilities": [
            "Multimodal Ingestion (PDF, Office, Spreadsheets, Audio, Video)",
            "Hybrid Retrieval (Dense + BM25 + Cross-Encoder Reranker)",
            "Agentic Query Routing & HyDE Expansion",
            "Multi-turn Sliding Memory with SQLite",
            "Real-time SSE Token Streaming",
            "Automated RAG Evaluation (Faithfulness, Relevancy, Precision, Recall)"
        ]
    }


@app.get("/api/documents")
def get_documents():
    """List all indexed documents across all modalities."""
    return {"documents": engine.list_documents()}


@app.get("/api/gallery")
def get_gallery():
    """Retrieve all extracted visual figures, charts, and diagrams."""
    return {"gallery": engine.get_gallery()}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest documents across any supported format:
    PDF, PNG/JPG/WEBP, DOCX/PPTX, CSV/Excel, MP3/WAV/M4A, MP4/WEBM.
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    allowed_exts = [
        ".pdf", ".png", ".jpg", ".jpeg", ".webp",
        ".docx", ".pptx", ".ppt", ".csv", ".xlsx", ".xls",
        ".mp3", ".wav", ".m4a", ".ogg", ".mp4", ".webm", ".mov"
    ]

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(allowed_exts)}"
        )

    file_path = os.path.join(UPLOADS_DIR, filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        ingest_result = engine.ingest_file(file_path, filename)
        return {
            "message": f"Successfully ingested {filename}",
            "document": ingest_result.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.post("/api/query")
def query_rag(request: QueryRequest):
    """Synchronous multimodal query endpoint."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        response = engine.query(request)
        return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/api/query/stream")
async def stream_query_rag(request: QueryRequest):
    """SSE streaming endpoint for token-by-token generation with route decisions & citations."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def event_generator():
        try:
            async for chunk in engine.stream_query(request):
                yield {
                    "event": "message",
                    "data": json.dumps(chunk)
                }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())


@app.get("/api/sessions")
def list_chat_sessions():
    """List conversation sessions from SQLite database."""
    return {"sessions": engine.memory.list_sessions()}


@app.get("/api/sessions/{session_id}")
def get_session_history(session_id: str):
    """Retrieve full conversation context and recent turns for a session."""
    return engine.memory.get_conversation_context(session_id)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """Clear conversation history for a session."""
    engine.memory.clear_session(session_id)
    return {"message": f"Session {session_id} deleted."}


@app.post("/api/evaluate")
def run_evaluation(request: EvaluationRequest):
    """Run automated RAG evaluation benchmark using LLM judge."""
    summary = evaluator.run_evaluation(
        test_questions=request.test_questions,
        top_k=request.top_k
    )
    return summary.model_dump()


@app.get("/api/evaluate/results")
def get_evaluation_results():
    """Fetch the latest benchmark evaluation results."""
    res = evaluator.get_latest_results()
    if not res:
        # Run default evaluation on demand if none exists
        summary = evaluator.run_evaluation()
        return summary.model_dump()
    return res


@app.delete("/api/clear")
def clear_all():
    """Clear all vector databases, uploaded assets, and extracted media."""
    engine.clear_all()
    return {"message": "NexusRAG workspace successfully wiped and reset."}


# Mount uploads directory for media delivery
if os.path.exists(UPLOADS_DIR):
    app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# Mount React frontend static assets if built
frontend_dist = os.path.join(BASE_DIR, "frontend", "dist")
frontend_static = os.path.join(BASE_DIR, "frontend")

if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/")
    def serve_react_app():
        return FileResponse(os.path.join(frontend_dist, "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa_routes(full_path: str):
        target = os.path.join(frontend_dist, full_path)
        if os.path.exists(target) and not os.path.isdir(target):
            return FileResponse(target)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    # Serve static vanilla/dev preview if dist not yet built
    if os.path.exists(frontend_static):
        app.mount("/static", StaticFiles(directory=frontend_static), name="static")

        @app.get("/")
        def serve_index():
            index_path = os.path.join(frontend_static, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return JSONResponse({"message": "NexusRAG backend running."})
