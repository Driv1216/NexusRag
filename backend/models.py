from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

# Ingestion models
class ChunkMetadata(BaseModel):
    chunk_id: str
    parent_chunk_id: Optional[str] = None
    source_file: str
    modality: Literal["text", "figure", "table", "audio", "video"] = "text"
    page_number: Optional[int] = 1
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    table_html: Optional[str] = None
    caption: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None

class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: ChunkMetadata

class IngestResult(BaseModel):
    filename: str
    file_type: str
    total_pages_or_duration: float
    total_chunks: int
    figures_extracted: int
    tables_extracted: int
    summary: str
    suggested_questions: List[str]

# Retrieval & Citation models
class Citation(BaseModel):
    id: int
    chunk_id: str
    type: Literal["text", "figure", "table", "audio", "video"]
    file: str
    page: Optional[int] = None
    timestamp: Optional[str] = None
    snippet: str
    image_url: Optional[str] = None
    score: Optional[float] = None

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default_session"
    top_k: int = 5
    query_image_b64: Optional[str] = None
    enable_agent: bool = True
    enable_rerank: bool = True
    filter_modality: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    route_taken: Optional[str] = None
    confidence_score: Optional[float] = 0.95
    sub_queries: Optional[List[str]] = None

# Evaluation models
class EvaluationRequest(BaseModel):
    test_questions: Optional[List[str]] = None
    top_k: int = 5

class QuestionEvalResult(BaseModel):
    question: str
    ground_truth: Optional[str] = None
    retrieved_contexts: List[str]
    answer: str
    faithfulness_score: float
    relevancy_score: float
    context_precision: float
    context_recall: float
    critique: str

class EvaluationSummary(BaseModel):
    total_evaluated: int
    average_faithfulness: float
    average_relevancy: float
    average_context_precision: float
    average_context_recall: float
    overall_rag_score: float
    results: List[QuestionEvalResult]

# Session models
class ChatMessage(BaseModel):
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    citations: Optional[List[Citation]] = None
    image_b64: Optional[str] = None
    timestamp: Optional[str] = None

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    title: str
    message_count: int
