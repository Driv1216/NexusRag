import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = str(Path(__file__).resolve().parent.parent)
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
IMAGES_DIR = os.path.join(UPLOADS_DIR, "extracted_images")
AUDIO_DIR = os.path.join(UPLOADS_DIR, "audio_tracks")
VIDEO_DIR = os.path.join(UPLOADS_DIR, "video_frames")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
DATA_DIR = os.path.join(BASE_DIR, "data")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation_data")

# Create directories
for d in [UPLOADS_DIR, IMAGES_DIR, AUDIO_DIR, VIDEO_DIR, CHROMA_DB_DIR, DATA_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

# API Keys & LLM settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODELS_FALLBACK = [
    "gemini-3.6-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Embeddings & Reranker models
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking settings
DEFAULT_CHILD_CHUNK_SIZE = 400
DEFAULT_CHILD_CHUNK_OVERLAP = 80
DEFAULT_PARENT_CHUNK_SIZE = 1200
DEFAULT_PARENT_CHUNK_OVERLAP = 150

# Retrieval settings
DEFAULT_TOP_K = 5
HYBRID_DENSE_WEIGHT = 0.6
HYBRID_SPARSE_WEIGHT = 0.4
RRF_K_CONSTANT = 60
