import os
from typing import List, Union
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from backend.config import TEXT_EMBEDDING_MODEL

class MultimodalEmbedder:
    """Multimodal Embedder supporting dense text embeddings and image representations."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MultimodalEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        print(f"Initializing embedding model: {TEXT_EMBEDDING_MODEL}...")
        self.model = None
        self.embedding_dim = 384

        # 1. Try local cache first to avoid network delays / offline errors
        try:
            self.model = SentenceTransformer(TEXT_EMBEDDING_MODEL, local_files_only=True)
            self.embedding_dim = getattr(self.model, "get_embedding_dimension", getattr(self.model, "get_sentence_embedding_dimension", lambda: 384))()
            print(f"Embedding model loaded from local cache! Dim: {self.embedding_dim}")
        except Exception:
            # 2. Try online download if local cache not found
            try:
                self.model = SentenceTransformer(TEXT_EMBEDDING_MODEL, local_files_only=False)
                self.embedding_dim = getattr(self.model, "get_embedding_dimension", getattr(self.model, "get_sentence_embedding_dimension", lambda: 384))()
                print(f"Embedding model downloaded successfully! Dim: {self.embedding_dim}")
            except Exception as e:
                print(f"Offline mode: Using deterministic normalized vectors for offline execution ({e}).")
                self.model = None
                self.embedding_dim = 384
        
        self.clip_model = None
        if os.getenv("ENABLE_CLIP_DOWNLOAD", "false").lower() == "true":
            self._try_load_clip()
        self._initialized = True

    def _try_load_clip(self):
        """Optionally load CLIP model for unified visual-text embedding space."""
        try:
            from sentence_transformers import SentenceTransformer
            self.clip_model = SentenceTransformer("clip-ViT-B-32")
            print("CLIP multimodal model (clip-ViT-B-32) loaded successfully.")
        except Exception as e:
            print(f"Note: CLIP model not loaded: {e}. Using dense visual caption vectors.")
            self.clip_model = None

    def _deterministic_hash_vector(self, text: str) -> List[float]:
        """Generate deterministic normalized unit vector without model weights (offline fallback)."""
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Seed pseudo-random generator with hash bytes to produce 384 floats
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.embedding_dim)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text string."""
        if not text or not text.strip():
            return [0.0] * self.embedding_dim
        if self.model is None:
            return self._deterministic_hash_vector(text.strip())
        vec = self.model.encode(text.strip(), convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized embedding vectors for a batch of text strings."""
        if not texts:
            return []
        cleaned = [t.strip() if t and t.strip() else "empty document" for t in texts]
        if self.model is None:
            return [self._deterministic_hash_vector(t) for t in cleaned]
        vecs = self.model.encode(cleaned, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.tolist()

    def embed_image(self, image_path: str, caption_fallback: str = "") -> List[float]:
        """Embed an image file directly via CLIP or via caption embedding."""
        if self.clip_model and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                vec = self.clip_model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
                return vec.tolist()
            except Exception as e:
                print(f"CLIP image embedding error: {e}. Using caption fallback.")

        fallback_text = caption_fallback if caption_fallback else os.path.basename(image_path)
        return self.embed_text(fallback_text)
