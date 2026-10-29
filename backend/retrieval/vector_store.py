import os
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from backend.config import CHROMA_DB_DIR
from backend.models import DocumentChunk, ChunkMetadata
from backend.embeddings.multimodal_embedder import MultimodalEmbedder

class ChromaVectorStore:
    """ChromaDB persistent vector store for multimodal chunks with metadata filtering."""

    def __init__(self, collection_name: str = "nexus_multimodal"):
        self.collection_name = collection_name
        self.embedder = MultimodalEmbedder()
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Embed and insert chunks into ChromaDB."""
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        texts_to_embed = [c.content for c in chunks]
        computed_embeddings = self.embedder.embed_texts(texts_to_embed)

        for chunk, emb in zip(chunks, computed_embeddings):
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            
            # ChromaDB metadata requires str, int, float, bool
            meta_dict = {
                "chunk_id": chunk.metadata.chunk_id,
                "parent_chunk_id": chunk.metadata.parent_chunk_id or "",
                "source_file": chunk.metadata.source_file,
                "modality": chunk.metadata.modality,
                "page_number": chunk.metadata.page_number or 1,
                "image_url": chunk.metadata.image_url or "",
                "image_path": chunk.metadata.image_path or "",
                "caption": (chunk.metadata.caption or "")[:500],
                "table_html": (chunk.metadata.table_html or "")[:1000],
                "extra_json": json.dumps(chunk.metadata.extra_metadata or {})
            }
            metadatas.append(meta_dict)
            embeddings.append(emb)

        # Batch upsert
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"Added {len(chunks)} chunks to ChromaDB collection '{self.collection_name}'. Total: {self.collection.count()}")

    def similarity_search(
        self,
        query_text: str,
        k: int = 5,
        modality_filter: Optional[str] = None,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB by query vector with optional metadata filtering."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.embed_text(query_text)
        
        where_filter = {}
        if modality_filter:
            where_filter["modality"] = modality_filter
        if source_filter:
            where_filter["source_file"] = source_filter

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.collection.query(**query_kwargs)

        matched_items = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else []
            distances = results["distances"][0] if "distances" in results else []

            for i in range(len(docs)):
                meta = metas[i] if i < len(metas) else {}
                dist = distances[i] if i < len(distances) else 0.0
                score = round(1.0 - (dist / 2.0), 4) if dist is not None else 0.5
                
                matched_items.append({
                    "chunk_id": meta.get("chunk_id", f"c_{i}"),
                    "content": docs[i],
                    "metadata": meta,
                    "score": score,
                    "retriever": "dense_chroma"
                })

        return matched_items

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all documents in the collection."""
        count = self.collection.count()
        if count == 0:
            return []
        data = self.collection.get(include=["documents", "metadatas"])
        items = []
        for i in range(len(data["ids"])):
            items.append({
                "chunk_id": data["ids"][i],
                "content": data["documents"][i],
                "metadata": data["metadatas"][i]
            })
        return items

    def clear(self):
        """Reset and wipe the vector store."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("ChromaDB vector store cleared.")
