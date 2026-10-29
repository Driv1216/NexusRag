import uuid
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.models import DocumentChunk, ChunkMetadata
from backend.config import (
    DEFAULT_CHILD_CHUNK_SIZE,
    DEFAULT_CHILD_CHUNK_OVERLAP,
    DEFAULT_PARENT_CHUNK_SIZE,
    DEFAULT_PARENT_CHUNK_OVERLAP
)

class HierarchicalChunker:
    """
    Implements Hierarchical Parent-Child chunking:
    - Parent chunks: Broad context blocks (1200 chars)
    - Child chunks: Granular retrieval targets (400 chars) linked to parent ID.
    """

    def __init__(
        self,
        child_size: int = DEFAULT_CHILD_CHUNK_SIZE,
        child_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
        parent_size: int = DEFAULT_PARENT_CHUNK_SIZE,
        parent_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_document_section(
        self,
        text: str,
        source_file: str,
        page_number: Optional[int] = 1,
        modality: str = "text",
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Split a text block into linked parent-child chunks."""
        if not text or not text.strip():
            return []

        chunks: List[DocumentChunk] = []
        parent_texts = self.parent_splitter.split_text(text)

        for p_idx, p_text in enumerate(parent_texts):
            parent_id = f"parent_{uuid.uuid4().hex[:8]}_{page_number}_{p_idx}"
            
            # Store parent as a background chunk context
            child_texts = self.child_splitter.split_text(p_text)
            if not child_texts:
                child_texts = [p_text]

            for c_idx, c_text in enumerate(child_texts):
                child_id = f"chunk_{uuid.uuid4().hex[:8]}_p{page_number}_{p_idx}_{c_idx}"
                metadata = ChunkMetadata(
                    chunk_id=child_id,
                    parent_chunk_id=parent_id,
                    source_file=source_file,
                    modality=modality,
                    page_number=page_number,
                    extra_metadata={
                        **(extra_meta or {}),
                        "parent_content": p_text,
                        "parent_id": parent_id
                    }
                )
                chunks.append(DocumentChunk(
                    chunk_id=child_id,
                    content=c_text.strip(),
                    metadata=metadata
                ))

        return chunks
