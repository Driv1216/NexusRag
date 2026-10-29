from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from backend.models import DocumentChunk, IngestResult

class BaseIngestor(ABC):
    """Abstract Base Class for multi-format document ingestors in NexusRAG."""

    @abstractmethod
    def ingest(self, file_path: str, filename: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Parse file into structured DocumentChunks (with parent-child linkages and metadata)
        and return (chunks, metadata_summary).
        """
        pass
