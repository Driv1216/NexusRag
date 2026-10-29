import os
import shutil
from typing import List, Dict, Any, Tuple
from PIL import Image
from backend.ingestion.base import BaseIngestor
from backend.models import DocumentChunk, ChunkMetadata
from backend.vision.captioner import VisionCaptioner
from backend.config import IMAGES_DIR

class ImageIngestor(BaseIngestor):
    """Ingests standalone image files (.png, .jpg, .webp) with Gemini Vision."""

    def __init__(self):
        self.captioner = VisionCaptioner()

    def ingest(self, file_path: str, filename: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        clean_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lstrip(".")
        target_path = os.path.join(IMAGES_DIR, f"{clean_name}.{ext}")
        if os.path.abspath(file_path) != os.path.abspath(target_path):
            shutil.copyfile(file_path, target_path)

        with Image.open(target_path) as img:
            width, height = img.size

        caption = self.captioner.caption_image(
            target_path,
            context_hint=f"Standalone Image Asset: {filename}"
        )

        chunk_id = f"img_{clean_name}"
        meta = ChunkMetadata(
            chunk_id=chunk_id,
            source_file=filename,
            modality="figure",
            page_number=1,
            image_url=f"/uploads/extracted_images/{clean_name}.{ext}",
            image_path=target_path,
            caption=caption,
            extra_metadata={"width": width, "height": height, "is_standalone": True}
        )

        content = f"[IMAGE ASSET: {filename}]\nDetailed Visual Analysis & OCR:\n{caption}"
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=meta
        )

        summary_meta = {
            "total_pages": 1,
            "figures_extracted": 1,
            "tables_extracted": 0,
            "total_chunks": 1
        }
        return [chunk], summary_meta
