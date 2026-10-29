import os
import glob
from typing import List, Dict, Any, Tuple
import pymupdf as fitz
import pdfplumber
from backend.ingestion.base import BaseIngestor
from backend.ingestion.chunking import HierarchicalChunker
from backend.models import DocumentChunk, ChunkMetadata
from backend.vision.captioner import VisionCaptioner
from backend.config import IMAGES_DIR

class PDFIngestor(BaseIngestor):
    """
    Advanced PDF Ingestor:
    - Extracts multi-column text
    - Identifies structured tables with pdfplumber
    - Extracts high-resolution figures, charts, and vector diagrams
    - Generates Gemini Vision descriptions and OCR
    - Links visual figures to surrounding page context
    """

    def __init__(self):
        self.chunker = HierarchicalChunker()
        self.captioner = VisionCaptioner()

    def ingest(self, file_path: str, filename: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        chunks: List[DocumentChunk] = []
        doc_clean_name = os.path.splitext(filename)[0]
        figures_extracted = 0
        tables_extracted = 0
        total_pages = 0

        # 1. Process with PyMuPDF for text and high-res visual figures
        doc = fitz.open(file_path)
        total_pages = len(doc)

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # A. Extract Text
            page_text = page.get_text("text").strip()
            if page_text:
                page_chunks = self.chunker.chunk_document_section(
                    text=page_text,
                    source_file=filename,
                    page_number=page_num,
                    modality="text"
                )
                chunks.extend(page_chunks)

            # B. Extract Embedded Raster Images
            image_list = page.get_images(full=True)
            img_count_on_page = 0

            for img_info in image_list:
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                if not base_img:
                    continue

                width = base_img["width"]
                height = base_img["height"]
                # Filter small icons / bullets / decorative lines
                if width < 120 or height < 120 or (width * height < 20000):
                    continue

                img_count_on_page += 1
                figures_extracted += 1
                image_bytes = base_img["image"]
                image_ext = base_img["ext"]
                img_filename = f"{doc_clean_name}_p{page_num}_fig{img_count_on_page}.{image_ext}"
                save_path = os.path.join(IMAGES_DIR, img_filename)
                rel_url = f"/uploads/extracted_images/{img_filename}"

                with open(save_path, "wb") as f:
                    f.write(image_bytes)

                # Generate Vision Caption & OCR
                caption = self.captioner.caption_image(
                    save_path,
                    context_hint=f"Document: {filename}, Page {page_num}"
                )

                fig_chunk_id = f"fig_{doc_clean_name}_p{page_num}_{img_count_on_page}"
                fig_meta = ChunkMetadata(
                    chunk_id=fig_chunk_id,
                    source_file=filename,
                    modality="figure",
                    page_number=page_num,
                    image_url=rel_url,
                    image_path=save_path,
                    caption=caption,
                    extra_metadata={"width": width, "height": height}
                )

                content_repr = f"[FIGURE / CHART / DIAGRAM: {img_filename}]\nSource: {filename} (Page {page_num})\nVisual Content & OCR Description:\n{caption}"
                chunks.append(DocumentChunk(
                    chunk_id=fig_chunk_id,
                    content=content_repr,
                    metadata=fig_meta
                ))

            # C. Check for Vector Charts / Diagrams (drawings without raster images)
            drawings = page.get_drawings()
            if len(drawings) > 15 and img_count_on_page == 0:
                pix = page.get_pixmap(dpi=150)
                img_filename = f"{doc_clean_name}_p{page_num}_diagram.png"
                save_path = os.path.join(IMAGES_DIR, img_filename)
                rel_url = f"/uploads/extracted_images/{img_filename}"
                pix.save(save_path)
                figures_extracted += 1

                caption = self.captioner.caption_image(
                    save_path,
                    context_hint=f"Document: {filename}, Page {page_num} (Vector Chart / Architecture Diagram)"
                )

                diag_chunk_id = f"diag_{doc_clean_name}_p{page_num}"
                diag_meta = ChunkMetadata(
                    chunk_id=diag_chunk_id,
                    source_file=filename,
                    modality="figure",
                    page_number=page_num,
                    image_url=rel_url,
                    image_path=save_path,
                    caption=caption,
                    extra_metadata={"width": pix.width, "height": pix.height, "is_vector_diagram": True}
                )

                chunks.append(DocumentChunk(
                    chunk_id=diag_chunk_id,
                    content=f"[VECTOR DIAGRAM / CHART: {img_filename}]\nSource: {filename} (Page {page_num})\nVisual Analysis:\n{caption}",
                    metadata=diag_meta
                ))

        doc.close()

        # 2. Extract Structured Tables with pdfplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_idx, plumber_page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    tables = plumber_page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue
                        
                        tables_extracted += 1
                        # Format as clean markdown table
                        headers = [str(col).replace("\n", " ").strip() if col else f"Col {c_i+1}" for c_i, col in enumerate(table[0])]
                        md_rows = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
                        for row in table[1:]:
                            cleaned_row = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
                            # match length
                            if len(cleaned_row) < len(headers):
                                cleaned_row += [""] * (len(headers) - len(cleaned_row))
                            md_rows.append(f"| {' | '.join(cleaned_row[:len(headers)])} |")
                        
                        table_md = "\n".join(md_rows)
                        tbl_chunk_id = f"table_{doc_clean_name}_p{page_num}_{t_idx+1}"
                        
                        tbl_meta = ChunkMetadata(
                            chunk_id=tbl_chunk_id,
                            source_file=filename,
                            modality="table",
                            page_number=page_num,
                            table_html=table_md,
                            extra_metadata={"rows": len(table), "cols": len(headers)}
                        )
                        
                        chunks.append(DocumentChunk(
                            chunk_id=tbl_chunk_id,
                            content=f"[STRUCTURED TABLE: {filename} - Page {page_num} Table {t_idx+1}]\n{table_md}",
                            metadata=tbl_meta
                        ))
        except Exception as e:
            print(f"pdfplumber table extraction note for {filename}: {e}")

        summary_meta = {
            "total_pages": total_pages,
            "figures_extracted": figures_extracted,
            "tables_extracted": tables_extracted,
            "total_chunks": len(chunks)
        }
        return chunks, summary_meta
