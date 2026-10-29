import os
import glob
import shutil
import base64
import mimetypes
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import pymupdf as fitz
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

VECTOR_DB_DIR = "vector_db"
UPLOADS_DIR = "uploads"
IMAGES_DIR = os.path.join(UPLOADS_DIR, "extracted_images")

os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


class RAGEngine:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        self._init_embeddings()
        self._init_llm()
        self.vector_store = None
        self._load_vector_store()

    def _init_embeddings(self):
        """Initialize HuggingFace embeddings locally."""
        try:
            print("Initializing local HuggingFace Embeddings (all-MiniLM-L6-v2)...")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Error initializing local embeddings: {e}")
            raise

    def _init_llm(self):
        """Initialize Gemini LLM with model fallback."""
        models_to_try = [
            "gemini-3.6-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro"
        ]
        self.llm = None
        for model in models_to_try:
            try:
                print(f"Attempting to initialize LLM with model: {model}")
                self.llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=self.api_key,
                    max_output_tokens=2048
                )
                self.active_model = model
                print(f"Successfully initialized LLM with {model}")
                break
            except Exception as e:
                print(f"Failed to load {model}: {e}")

        if not self.llm:
            raise RuntimeError("Could not initialize any Gemini LLM model with the provided API key.")

    def _load_vector_store(self):
        """Load FAISS vector store from disk if present."""
        index_path = os.path.join(VECTOR_DB_DIR, "index.faiss")
        if os.path.exists(index_path):
            try:
                self.vector_store = FAISS.load_local(
                    VECTOR_DB_DIR,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("Loaded existing FAISS vector store.")
            except Exception as e:
                print(f"Error loading vector store: {e}. Starting fresh.")
                self.vector_store = None
        else:
            self.vector_store = None

    def _save_vector_store(self):
        """Save FAISS vector store to disk."""
        if self.vector_store:
            self.vector_store.save_local(VECTOR_DB_DIR)
            print("Persisted FAISS vector store to disk.")

    def _extract_text(self, response) -> str:
        """Helper to safely extract string text from Gemini response object."""
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif hasattr(item, "text"):
                    text_parts.append(str(item.text))
            return "\n".join(text_parts).strip()
        return str(content).strip()

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Convert local image file to base64 string with MIME type prefix."""
        if not os.path.exists(image_path):
            return None
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        with open(image_path, "rb") as img_file:
            b64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64_data}"

    def extract_images_from_pdf(self, pdf_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract embedded figures, charts, and diagrams from PDF pages using PyMuPDF."""
        extracted_figures = []
        doc = fitz.open(pdf_path)
        doc_clean_name = os.path.splitext(filename)[0]

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            image_list = page.get_images(full=True)

            img_count_on_page = 0
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                if not base_img:
                    continue

                image_bytes = base_img["image"]
                image_ext = base_img["ext"]
                width = base_img["width"]
                height = base_img["height"]

                # Filter out tiny icons, horizontal rule lines, and decorative logos
                if width < 120 or height < 120 or (width * height < 20000):
                    continue

                img_count_on_page += 1
                img_filename = f"{doc_clean_name}_p{page_num}_fig{img_count_on_page}.{image_ext}"
                save_path = os.path.join(IMAGES_DIR, img_filename)
                rel_path = os.path.join("uploads", "extracted_images", img_filename)

                with open(save_path, "wb") as f:
                    f.write(image_bytes)

                extracted_figures.append({
                    "id": f"fig_{page_num}_{img_count_on_page}",
                    "filename": img_filename,
                    "save_path": save_path,
                    "rel_path": rel_path,
                    "page_number": page_num,
                    "source_file": filename,
                    "width": width,
                    "height": height,
                    "type": "figure"
                })

            # Check if page has visual drawings/charts (vector graphics) but no raster images
            # If so, render a high-res crop of the page as a chart figure
            drawings = page.get_drawings()
            if len(drawings) > 15 and img_count_on_page == 0:
                pix = page.get_pixmap(dpi=150)
                img_filename = f"{doc_clean_name}_p{page_num}_diagram.png"
                save_path = os.path.join(IMAGES_DIR, img_filename)
                rel_path = os.path.join("uploads", "extracted_images", img_filename)
                pix.save(save_path)

                extracted_figures.append({
                    "id": f"diag_{page_num}",
                    "filename": img_filename,
                    "save_path": save_path,
                    "rel_path": rel_path,
                    "page_number": page_num,
                    "source_file": filename,
                    "width": pix.width,
                    "height": pix.height,
                    "type": "diagram"
                })

        doc.close()
        print(f"Extracted {len(extracted_figures)} figures/diagrams from {filename}.")
        return extracted_figures

    def caption_image_with_vision(self, image_path: str, context_hint: str = "") -> str:
        """Generate dense descriptive captions and OCR data from an image using Gemini Vision."""
        b64_img = self._encode_image(image_path)
        if not b64_img:
            return "Visual figure excerpt from document."

        prompt = f"""
You are an expert document vision analyst.
Analyze the following figure/chart/diagram extracted from a document.
Context: {context_hint}

Please provide:
1. Visual Type: (e.g., Bar Chart, Architecture Diagram, Flowchart, Table, Illustration)
2. Detailed Description: Summarize all axes, trends, numeric data points, labels, and relationships shown.
3. Visible Text & OCR: Transcribe key text and equations clearly.
"""
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": b64_img}}
            ]
        )

        try:
            response = self.llm.invoke([message])
            return self._extract_text(response)
        except Exception as e:
            print(f"Vision captioning fallback for {image_path}: {e}")
            return f"Figure depicting data/diagram on {context_hint}."

    def list_uploaded_documents(self) -> List[Dict[str, Any]]:
        """List all indexed PDF documents and standalone images in uploads directory."""
        pdf_files = glob.glob(os.path.join(UPLOADS_DIR, "*.pdf"))
        standalone_images = glob.glob(os.path.join(UPLOADS_DIR, "*.png")) + \
                            glob.glob(os.path.join(UPLOADS_DIR, "*.jpg")) + \
                            glob.glob(os.path.join(UPLOADS_DIR, "*.jpeg")) + \
                            glob.glob(os.path.join(UPLOADS_DIR, "*.webp"))
        
        docs_info = []
        for file_path in pdf_files + standalone_images:
            filename = os.path.basename(file_path)
            size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
            is_pdf = filename.lower().endswith(".pdf")
            
            # Count extracted figures for this PDF
            clean_name = os.path.splitext(filename)[0]
            fig_count = len(glob.glob(os.path.join(IMAGES_DIR, f"{clean_name}_*.*")))

            docs_info.append({
                "filename": filename,
                "size_mb": size_mb,
                "path": file_path,
                "is_pdf": is_pdf,
                "figures_count": fig_count if is_pdf else 1
            })
        return docs_info

    def get_visual_gallery(self) -> List[Dict[str, Any]]:
        """Retrieve all extracted and uploaded visual assets for the media browser."""
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

    def ingest_pdf(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Ingest PDF: Extract text chunks + extract and vision-caption visual figures."""
        print(f"Ingesting PDF: {filename}...")
        loader = PyPDFLoader(file_path)
        raw_documents = loader.load()

        # Update metadata for clear citation tracking
        for doc in raw_documents:
            page_num = doc.metadata.get("page", 0) + 1
            doc.metadata["source_file"] = filename
            doc.metadata["page_number"] = page_num
            doc.metadata["type"] = "text"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(raw_documents)
        print(f"Split {filename} into {len(chunks)} text chunks.")

        # Extract figures & diagrams from PDF
        extracted_figures = self.extract_images_from_pdf(file_path, filename)
        image_docs = []

        for fig in extracted_figures:
            caption = self.caption_image_with_vision(
                fig["save_path"],
                context_hint=f"Document: {filename}, Page: {fig['page_number']}"
            )
            fig["caption"] = caption
            img_doc = Document(
                page_content=f"[FIGURE / CHART / DIAGRAM: {fig['filename']}]\nDocument: {filename}\nPage: {fig['page_number']}\nVisual Content Description:\n{caption}",
                metadata={
                    "type": "image",
                    "source_file": filename,
                    "page_number": fig["page_number"],
                    "image_path": fig["save_path"],
                    "image_url": f"/uploads/extracted_images/{fig['filename']}",
                    "caption": caption,
                    "width": fig["width"],
                    "height": fig["height"]
                }
            )
            image_docs.append(img_doc)

        all_docs_to_index = chunks + image_docs

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(all_docs_to_index, self.embeddings)
        else:
            self.vector_store.add_documents(all_docs_to_index)

        self._save_vector_store()

        # Generate summary and suggested questions (NotebookLM style)
        summary_info = self.generate_document_summary(raw_documents[:5], filename)

        return {
            "filename": filename,
            "total_pages": len(raw_documents),
            "total_chunks": len(chunks),
            "figures_extracted": len(extracted_figures),
            "summary": summary_info.get("summary", ""),
            "suggested_questions": summary_info.get("suggested_questions", [])
        }

    def ingest_standalone_image(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Ingest a standalone image file, generate vision caption, and index into FAISS."""
        clean_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lstrip(".")
        saved_img_path = os.path.join(IMAGES_DIR, f"{clean_name}.{ext}")
        shutil.copyfile(file_path, saved_img_path)

        caption = self.caption_image_with_vision(saved_img_path, context_hint=f"Standalone Image: {filename}")
        
        with Image.open(saved_img_path) as im:
            w, h = im.size

        img_doc = Document(
            page_content=f"[STANDALONE IMAGE: {filename}]\nVisual Content Analysis:\n{caption}",
            metadata={
                "type": "image",
                "source_file": filename,
                "page_number": 1,
                "image_path": saved_img_path,
                "image_url": f"/uploads/extracted_images/{clean_name}.{ext}",
                "caption": caption,
                "width": w,
                "height": h
            }
        )

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents([img_doc], self.embeddings)
        else:
            self.vector_store.add_documents([img_doc])

        self._save_vector_store()

        return {
            "filename": filename,
            "total_pages": 1,
            "total_chunks": 1,
            "figures_extracted": 1,
            "summary": f"Image '{filename}' indexed successfully.",
            "suggested_questions": [
                f"What is shown in {filename}?",
                f"Describe the key visual elements in {filename}."
            ]
        }

    def generate_document_summary(self, sample_docs: list, filename: str) -> Dict[str, Any]:
        """Generate a concise summary and sample questions for NotebookLM UX."""
        sample_text = "\n\n".join([doc.page_content for doc in sample_docs])[:4000]
        prompt = f"""
You are an expert document analyst.
Analyze the following excerpt from the uploaded document "{filename}":

{sample_text}

Provide a response in the exact format below:
SUMMARY: A 2 to 3 sentence concise overview of what this document is about.
QUESTIONS:
1. First sample question user might ask
2. Second sample question user might ask
3. Third sample question user might ask
4. Fourth sample question user might ask
"""
        try:
            response = self.llm.invoke(prompt)
            content = self._extract_text(response)

            summary = ""
            questions = []

            if "SUMMARY:" in content:
                parts = content.split("QUESTIONS:")
                summary = parts[0].replace("SUMMARY:", "").strip()
                if len(parts) > 1:
                    raw_q = parts[1].strip().split("\n")
                    questions = [q.strip().lstrip("0123456789.- ") for q in raw_q if q.strip()]
            else:
                summary = content[:300]
                questions = [
                    f"What is the main summary of {filename}?",
                    f"What are the key points in {filename}?",
                    f"Can you extract major findings from {filename}?"
                ]

            return {
                "summary": summary,
                "suggested_questions": questions[:4]
            }
        except Exception as e:
            print(f"Error generating document summary: {e}")
            return {
                "summary": f"Document '{filename}' uploaded successfully.",
                "suggested_questions": [
                    f"What is the main topic of {filename}?",
                    f"Summarize the key information in {filename}."
                ]
            }

    def query(self, question: str, top_k: int = 4, query_image_b64: Optional[str] = None) -> Dict[str, Any]:
        """Perform Multimodal RAG retrieval and answer generation with visual citations."""
        if not self.vector_store:
            return {
                "answer": "No documents or images have been indexed yet. Please upload a PDF or image first!",
                "citations": []
            }

        # Search top relevant chunks (combines text & image descriptions)
        relevant_docs = self.vector_store.similarity_search(question, k=top_k)

        # Build structured context & citation mappings
        context_blocks = []
        citations = []
        image_parts = []

        for idx, doc in enumerate(relevant_docs, start=1):
            source_file = doc.metadata.get("source_file", "Document")
            page_num = doc.metadata.get("page_number", 1)
            doc_type = doc.metadata.get("type", "text")
            snippet = doc.page_content.strip()

            if doc_type == "image":
                img_path = doc.metadata.get("image_path")
                img_url = doc.metadata.get("image_url", "")
                caption = doc.metadata.get("caption", snippet)
                
                context_blocks.append(
                    f"--- [FIGURE CITATION {idx}] (File: {source_file}, Page {page_num}) ---\n"
                    f"Visual Description: {caption}"
                )
                citations.append({
                    "id": idx,
                    "type": "figure",
                    "file": source_file,
                    "page": page_num,
                    "snippet": caption[:280] + ("..." if len(caption) > 280 else ""),
                    "image_url": img_url,
                    "width": doc.metadata.get("width", 0),
                    "height": doc.metadata.get("height", 0)
                })

                # If image file exists on disk, add base64 image part so Gemini inspects pixels directly
                if img_path and os.path.exists(img_path):
                    b64_data = self._encode_image(img_path)
                    if b64_data:
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": b64_data}
                        })
            else:
                context_blocks.append(
                    f"--- [DOCUMENT CITATION {idx}] (File: {source_file}, Page {page_num}) ---\n"
                    f"{snippet}"
                )
                citations.append({
                    "id": idx,
                    "type": "text",
                    "file": source_file,
                    "page": page_num,
                    "snippet": snippet[:280] + ("..." if len(snippet) > 280 else ""),
                    "image_url": None
                })

        formatted_context = "\n\n".join(context_blocks)

        prompt_text = f"""
You are NotebookLM AI, an intelligent multimodal research and synthesis assistant.
Your goal is to answer the user's question accurately using the provided document citations and any accompanying visual figures.

DOCUMENT & FIGURE CITATIONS:
{formatted_context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Provide a comprehensive, articulate, and well-structured answer.
2. Direct inline citation: Every time you reference a source or visual figure, add an inline citation tag:
   - For text excerpts: `[Doc {citations[0]['id'] if citations else 1} (Pg X)]`
   - For visual figures / charts: `[Fig. {citations[0]['id'] if citations else 1} (Pg X)]`
3. Highlight key terms and metrics in **bold**.
4. When charts or visual diagrams are cited, explain the data points and conclusions shown in the visual.
5. If the context does not contain sufficient information, state: "Based on the indexed documents, I could not find information regarding [topic]."
"""

        # Construct multimodal message parts
        message_content = [{"type": "text", "text": prompt_text}]

        # Add user query image if attached
        if query_image_b64:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": query_image_b64}
            })

        # Add retrieved figure images
        for img_part in image_parts[:3]:  # Include up to 3 retrieved figures
            message_content.append(img_part)

        try:
            response = self.llm.invoke([HumanMessage(content=message_content)])
            answer_text = self._extract_text(response)
            return {
                "answer": answer_text,
                "citations": citations
            }
        except Exception as e:
            print(f"Error executing Multimodal Gemini query: {e}")
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "citations": citations
            }

    def clear_all(self):
        """Clear all stored vector embeddings, uploaded PDFs, and extracted figures."""
        if os.path.exists(VECTOR_DB_DIR):
            shutil.rmtree(VECTOR_DB_DIR)
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        if os.path.exists(UPLOADS_DIR):
            shutil.rmtree(UPLOADS_DIR)
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            os.makedirs(IMAGES_DIR, exist_ok=True)
        self.vector_store = None
        print("Cleared vector store, documents, and visual figure archives.")
