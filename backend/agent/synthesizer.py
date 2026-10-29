import os
import json
import base64
import mimetypes
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from backend.config import GOOGLE_API_KEY, LLM_MODELS_FALLBACK
from backend.models import Citation

class MultimodalSynthesizer:
    """Generates grounded multimodal answers with structured citations and streaming support."""

    def __init__(self):
        self.llm = None
        self.streaming_llm = None
        self._init_models()

    def _init_models(self):
        for model in LLM_MODELS_FALLBACK:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                    max_output_tokens=2500
                )
                self.streaming_llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                    max_output_tokens=2500,
                    streaming=True
                )
                self.active_model = model
                break
            except Exception as e:
                print(f"Synthesizer fallback from {model}: {e}")

    def _encode_image_file(self, image_path: str) -> Optional[str]:
        if not image_path or not os.path.exists(image_path):
            return None
        mime, _ = mimetypes.guess_type(image_path)
        mime = mime or "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def build_prompt_and_media(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_context: Optional[Dict[str, Any]] = None,
        query_image_b64: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Citation]]:
        context_blocks = []
        citations: List[Citation] = []
        image_parts = []

        for idx, item in enumerate(retrieved_chunks, start=1):
            meta = item.get("metadata", {})
            modality = meta.get("modality", "text")
            src_file = meta.get("source_file", "Document")
            page_num = meta.get("page_number", 1)
            content_str = item.get("content", "").strip()
            img_url = meta.get("image_url")
            img_path = meta.get("image_path")
            score = item.get("rerank_score") or item.get("score") or 0.85

            cite_type = modality if modality in ["text", "figure", "table", "audio", "video"] else "text"

            if modality == "figure" or modality == "video":
                caption = meta.get("caption", content_str)
                context_blocks.append(
                    f"--- [FIGURE CITATION {idx}] (File: {src_file}, Page {page_num}) ---\n"
                    f"Visual Caption & OCR Data:\n{caption}"
                )
                citations.append(Citation(
                    id=idx,
                    chunk_id=item.get("chunk_id", f"c_{idx}"),
                    type=cite_type,
                    file=src_file,
                    page=page_num,
                    snippet=caption[:280] + ("..." if len(caption) > 280 else ""),
                    image_url=img_url,
                    score=score
                ))
                if img_path and os.path.exists(img_path) and len(image_parts) < 3:
                    b64 = self._encode_image_file(img_path)
                    if b64:
                        image_parts.append({"type": "image_url", "image_url": {"url": b64}})

            elif modality == "table":
                context_blocks.append(
                    f"--- [TABLE CITATION {idx}] (File: {src_file}, Page {page_num}) ---\n"
                    f"{content_str}"
                )
                citations.append(Citation(
                    id=idx,
                    chunk_id=item.get("chunk_id", f"c_{idx}"),
                    type="table",
                    file=src_file,
                    page=page_num,
                    snippet=content_str[:280] + ("..." if len(content_str) > 280 else ""),
                    score=score
                ))

            elif modality == "audio":
                context_blocks.append(
                    f"--- [AUDIO CITATION {idx}] (File: {src_file}) ---\n"
                    f"{content_str}"
                )
                citations.append(Citation(
                    id=idx,
                    chunk_id=item.get("chunk_id", f"c_{idx}"),
                    type="audio",
                    file=src_file,
                    page=page_num,
                    snippet=content_str[:280] + ("..." if len(content_str) > 280 else ""),
                    image_url=img_url,
                    score=score
                ))

            else:
                context_blocks.append(
                    f"--- [DOCUMENT CITATION {idx}] (File: {src_file}, Page {page_num}) ---\n"
                    f"{content_str}"
                )
                citations.append(Citation(
                    id=idx,
                    chunk_id=item.get("chunk_id", f"c_{idx}"),
                    type="text",
                    file=src_file,
                    page=page_num,
                    snippet=content_str[:280] + ("..." if len(content_str) > 280 else ""),
                    score=score
                ))

        formatted_context = "\n\n".join(context_blocks) if context_blocks else "No relevant documents found."

        # History context
        history_str = ""
        if conversation_context:
            summary = conversation_context.get("summary", "")
            recent = conversation_context.get("recent_messages", [])
            if summary:
                history_str += f"\nPRIOR CONVERSATION SUMMARY:\n{summary}\n"
            if recent:
                history_str += "\nRECENT TURNS:\n" + "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent[-4:]])

        prompt_text = f"""
You are NexusRAG, an advanced Multimodal Agentic Research Synthesizer.
Synthesize a comprehensive, rigorous, and grounded answer to the user's question using the retrieved citations and visual figure inputs.

RETRIEVED MULTIMODAL CITATIONS:
{formatted_context}
{history_str}

USER QUESTION:
{question}

CITATION INSTRUCTIONS:
1. Ground your synthesis strictly in the provided citations and attached figures.
2. Direct inline citation footnotes are REQUIRED:
   - For text excerpt: `[Doc X (Pg Y)]` (e.g. `[Doc 1 (Pg 2)]`)
   - For visual figures / charts / diagrams: `[Fig. X (Pg Y)]` (e.g. `[Fig. 2 (Pg 1)]`)
   - For tables: `[Table X (Pg Y)]` (e.g. `[Table 3 (Pg 1)]`)
   - For audio excerpts: `[Audio X]`
3. Highlight key metrics, names, algorithms, and numbers in **bold**.
4. When explaining charts or diagrams, detail all observed axes, trends, and data points.
5. If the context is missing the answer, say: "Based on the indexed documents, no relevant information was found regarding [topic]."
"""
        message_parts = [{"type": "text", "text": prompt_text}]
        if query_image_b64:
            message_parts.append({"type": "image_url", "image_url": {"url": query_image_b64}})
        for img_part in image_parts:
            message_parts.append(img_part)

        return message_parts, citations

    def _format_grounded_fallback(self, question: str, citations: List[Citation], error_reason: str) -> str:
        """Fallback synthesis when upstream LLM service is unavailable (e.g. 503/504)."""
        lines = [
            f"> [!WARNING]",
            f"> **Upstream LLM Temporarily Unavailable ({error_reason})**: The response below was synthesized directly from the top retrieved document chunks and citations.",
            "",
            f"### Grounded Evidence for: \"{question}\"",
            ""
        ]
        if not citations:
            lines.append("No directly matching indexed content was retrieved for this inquiry.")
        else:
            for c in citations:
                badge = f"[{c.type.capitalize()} {c.id} (Pg {c.page})]" if c.page else f"[{c.type.capitalize()} {c.id}]"
                lines.append(f"- **{badge}** *({c.file})*: {c.snippet}")
        return "\n".join(lines)

    def synthesize(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_context: Optional[Dict[str, Any]] = None,
        query_image_b64: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute synchronous synthesis with resilience against 503/504 outages."""
        message_parts, citations = self.build_prompt_and_media(
            question, retrieved_chunks, conversation_context, query_image_b64
        )
        if not self.llm:
            fallback = self._format_grounded_fallback(question, citations, "LLM not initialized or API key missing")
            return {"answer": fallback, "citations": [c.model_dump() for c in citations]}

        try:
            res = self.llm.invoke([HumanMessage(content=message_parts)])
            ans = res.content
            if isinstance(ans, list):
                ans = "".join([x.get("text", str(x)) if isinstance(x, dict) else str(x) for x in ans])
            return {"answer": str(ans).strip(), "citations": [c.model_dump() for c in citations]}
        except Exception as e:
            err_str = str(e)
            if any(code in err_str for code in ["503", "504", "unavailable", "timeout", "overloaded", "quota"]):
                fallback = self._format_grounded_fallback(question, citations, "Service 503/504 / Overloaded")
                return {"answer": fallback, "citations": [c.model_dump() for c in citations]}
            return {"answer": f"Error synthesizing answer: {err_str}", "citations": [c.model_dump() for c in citations]}

    async def stream_synthesize(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_context: Optional[Dict[str, Any]] = None,
        query_image_b64: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream answer tokens asynchronously with graceful fallback handling."""
        message_parts, citations = self.build_prompt_and_media(
            question, retrieved_chunks, conversation_context, query_image_b64
        )
        if not self.streaming_llm:
            fallback = self._format_grounded_fallback(question, citations, "LLM not initialized")
            for token in fallback.split(" "):
                yield {"type": "token", "content": token + " "}
            yield {
                "type": "done",
                "citations": [c.model_dump() for c in citations]
            }
            return

        try:
            async for chunk in self.streaming_llm.astream([HumanMessage(content=message_parts)]):
                token = chunk.content
                if isinstance(token, list):
                    token = "".join([t.get("text", str(t)) if isinstance(t, dict) else str(t) for t in token])
                if token:
                    yield {"type": "token", "content": str(token)}
            
            yield {
                "type": "done",
                "citations": [c.model_dump() for c in citations]
            }
        except Exception as e:
            err_str = str(e)
            if any(code in err_str for code in ["503", "504", "unavailable", "timeout", "overloaded"]):
                fallback = self._format_grounded_fallback(question, citations, "503/504 Gateway/Service Error")
                for token in fallback.split(" "):
                    yield {"type": "token", "content": token + " "}
                yield {
                    "type": "done",
                    "citations": [c.model_dump() for c in citations]
                }
            else:
                yield {"type": "error", "content": err_str}
