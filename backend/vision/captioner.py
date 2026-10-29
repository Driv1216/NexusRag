import os
import base64
import mimetypes
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from backend.config import GOOGLE_API_KEY, LLM_MODELS_FALLBACK

class VisionCaptioner:
    """Uses Gemini Vision to produce dense multimodal descriptions, chart analysis, and OCR."""

    def __init__(self):
        self.api_key = GOOGLE_API_KEY
        self.llm = None
        self._init_vision_llm()

    def _init_vision_llm(self):
        for model in LLM_MODELS_FALLBACK:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=self.api_key,
                    max_output_tokens=1500,
                    temperature=0.2
                )
                self.active_model = model
                break
            except Exception as e:
                print(f"VisionCaptioner fallback from {model}: {e}")

    def encode_image_b64(self, image_path: str) -> Optional[str]:
        if not os.path.exists(image_path):
            return None
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def caption_image(self, image_path: str, context_hint: str = "") -> str:
        """Generate structured visual breakdown: type, quantitative metrics, OCR text."""
        b64_uri = self.encode_image_b64(image_path)
        if not b64_uri:
            return "Visual excerpt from document."

        if not self.llm:
            return f"Visual diagram/figure: {os.path.basename(image_path)}. ({context_hint})"

        prompt = f"""
You are an expert technical visual analyst and OCR parser.
Analyze this image/figure extracted from a document.
Context: {context_hint}

Provide:
1. Visual Type: (e.g. Bar Chart, Architecture Diagram, Flowchart, Circuit, Table, Plot, Photograph)
2. Detailed Description: Summarize all trends, axes, labels, values, flow connections, and visual conclusions.
3. Visible Text & OCR: Transcribe key text, numbers, and formulas verbatim.
"""
        try:
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": b64_uri}}
                ]
            )
            res = self.llm.invoke([msg])
            content = res.content
            if isinstance(content, list):
                text_parts = [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in content]
                return "\n".join(text_parts).strip()
            return str(content).strip()
        except Exception as e:
            print(f"Error captioning image {image_path}: {e}")
            return f"Figure depicting data/diagram related to {context_hint}."
