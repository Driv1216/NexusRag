from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import GOOGLE_API_KEY, LLM_MODELS_FALLBACK

class QueryRewriter:
    """Implements HyDE (Hypothetical Document Embeddings) and query expansion."""

    def __init__(self):
        self.llm = None
        for model in LLM_MODELS_FALLBACK:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.3,
                    max_output_tokens=300
                )
                break
            except Exception:
                pass

    def generate_hyde_passage(self, question: str) -> str:
        """Generate a hypothetical passage for enhanced dense embedding retrieval."""
        if not self.llm or len(question.split()) < 3:
            return question

        prompt = f"""
Given the question below, write a short, hypothetical academic or technical paragraph (3-4 sentences) that provides a direct, factual answer.
Do not say "I don't know". Write purely as if you are the source document excerpt.

Question: {question}
Hypothetical Excerpt:
"""
        try:
            res = self.llm.invoke(prompt)
            content = res.content
            if isinstance(content, list):
                content = " ".join([c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content])
            return str(content).strip() or question
        except Exception:
            return question
