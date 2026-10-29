import json
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import GOOGLE_API_KEY, LLM_MODELS_FALLBACK

class QueryRouter:
    """Agentic Query Router deciding optimal execution path for user queries."""

    def __init__(self):
        self.llm = None
        for model in LLM_MODELS_FALLBACK:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.0,
                    max_output_tokens=400
                )
                break
            except Exception as e:
                print(f"Router fallback from {model}: {e}")

    def _rule_based_route(self, question: str) -> Dict[str, Any]:
        """High-precision heuristic routing for offline execution and fast paths."""
        q_lower = question.strip().lower()

        # 1. Direct conversational
        if any(q_lower.startswith(g) or q_lower == g for g in [
            "hi", "hello", "hey", "who are you", "what can you do", "help", "thanks", "thank you", "good morning"
        ]):
            return {
                "route": "DIRECT",
                "sub_queries": [],
                "reasoning": "Standard conversational greeting or meta-inquiry."
            }

        # 2. Table / Spreadsheet inquiry
        if any(term in q_lower for term in ["table", "spreadsheet", "column", "rows", "csv", "xlsx", "data sheet"]):
            return {
                "route": "TABLE_LOOKUP",
                "sub_queries": [question],
                "reasoning": "Quantitative or tabular schema lookup inquiry."
            }

        # 3. Multi-step comparative inquiry
        if any(term in q_lower for term in ["compare", "vs", "versus", "difference between", "differences between"]):
            # Split comparative query into sub-queries if possible
            parts = [question]
            if " vs " in q_lower:
                sub = q_lower.split(" vs ")
                parts = [f"Details on {sub[0].strip()}", f"Details on {sub[1].strip()}"]
            elif " versus " in q_lower:
                sub = q_lower.split(" versus ")
                parts = [f"Details on {sub[0].strip()}", f"Details on {sub[1].strip()}"]
            elif "compare " in q_lower:
                parts = [f"Aspects of {question}", "Comparative metric evaluation"]
            return {
                "route": "MULTI_STEP",
                "sub_queries": parts,
                "reasoning": "Comparative multi-part query decomposed for granular retrieval."
            }

        return {
            "route": "RETRIEVAL",
            "sub_queries": [question],
            "reasoning": "Standard hybrid retrieval path."
        }

    def route_query(self, question: str, has_image: bool = False) -> Dict[str, Any]:
        """Classify user question and generate sub-queries if needed."""
        if has_image:
            return {
                "route": "RETRIEVAL",
                "sub_queries": [question],
                "reasoning": "Query contains visual image attachment requiring multimodal comparison."
            }

        # Fast rule-based check
        heuristic = self._rule_based_route(question)
        if heuristic["route"] in ["DIRECT", "TABLE_LOOKUP", "MULTI_STEP"] and not self.llm:
            return heuristic
        if heuristic["route"] == "DIRECT":
            return heuristic

        if not self.llm:
            return heuristic

        prompt = f"""
You are an intelligent RAG query router.
Analyze the user's inquiry and classify it into EXACTLY ONE route:

ROUTES:
- DIRECT: Conversational greetings, general AI assistance, or system capabilities inquiry without needing documents.
- RETRIEVAL: Standard factual, analytical, or visual question answering based on indexed documents.
- MULTI_STEP: Complex multi-part, comparative, or reasoning questions that require breaking down into 2-3 focused sub-queries.
- TABLE_LOOKUP: Specific statistical, numerical, or tabular lookup inquiries.

USER INQUIRY: "{question}"

Respond in STRICT JSON format with NO markdown wrapping:
{{
  "route": "DIRECT" | "RETRIEVAL" | "MULTI_STEP" | "TABLE_LOOKUP",
  "sub_queries": ["sub query 1", "sub query 2"],
  "reasoning": "short explanation"
}}
"""
        try:
            res = self.llm.invoke(prompt)
            raw = res.content if hasattr(res, 'content') else str(res)
            if isinstance(raw, list):
                raw = "".join([x.get("text", str(x)) if isinstance(x, dict) else str(x) for x in raw])
            raw = raw.strip().lstrip("```json").rstrip("```").strip()
            parsed = json.loads(raw)
            return parsed
        except Exception as e:
            # Fallback to single retrieval
            return {
                "route": "RETRIEVAL",
                "sub_queries": [question],
                "reasoning": f"Default fallback due to router parsing note: {e}"
            }
