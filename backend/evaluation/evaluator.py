import os
import json
from typing import List, Dict, Any, Optional
from backend.models import EvaluationSummary, QuestionEvalResult, QueryRequest
from backend.evaluation.metrics import RAGEvaluationMetrics
from backend.evaluation.golden_dataset import GoldenDatasetManager
from backend.config import EVAL_DIR

LATEST_EVAL_PATH = os.path.join(EVAL_DIR, "latest_eval_summary.json")

class RAGEvaluator:
    """Orchestrates comprehensive RAG evaluation runs."""

    def __init__(self, engine):
        self.engine = engine
        self.metrics = RAGEvaluationMetrics()
        self.dataset_manager = GoldenDatasetManager()

    def run_evaluation(self, test_questions: Optional[List[str]] = None, top_k: int = 4) -> EvaluationSummary:
        """Run batch evaluation over test questions or golden dataset."""
        if not test_questions:
            golden_items = self.dataset_manager.load_dataset()
            q_list = [g["question"] for g in golden_items]
            gt_map = {g["question"]: g.get("ground_truth", "") for g in golden_items}
        else:
            q_list = test_questions
            gt_map = {}

        if not q_list:
            q_list = [
                "Summarize the key findings in the documents.",
                "What are the main quantitative metrics reported?"
            ]

        results: List[QuestionEvalResult] = []
        faith_scores = []
        rel_scores = []
        prec_scores = []
        rec_scores = []

        for q in q_list:
            # Execute retrieval & query
            req = QueryRequest(question=q, top_k=top_k, enable_agent=True)
            res = self.engine.query(req)

            contexts = [f"[{c.type.upper()}] {c.snippet}" for c in res.citations]

            # Score via LLM Judge
            score_data = self.metrics.evaluate_qa(
                question=q,
                retrieved_contexts=contexts,
                generated_answer=res.answer,
                ground_truth=gt_map.get(q, "")
            )

            q_res = QuestionEvalResult(
                question=q,
                ground_truth=gt_map.get(q, ""),
                retrieved_contexts=contexts,
                answer=res.answer,
                faithfulness_score=score_data["faithfulness"],
                relevancy_score=score_data["relevancy"],
                context_precision=score_data["context_precision"],
                context_recall=score_data["context_recall"],
                critique=score_data["critique"]
            )
            results.append(q_res)

            faith_scores.append(q_res.faithfulness_score)
            rel_scores.append(q_res.relevancy_score)
            prec_scores.append(q_res.context_precision)
            rec_scores.append(q_res.context_recall)

        avg_faith = round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else 0.0
        avg_rel = round(sum(rel_scores) / len(rel_scores), 3) if rel_scores else 0.0
        avg_prec = round(sum(prec_scores) / len(prec_scores), 3) if prec_scores else 0.0
        avg_rec = round(sum(rec_scores) / len(rec_scores), 3) if rec_scores else 0.0
        overall = round((avg_faith + avg_rel + avg_prec + avg_rec) / 4.0, 3)

        summary = EvaluationSummary(
            total_evaluated=len(results),
            average_faithfulness=avg_faith,
            average_relevancy=avg_rel,
            average_context_precision=avg_prec,
            average_context_recall=avg_rec,
            overall_rag_score=overall,
            results=results
        )

        # Cache latest evaluation summary
        try:
            with open(LATEST_EVAL_PATH, "w") as f:
                json.dump(summary.model_dump(), f, indent=2)
        except Exception as e:
            print(f"Error saving eval summary: {e}")

        return summary

    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """Fetch previously stored evaluation run."""
        if os.path.exists(LATEST_EVAL_PATH):
            try:
                with open(LATEST_EVAL_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None
