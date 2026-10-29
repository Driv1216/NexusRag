import os
import json
from typing import List, Dict, Any
from backend.config import EVAL_DIR

GOLDEN_PATH = os.path.join(EVAL_DIR, "golden_qa.json")

class GoldenDatasetManager:
    """Manages ground-truth Q&A datasets for automated evaluation benchmarks."""

    def __init__(self):
        self._ensure_dataset_file()

    def _ensure_dataset_file(self):
        if not os.path.exists(GOLDEN_PATH):
            default_data = [
                {
                    "question": "What is the primary topic discussed in the uploaded document?",
                    "ground_truth": "The document provides an in-depth analysis and technical benchmarks of the system."
                },
                {
                    "question": "What visual figures or charts are included in the document?",
                    "ground_truth": "The document includes visual architecture diagrams, charts, and performance plots."
                }
            ]
            with open(GOLDEN_PATH, "w") as f:
                json.dump(default_data, f, indent=2)

    def load_dataset(self) -> List[Dict[str, Any]]:
        if not os.path.exists(GOLDEN_PATH):
            self._ensure_dataset_file()
        try:
            with open(GOLDEN_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def add_qa_pair(self, question: str, ground_truth: str):
        data = self.load_dataset()
        data.append({"question": question, "ground_truth": ground_truth})
        with open(GOLDEN_PATH, "w") as f:
            json.dump(data, f, indent=2)
