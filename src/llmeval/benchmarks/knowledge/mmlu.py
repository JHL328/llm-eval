"""MMLU benchmark — 5-shot, pass@1 (greedy decoding), per-subject accuracy.

Few-shot prompts : data/knowledge/mmlu/mmlu_prompts.json
  - Pre-built string per subject: preamble + 5 Q&A shots
  - Build prompt: fewshot_string + "\n\nQ: <question>\n(A) ... A:"

Test questions   : hails/mmlu_no_train (HuggingFace, 57 subjects)
Ground truth     : example['answer'] int 0-3 → "(A)"/"(B)"/"(C)"/"(D)"
"""

import json
from typing import Any, Dict, List, Optional

from llmeval.benchmarks.base import BaseBenchmark
from llmeval.benchmarks.knowledge.answer_extractor import extract_abcd
from llmeval.domain.eval_result import EvalResult
from llmeval.domain.model import Model

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

_LABELS = ["A", "B", "C", "D"]


class MMLUBenchmark(BaseBenchmark):
    """MMLU — 5-shot, greedy (pass@1), with per-subject accuracy breakdown."""

    def _load_fewshot(self) -> Dict[str, str]:
        """Load the pre-built few-shot prefix strings from mmlu_prompts.json."""
        path = self._resolve_local_path(self.benchmark.dataset.name)
        with open(path) as f:
            return json.load(f)

    def load_dataset(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset  # type: ignore[import]

        fewshot = self._load_fewshot()
        examples = []
        for subject in MMLU_SUBJECTS:
            ds = load_dataset("hails/mmlu_no_train", subject, split="test")
            for row in ds:
                examples.append({
                    "subject": subject,
                    "question": row["question"],
                    "choices": list(row["choices"]),
                    "answer": int(row["answer"]),   # 0-3
                    "_fewshot": fewshot.get(subject, ""),
                })
        return examples

    def build_prompt(self, example: Dict[str, Any]) -> str:
        # fewshot is the full pre-built prefix (preamble + 5 shots).
        # We append the test question in the same Q/A format.
        fewshot = example["_fewshot"].rstrip()
        q = example["question"].strip()
        c = example["choices"]
        return (
            fewshot
            + f"\n\nQ: {q}\n"
            + f"(A) {c[0]} (B) {c[1]} (C) {c[2]} (D) {c[3]}\n"
            + "A:"
        )

    def check_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        gold = f"({_LABELS[example['answer']]})"
        return extract_abcd(prediction) == gold

    def build_result(
        self,
        model: Model,
        predictions: List[List[str]],
        examples: List[Dict[str, Any]],
        per_category: Optional[Dict[str, float]] = None,
    ) -> EvalResult:
        """Override to add per-subject accuracy to per_category."""
        subject_correct: Dict[str, int] = {}
        subject_total: Dict[str, int] = {}
        for preds, ex in zip(predictions, examples):
            s = ex["subject"]
            subject_total[s] = subject_total.get(s, 0) + 1
            if preds and self.check_answer(preds[0], ex):
                subject_correct[s] = subject_correct.get(s, 0) + 1

        per_subject = {
            s: subject_correct.get(s, 0) / subject_total[s]
            for s in subject_total
        }
        return super().build_result(model, predictions, examples, per_category=per_subject)
