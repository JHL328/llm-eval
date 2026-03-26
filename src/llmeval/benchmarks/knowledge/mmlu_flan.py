"""MMLU-FLAN benchmark — 4-shot CoT, pass@1 (greedy), per-subject accuracy.

Few-shot prompts : data/knowledge/mmlu/mmlu_cot_prompts.json
Test questions   : hails/mmlu_no_train  (HuggingFace, 57 subjects)
Ground truth     : example['answer'] int 0-3 → "(A)"

Prompt format (from evaluate_mmlu_flan_cot_fewshot.py):
  <few-shot prefix (4 shots with CoT reasoning)>

  Q: <question>
  (A) ... (B) ... (C) ... (D) ...
  A: Let's think step by step.
"""

import json
from typing import Any, Dict, List, Optional

from llmeval.benchmarks.base import BaseBenchmark
from llmeval.benchmarks.knowledge.answer_extractor import extract_abcd
from llmeval.domain.eval_result import EvalResult
from llmeval.domain.model import Model

# Same 57 subjects as MMLU
from llmeval.benchmarks.knowledge.mmlu import MMLU_SUBJECTS

_LABELS = ["A", "B", "C", "D"]
_NUM_SHOTS = 4


class MMLUFlanBenchmark(BaseBenchmark):
    """MMLU-FLAN — 4-shot CoT, greedy (pass@1), with per-subject accuracy."""

    def _load_fewshot(self) -> Dict[str, str]:
        path = self._resolve_local_path(self.benchmark.dataset.name)
        with open(path) as f:
            raw = json.load(f)

        # Trim to num_shots Q&A blocks (same logic as evaluate_mmlu_flan_cot_fewshot.py)
        fewshot: Dict[str, str] = {}
        for subject, content in raw.items():
            blocks = content.split("\nQ: ")
            prefix = blocks[0]
            qa_blocks = blocks[1 : _NUM_SHOTS + 1]
            text = prefix
            for qa in qa_blocks:
                text += "\nQ: " + qa
            fewshot[subject] = text
        return fewshot

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
                    "answer": int(row["answer"]),
                    "_fewshot": fewshot.get(subject, ""),
                })
        return examples

    def build_prompt(self, example: Dict[str, Any]) -> str:
        fewshot = example["_fewshot"].rstrip()
        q = example["question"].strip()
        c = example["choices"]
        return (
            fewshot
            + f"\n\nQ: {q}\n"
            + f"(A) {c[0]} (B) {c[1]} (C) {c[2]} (D) {c[3]}\n"
            + "A: Let's think step by step."
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
