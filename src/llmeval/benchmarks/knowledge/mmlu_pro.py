"""MMLU-Pro benchmark — 5-shot CoT, pass@1 (greedy), per-category accuracy.

Few-shot prefixes : data/knowledge/mmlu/mmlu_pro_fewshot.json  (extracted from mmlu_pro_prompts.json)
Test questions    : TIGER-Lab/MMLU-Pro  (HuggingFace, 14 categories)
Ground truth      : example['answer']  — single uppercase letter e.g. "I"

Prompt format:
  <few-shot prefix ending with "Question:\\n">
  <question>
  Options:
  A. ...
  ...
  Answer: Let's think step by step.
"""

import json
from typing import Any, Dict, List

from llmeval.benchmarks.base import BaseBenchmark
from llmeval.benchmarks.knowledge.answer_extractor import extract_abcdj
from llmeval.domain.eval_result import EvalResult
from llmeval.domain.model import Model

_OPTION_LABELS = list("ABCDEFGHIJ")


class MMLUProBenchmark(BaseBenchmark):
    """MMLU-Pro — 5-shot CoT, greedy (pass@1), with per-category accuracy."""

    def _load_fewshot(self) -> Dict[str, str]:
        path = self._resolve_local_path("data/knowledge/mmlu/mmlu_pro_fewshot.json")
        with open(path) as f:
            return json.load(f)

    def load_dataset(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset  # type: ignore[import]

        fewshot = self._load_fewshot()
        ds = load_dataset(
            self.benchmark.dataset.name,
            split=self.benchmark.dataset.split,
        )
        examples = []
        for row in ds:
            category = row["category"]
            examples.append({
                "question": row["question"],
                "options": list(row["options"]),
                "answer": row["answer"].strip().upper(),  # e.g. "I"
                "category": category,
                "_fewshot": fewshot.get(category, ""),
            })
        return examples

    def build_prompt(self, example: Dict[str, Any]) -> str:
        # Prefix already ends with "Question:\n"
        prefix = example["_fewshot"]
        q = example["question"].strip()
        opts = "\n".join(
            f"{_OPTION_LABELS[i]}. {opt}"
            for i, opt in enumerate(example["options"])
        )
        return prefix + f"{q}\nOptions:\n{opts}\nAnswer: Let's think step by step."

    def check_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        return extract_abcdj(prediction) == example["answer"]

    def build_result(
        self,
        model: Model,
        predictions: List[List[str]],
        examples: List[Dict[str, Any]],
        per_category: Dict[str, float] = None,
    ) -> EvalResult:
        cat_correct: Dict[str, int] = {}
        cat_total: Dict[str, int] = {}
        for preds, ex in zip(predictions, examples):
            c = ex["category"]
            cat_total[c] = cat_total.get(c, 0) + 1
            if preds and self.check_answer(preds[0], ex):
                cat_correct[c] = cat_correct.get(c, 0) + 1

        per_cat = {c: cat_correct.get(c, 0) / cat_total[c] for c in cat_total}
        return super().build_result(model, predictions, examples, per_category=per_cat)
