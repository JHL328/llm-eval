"""BBH (Big-Bench Hard) benchmark — 3-shot CoT, pass@1 (greedy), per-task accuracy.

Few-shot prompts : data/knowledge/bbh/bbh_cot_prompts.json  (27 tasks)
Test questions   : lukaemon/bbh  (HuggingFace, per-task split)
Ground truth     : example['target']  — free text (True/False, "(A)", phrases, etc.)

Prompt format (from k2/bbh.py):
  <few-shot CoT prefix for this task>

  Q: <question>
  A: Let's think step by step.
"""

import json
from typing import Any, Dict, List

from llmeval.benchmarks.base import BaseBenchmark
from llmeval.benchmarks.knowledge.answer_extractor import extract_bbh
from llmeval.domain.eval_result import EvalResult
from llmeval.domain.model import Model

BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two",
    "navigate", "object_counting", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]


class BBHBenchmark(BaseBenchmark):
    """BBH — 3-shot CoT, greedy (pass@1), with per-task accuracy breakdown."""

    def _load_fewshot(self) -> Dict[str, str]:
        path = self._resolve_local_path(self.benchmark.dataset.name)
        with open(path) as f:
            return json.load(f)

    def load_dataset(self) -> List[Dict[str, Any]]:
        from datasets import load_dataset  # type: ignore[import]

        fewshot = self._load_fewshot()
        examples = []
        for task in BBH_TASKS:
            ds = load_dataset("lukaemon/bbh", task, split="test", trust_remote_code=True)
            for row in ds:
                examples.append({
                    "task": task,
                    "input": row["input"],
                    "target": row["target"].strip(),
                    "_fewshot": fewshot.get(task, ""),
                })
        return examples

    def build_prompt(self, example: Dict[str, Any]) -> str:
        return (
            example["_fewshot"]
            + "\n\nQ: " + example["input"]
            + "\nA: Let's think step by step."
        )

    def check_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        pred = extract_bbh(prediction)
        return pred.lower() == example["target"].lower()

    def build_result(
        self,
        model: Model,
        predictions: List[List[str]],
        examples: List[Dict[str, Any]],
        per_category: Dict[str, float] = None,
    ) -> EvalResult:
        task_correct: Dict[str, int] = {}
        task_total: Dict[str, int] = {}
        for preds, ex in zip(predictions, examples):
            t = ex["task"]
            task_total[t] = task_total.get(t, 0) + 1
            if preds and self.check_answer(preds[0], ex):
                task_correct[t] = task_correct.get(t, 0) + 1

        per_task = {t: task_correct.get(t, 0) / task_total[t] for t in task_total}
        return super().build_result(model, predictions, examples, per_category=per_task)
