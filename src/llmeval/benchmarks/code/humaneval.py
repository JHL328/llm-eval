"""HumanEval+ benchmark — 0-shot, pass@k evaluation.

Dataset  : evalplus get_human_eval_plus()  (HumanEval with augmented tests)
Prompt   : problem["prompt"]  (function signature + docstring; model completes the body)
EOS stop : \\ndef, \\nclass, \\nimport, \\nfrom, \\nassert  (same as generate_code.py)
Solution : prompt + completion  (prepend prompt before running tests)
"""

from typing import Any, Dict, List

from llmeval.benchmarks.base import BaseBenchmark
from llmeval.benchmarks.code.code_executor import check_one, get_problems_and_groundtruth

# Stop tokens that signal end-of-function for HumanEval (from generate_code.py)
HUMANEVAL_STOP_TOKENS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n#",
    "\ndef ",
    "\nclass ",
    "\nimport ",
    "\nfrom ",
    "\nassert ",
]


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval+ — 0-shot completion, pass@k."""

    @property
    def stop_tokens(self) -> List[str]:
        return HUMANEVAL_STOP_TOKENS

    def load_dataset(self) -> List[Dict[str, Any]]:
        problems, expected_output = get_problems_and_groundtruth("humaneval")
        # Store expected_output on instance for check_answer
        self._expected_output = expected_output
        self._problems = problems
        return [dict(p) for p in problems.values()]

    def build_prompt(self, example: Dict[str, Any]) -> str:
        # The prompt is the function signature + docstring.
        # The model should complete the function body.
        return example["prompt"]

    def check_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        problem = self._problems[example["task_id"]]
        # For HumanEval, solution = prompt + completion
        solution = example["prompt"] + prediction
        return check_one(
            dataset="humaneval",
            problem=problem,
            solution=solution,
            expected_output=self._expected_output,
        )
