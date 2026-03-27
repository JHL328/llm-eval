"""MATH-500 benchmark — 4-shot CoT, pass@k evaluation.

Dataset: data/math/math500_test.jsonl (copied from qwen2.5-math/evaluation/data/math500/)
Columns: problem, solution, answer, subject, level, unique_id
Ground truth: `answer` field (pre-extracted from \\boxed{} in solution).

Uses qwen2.5-math grader (extract_answer + math_equal) instead of math_verify
for more stable comparison of LaTeX expressions, fractions, symbolic answers.
"""

import json
from typing import Any, Dict, List

from llmeval.benchmarks.base import BaseBenchmark
from llmeval.benchmarks.math.qwen_math_grader import (
    qwen_extract_answer,
    qwen_math_equal,
    qwen_strip_string,
)

# 4-shot examples from qwen2.5-math/evaluation/examples.py (math500 split)
_FEWSHOT_EXAMPLES = [
    (
        r"What is $\frac{2^2 \cdot 2^{-3}}{2^3 \cdot 2^{-2}}$?",
        r"We compute that \[\frac{2^2 \cdot 2^{-3}}{2^3 \cdot 2^{-2}} = \frac{2^{2 - 3}}{2^{3 - 2}} = \frac{2^{-1}}{2^1} = 2^{-1 - 1} = 2^{-2} = \frac{1}{2^2} = \boxed{\frac{1}{4}}.",
    ),
    (
        r"What is the value of $\dfrac{3 \times 4}{6}?$",
        r"Calculating the numerator first, $\dfrac{3 \times 4}{6} = \dfrac{12}{6} = \boxed{2}$.",
    ),
    (
        r"How many positive integers less than $101$ are multiples of either $5$ or $7$, but not both at once?",
        r"There are $20$ positive multiples of $5$ less than $101$. There are $14$ positive multiples of $7$ less than $101$. The least common multiple of $5$ and $7$ is $35$, and there are $2$ positive multiples of $35$ less than $101$. This means there are $20 - 2 = 18$ multiples of $5$ that aren't multiples of $7$, and $14 - 2 = 12$ multiples of $7$ that aren't multiples of $5$, for a total of $18 + 12 = \boxed{30}$.",
    ),
    (
        r"What is the product of the two largest one-digit primes and the largest two-digit prime?",
        r"The two largest one-digit primes are 5 and 7; the largest two-digit prime is 97. The product is $5 \cdot 7 \cdot 97 = \boxed{3395}$.",
    ),
]

_FEWSHOT_PREFIX = "".join(
    f"Question: {q}\nSolution: Let's think step by step. {a}\n\n"
    for q, a in _FEWSHOT_EXAMPLES
)


class Math500Benchmark(BaseBenchmark):
    """MATH-500 test set — 4-shot CoT, pass@k."""

    def load_dataset(self) -> List[Dict[str, Any]]:
        path = self._resolve_local_path(self.benchmark.dataset.name)
        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    def build_prompt(self, example: Dict[str, Any]) -> str:
        return _FEWSHOT_PREFIX + f"Question: {example['problem']}\nSolution: Let's think step by step."

    def extract_answer(self, prediction: str) -> str:
        """Extract answer using qwen2.5-math style extraction."""
        return qwen_extract_answer(prediction, data_name="math500")

    def check_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        # Ground truth: extract from solution field (same as qwen2.5-math)
        gold = qwen_extract_answer(example["solution"], data_name="math500")
        # Prediction: extract answer then compare
        pred = qwen_extract_answer(prediction, data_name="math500")
        return qwen_math_equal(pred, gold)
