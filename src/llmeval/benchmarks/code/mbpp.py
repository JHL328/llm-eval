"""MBPP+ benchmark — 3-shot, pass@k evaluation.

Dataset  : evalplus get_mbpp_plus()  (MBPP with augmented tests)
Prompt   : xllm [BEGIN]/[DONE] format (compatible with generate_code.py)

Prompt format (from xllm/xllm/eval/task/mbpp.py):
    You are an expert Python programmer, and here is your task: {description}
    Your code should pass these tests:

    {test_assertions}
    [BEGIN]

Model generates code body; extract by splitting on "[DONE]".
EOS stop : [DONE], \\n###, \\nassert, \\n```
"""

from typing import Any, Dict, List

from llmeval.benchmarks.base import BaseBenchmark
from llmeval.benchmarks.code.code_executor import check_one, get_problems_and_groundtruth

# Stop tokens for MBPP (from generate_code.py + xllm)
MBPP_STOP_TOKENS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n#",
    "[DONE]",
    "\n###",
    "\nassert",
    "\n```",
]

# Prompt template (from xllm/eval/task/mbpp.py)
_PROMPT_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: "
    "{description} Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
)

# 3-shot few-shot examples (task_ids 2, 3, 4 — same as xllm fewshot_index [1,2,3])
# These are hardcoded so they are always available, even before evalplus is loaded
_FEWSHOT_EXAMPLES = [
    {
        "description": "Write a function to find the shared elements from the given two lists.",
        "tests": (
            "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n"
            "assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))\n"
            "assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))"
        ),
        "solution": (
            "def similar_elements(test_tup1, test_tup2):\n"
            "    return tuple(set(test_tup1) & set(test_tup2))"
        ),
    },
    {
        "description": "Write a python function to identify non-prime numbers.",
        "tests": (
            "assert is_not_prime(2) == False\n"
            "assert is_not_prime(10) == True\n"
            "assert is_not_prime(35) == True"
        ),
        "solution": (
            "import math\n"
            "def is_not_prime(n):\n"
            "    if n == 1:\n"
            "        return True\n"
            "    for i in range(2, int(math.sqrt(n))+1):\n"
            "        if n % i == 0:\n"
            "            return True\n"
            "    return False"
        ),
    },
    {
        "description": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
        "tests": (
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]\n"
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]\n"
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"
        ),
        "solution": (
            "import heapq as hq\n"
            "def heap_queue_largest(nums, n):\n"
            "    return hq.nlargest(n, nums)"
        ),
    },
]

_FEWSHOT_PREFIX = "".join(
    _PROMPT_TEMPLATE.format(description=ex["description"], tests=ex["tests"])
    + ex["solution"] + "\n[DONE]\n\n"
    for ex in _FEWSHOT_EXAMPLES
)


class MBPPBenchmark(BaseBenchmark):
    """MBPP+ — 3-shot [BEGIN]/[DONE] format, pass@k."""

    def load_dataset(self) -> List[Dict[str, Any]]:
        problems, expected_output = get_problems_and_groundtruth("mbpp")
        self._expected_output = expected_output
        self._problems = problems
        # Skip the 3 few-shot task_ids (Mbpp/2, Mbpp/3, Mbpp/4)
        skip = {"Mbpp/2", "Mbpp/3", "Mbpp/4"}
        return [dict(p) for tid, p in problems.items() if tid not in skip]

    def build_prompt(self, example: Dict[str, Any]) -> str:
        description = example["prompt"].strip()  # evalplus uses "prompt" = description text
        # Use the first few test assertions as the visible tests in the prompt
        test_lines = example.get("test_list", [])
        tests = "\n".join(test_lines[:3])  # show at most 3 tests in the prompt
        question = _PROMPT_TEMPLATE.format(description=description, tests=tests)
        return _FEWSHOT_PREFIX + question

    def check_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        # Extract code: take everything before [DONE]
        code = prediction.split("[DONE]")[0].strip()
        # For MBPP, solution is the code itself (no prompt prefix needed)
        problem = self._problems[example["task_id"]]
        return check_one(
            dataset="mbpp",
            problem=problem,
            solution=code,
            expected_output=self._expected_output,
        )
