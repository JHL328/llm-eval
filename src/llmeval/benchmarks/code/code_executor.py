"""Code execution sandbox — wraps evalplus for correctness checking.

Uses evalplus.evaluate.check_correctness() which runs solutions in an
isolated subprocess with timeout. Requires the llmeval-code conda env.

Public API:
    get_problems_and_groundtruth(dataset) — load problems + precompute expected outputs
    check_one(dataset, problem, solution, expected_output) — run tests, return bool
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure third_party/evalplus is importable
_EVALPLUS = Path(__file__).parents[5] / "third_party" / "evalplus"
if str(_EVALPLUS) not in sys.path and _EVALPLUS.is_dir():
    sys.path.insert(0, str(_EVALPLUS))

# Cached problems + ground-truth per dataset (expensive to compute once)
_CACHE: Dict[str, Tuple[Dict, Dict]] = {}


def get_problems_and_groundtruth(dataset: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load problems and pre-compute expected outputs for *dataset*.

    Returns:
        (problems dict, expected_output dict)
        Both are keyed by task_id.
    """
    if dataset in _CACHE:
        return _CACHE[dataset]

    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus  # type: ignore[import]
        from evalplus.evaluate import get_groundtruth  # type: ignore[import]
        from evalplus.eval import estimate_pass_at_k  # noqa: F401 — verifies install
    except ImportError as exc:
        raise ImportError(
            "evalplus is required for code benchmarks. "
            "Install with: pip install -e third_party/evalplus"
        ) from exc

    if dataset == "humaneval":
        problems = get_human_eval_plus()
    elif dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"Unknown code dataset: {dataset!r}. Use 'humaneval' or 'mbpp'.")

    expected_output = get_groundtruth(
        problems,
        hashcode=None,
        tasks_only_output_not_none=[],
    )
    _CACHE[dataset] = (problems, expected_output)
    return problems, expected_output


def check_one(
    dataset: str,
    problem: Dict[str, Any],
    solution: str,
    expected_output: Dict[str, Any],
    fast_check: bool = True,
) -> bool:
    """Execute *solution* against the evalplus test suite.

    Args:
        dataset:         "humaneval" or "mbpp".
        problem:         Problem dict from get_human_eval_plus()/get_mbpp_plus().
        solution:        Full solution string (prompt + completion for HumanEval).
        expected_output: Pre-computed expected outputs from get_groundtruth().
        fast_check:      Stop at first failure (faster).

    Returns:
        True if all base tests pass, False otherwise.
    """
    from evalplus.evaluate import check_correctness  # type: ignore[import]
    from evalplus.eval import PASS  # type: ignore[import]

    task_id = problem["task_id"]
    result = check_correctness(
        dataset=dataset,
        completion_id=0,
        problem=problem,
        solution=solution,
        expected_output=expected_output[task_id],
        base_only=False,
        fast_check=fast_check,
    )
    # Pass requires both base and plus tests to pass
    base_ok = result.get("base", (None,))[0] == PASS
    plus_ok = result.get("plus", (None,))[0] == PASS
    return base_ok and plus_ok
