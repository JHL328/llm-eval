"""Collect and display evaluation results from result.json files."""

from typing import Dict, List, Optional

from llmeval.domain.eval_result import EvalResult
from llmeval.infrastructure.result_store import ResultStore


def collect_results(
    task_names: List[str],
    model_names: List[str],
    result_store: ResultStore,
) -> Dict[str, Dict[str, Optional[EvalResult]]]:
    """Read result.json for each (task, model) pair.

    Returns:
        Nested dict: task_name → model_name → EvalResult (or None if missing).
    """
    results: Dict[str, Dict[str, Optional[EvalResult]]] = {}
    for task in task_names:
        results[task] = {}
        for model in model_names:
            results[task][model] = result_store.read_result(task, model)
    return results


def print_results_table(
    results: Dict[str, Dict[str, Optional[EvalResult]]],
) -> None:
    """Print a formatted summary table to stdout."""
    for task_name, model_results in results.items():
        print(f"\n=== {task_name} ===")
        header = f"{'Model':<40} {'Accuracy':>10} {'pass@1':>8} {'pass@k':>8}  N"
        print(header)
        print("-" * len(header))
        for model_name, result in model_results.items():
            if result is None:
                print(f"{model_name:<40} {'—':>10}")
                continue
            acc = result.metrics.get("accuracy", result.metrics.get("pass@1", 0.0))
            p1 = result.metrics.get("pass@1")
            pk = max(
                (v for k, v in result.metrics.items() if k.startswith("pass@") and k != "pass@1"),
                default=None,
            )
            p1_str = f"{p1:.4f}" if p1 is not None else "—"
            pk_str = f"{pk:.4f}" if pk is not None else "—"
            print(f"{model_name:<40} {acc:>10.4f} {p1_str:>8} {pk_str:>8}  {result.n_total}")
