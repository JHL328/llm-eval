"""Abstract base class for all benchmark implementations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmeval.domain.benchmark import Benchmark
from llmeval.domain.eval_result import EvalResult
from llmeval.domain.model import Model


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased combinatorial estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


class BaseBenchmark(ABC):
    """Base class every benchmark must inherit from.

    Subclasses implement three concerns:
      1. Dataset loading      — `load_dataset()`
      2. Prompt construction  — `build_prompt()`
      3. Answer checking      — `check_answer()`

    The evaluation loop (generate → check → aggregate) is shared and lives
    in the application layer, not here. Benchmark classes stay pure:
    no vLLM, no SLURM, no I/O beyond reading local data files.

    Args:
        benchmark: The Benchmark domain object (from config/tasks.yaml).
        repo_root: Absolute path to the llm-eval repo root (for resolving
                   local dataset paths).
    """

    def __init__(self, benchmark: Benchmark, repo_root: Path) -> None:
        self.benchmark = benchmark
        self.repo_root = repo_root

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and return the evaluation examples as a list of dicts.

        Each dict must contain at least the keys needed by `build_prompt`
        and `check_answer`.
        """

    @abstractmethod
    def build_prompt(self, example: Dict[str, Any]) -> str:
        """Build the full input prompt for a single example.

        Args:
            example: One item from `load_dataset()`.

        Returns:
            The prompt string to send to the model.
        """

    @abstractmethod
    def check_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        """Return True if the model's prediction is correct for this example.

        Args:
            prediction: One raw model output string.
            example:    The original example dict (contains ground truth).
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @property
    def stop_tokens(self) -> List[str]:
        """Stop sequences for the inference runner.

        Subclasses override this to provide task-specific stop sequences.
        Falls back to sampling_config.stop (empty for most tasks).
        """
        return list(self.benchmark.sampling_config.stop)

    def build_result(
        self,
        model: Model,
        predictions: List[List[str]],
        examples: List[Dict[str, Any]],
        per_category: Optional[Dict[str, float]] = None,
    ) -> EvalResult:
        """Aggregate predictions into an EvalResult.

        Args:
            model:          The model that was evaluated.
            predictions:    Output of VLLMRunner.generate() —
                            outer list = examples, inner list = n_sampling outputs.
            examples:       The dataset examples (same order as predictions).
            per_category:   Optional dict of category → accuracy breakdowns.

        Returns:
            EvalResult with metrics populated.
        """
        from llmeval.domain.eval_result import EvalResult  # avoid circular at module level

        n_total = len(examples)
        # For pass@1 / accuracy tasks: first (and only) sample per example
        n_correct = sum(
            self.check_answer(preds[0], ex)
            for preds, ex in zip(predictions, examples)
            if preds
        )

        metrics = {"accuracy": n_correct / n_total if n_total else 0.0}

        # pass@k for multi-sample benchmarks
        if self.benchmark.is_pass_at_k:
            metrics.update(self._compute_pass_at_k(predictions, examples))

        return EvalResult(
            model_name=model.name,
            benchmark_name=self.benchmark.name,
            metrics=metrics,
            n_total=n_total,
            n_correct=n_correct,
            per_category=per_category or {},
        )

    def _compute_pass_at_k(
        self,
        predictions: List[List[str]],
        examples: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute pass@k using the unbiased combinatorial estimator."""
        n_samples = self.benchmark.sampling_config.n_sampling
        correct_counts = [
            sum(self.check_answer(pred, ex) for pred in preds)
            for preds, ex in zip(predictions, examples)
        ]

        results: Dict[str, float] = {}
        for k in self.benchmark.sampling_config.k_list:
            scores = [_pass_at_k(n_samples, c, k) for c in correct_counts]
            results[f"pass@{k}"] = sum(scores) / len(scores) if scores else 0.0
        return results

    def _resolve_local_path(self, relative: str) -> Path:
        return self.repo_root / relative
