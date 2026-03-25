from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class EvalResult:
    """Immutable value object holding the evaluation outcome for one model × benchmark.

    Attributes:
        model_name:     Name of the evaluated model.
        benchmark_name: Name of the benchmark.
        metrics:        Primary metrics dict, e.g. {"pass@1": 0.72, "pass@4": 0.85}.
        n_total:        Total number of problems evaluated.
        n_correct:      Number of problems answered correctly (for accuracy tasks).
        per_category:   Optional breakdown by subtask/category (e.g. BBH subtasks,
                        MMLU subjects). Maps category name → accuracy.
        raw_outputs:    Optional list of raw model outputs (not persisted by default).
    """

    model_name: str
    benchmark_name: str
    metrics: Dict[str, float]
    n_total: int
    n_correct: int = 0
    per_category: Dict[str, float] = field(default_factory=dict)
    raw_outputs: Optional[List[str]] = field(default=None)

    def __post_init__(self) -> None:
        if self.n_total < 0:
            raise ValueError(f"n_total must be >= 0, got {self.n_total}")
        if self.n_correct < 0:
            raise ValueError(f"n_correct must be >= 0, got {self.n_correct}")
        if self.n_correct > self.n_total:
            raise ValueError(
                f"n_correct ({self.n_correct}) must be <= n_total ({self.n_total})"
            )

    @property
    def accuracy(self) -> float:
        """Simple accuracy for single-sample tasks."""
        if self.n_total == 0:
            return 0.0
        return self.n_correct / self.n_total

    def to_dict(self) -> dict:
        """Serialise to the format written into result.json."""
        return {
            "model": self.model_name,
            "benchmark": self.benchmark_name,
            "metrics": self.metrics,
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "per_category": self.per_category,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalResult":
        return cls(
            model_name=d["model"],
            benchmark_name=d["benchmark"],
            metrics=d["metrics"],
            n_total=d["n_total"],
            n_correct=d.get("n_correct", 0),
            per_category=d.get("per_category", {}),
        )
