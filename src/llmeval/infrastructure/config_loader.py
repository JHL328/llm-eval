"""Loads cluster.yaml, models.yaml, and tasks.yaml into domain objects."""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from llmeval.domain.benchmark import Benchmark, BenchmarkCategory
from llmeval.domain.model import Model
from llmeval.domain.sampling_config import SamplingConfig


# Maps task name prefixes/exact names to benchmark categories
_CATEGORY_MAP: Dict[str, BenchmarkCategory] = {
    "gsm8k":         BenchmarkCategory.MATH,
    "math500":       BenchmarkCategory.MATH,
    "aime24":        BenchmarkCategory.MATH,
    "aime25":        BenchmarkCategory.MATH,
    "mmlu":          BenchmarkCategory.KNOWLEDGE,
    "mmlu_pro":      BenchmarkCategory.KNOWLEDGE,
    "mmlu_flan":     BenchmarkCategory.KNOWLEDGE,
    "bbh":           BenchmarkCategory.KNOWLEDGE,
    "gpqa":          BenchmarkCategory.KNOWLEDGE,
    "ifeval":        BenchmarkCategory.KNOWLEDGE,
    "humaneval":     BenchmarkCategory.CODE,
    "mbpp":          BenchmarkCategory.CODE,
    "arc_easy":      BenchmarkCategory.LIKELIHOOD,
    "arc_challenge": BenchmarkCategory.LIKELIHOOD,
    "hellaswag":     BenchmarkCategory.LIKELIHOOD,
    "piqa":          BenchmarkCategory.LIKELIHOOD,
    "winogrande":    BenchmarkCategory.LIKELIHOOD,
    "triviaqa":      BenchmarkCategory.LIKELIHOOD,
    "drop":          BenchmarkCategory.LIKELIHOOD,
    "commonsense_qa":BenchmarkCategory.LIKELIHOOD,
}


class ConfigLoader:
    """Reads all YAML config files and vends domain objects.

    Args:
        repo_root: Absolute path to the llm-eval repo root.
                   Defaults to the directory three levels above this file.
    """

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        if repo_root is None:
            # src/llmeval/infrastructure/config_loader.py → repo root is 3 levels up
            repo_root = Path(__file__).resolve().parents[3]
        self.repo_root = repo_root
        self._cluster: Optional[dict] = None
        self._models_raw: Optional[dict] = None
        self._tasks_raw: Optional[dict] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_yaml(self, relative_path: str) -> dict:
        path = self.repo_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            return yaml.safe_load(f)

    @property
    def cluster(self) -> dict:
        if self._cluster is None:
            self._cluster = self._load_yaml("config/cluster.yaml")
        return self._cluster

    @property
    def _models_raw_data(self) -> dict:
        if self._models_raw is None:
            self._models_raw = self._load_yaml("config/models.yaml")
        return self._models_raw

    @property
    def _tasks_raw_data(self) -> dict:
        if self._tasks_raw is None:
            self._tasks_raw = self._load_yaml("config/tasks.yaml")
        return self._tasks_raw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_models(self, model_type: Optional[str] = None) -> List[Model]:
        """Return all models, optionally filtered by type ("base" or "sft")."""
        raw = self._models_raw_data
        models: List[Model] = []
        for type_key in ("sft", "base"):
            for entry in raw.get(type_key, []):
                m = Model.from_dict({**entry, "type": type_key})
                if model_type is None or m.type.value == model_type:
                    models.append(m)
        return models

    def load_benchmark(self, task_name: str) -> Benchmark:
        """Return a Benchmark domain object for the given task name."""
        tasks = self._tasks_raw_data
        if task_name not in tasks:
            available = ", ".join(tasks.keys())
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {available}"
            )
        task_dict = tasks[task_name]
        category = _CATEGORY_MAP.get(task_name, BenchmarkCategory.KNOWLEDGE)
        return Benchmark.from_dict(task_name, category, task_dict)

    def load_all_benchmarks(self) -> Dict[str, Benchmark]:
        """Return all benchmarks defined in tasks.yaml."""
        return {name: self.load_benchmark(name) for name in self._tasks_raw_data}

    def slurm_defaults(self) -> dict:
        """Return the cluster-level SLURM defaults."""
        return self.cluster.get("slurm", {})

    def output_root(self) -> str:
        """Return the absolute path to the results output root."""
        return self.cluster["output_root"]

    def conda_env(self, env_type: str = "primary") -> str:
        """Return the conda env path for the given type ('primary', 'code', or 'harness')."""
        return self.cluster["envs"][env_type]

    def resolve_dataset_path(self, relative_path: str) -> Path:
        """Resolve a local dataset path relative to the repo root."""
        return self.repo_root / relative_path
