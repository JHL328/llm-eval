from dataclasses import dataclass
from enum import Enum

from llmeval.domain.sampling_config import SamplingConfig


class BenchmarkCategory(str, Enum):
    MATH       = "math"
    KNOWLEDGE  = "knowledge"
    CODE       = "code"
    LIKELIHOOD = "likelihood"


class DatasetSource(str, Enum):
    LOCAL      = "local"       # file under data/
    HF         = "hf"          # HuggingFace datasets
    LM_HARNESS = "lm_harness"  # delegated to lm-evaluation-harness


@dataclass(frozen=True)
class DatasetConfig:
    """Where and how to load the benchmark dataset."""

    source: DatasetSource
    # For LOCAL: relative path from repo root (e.g. "data/math/gsm8k_test.jsonl")
    # For HF:    HuggingFace dataset id (e.g. "openai/gsm8k")
    # For LM_HARNESS: lm_eval task name (e.g. "arc_easy")
    name: str
    subset: str = ""   # HF subset/config name, if applicable
    split: str = "test"

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetConfig":
        source = DatasetSource(d["source"])
        if source == DatasetSource.LOCAL:
            return cls(source=source, name=d["path"])
        if source == DatasetSource.LM_HARNESS:
            return cls(source=source, name=d["task_name"])
        return cls(
            source=source,
            name=d["name"],
            subset=d.get("subset", ""),
            split=d.get("split", "test"),
        )


@dataclass(frozen=True)
class Benchmark:
    """Immutable entity describing a benchmark task.

    Attributes:
        name:            Task identifier matching keys in tasks.yaml.
        category:        High-level category (math, knowledge, code, likelihood).
        sampling_config: How to sample from the model.
        dataset:         Where to load evaluation data from.
        num_shots:       Number of few-shot examples to prepend.
        prompt_type:     Prompt format (cot, flan_cot, default, ...).
    """

    name: str
    category: BenchmarkCategory
    sampling_config: SamplingConfig
    dataset: DatasetConfig
    num_shots: int = 0
    prompt_type: str = "default"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Benchmark name must not be empty")
        if self.num_shots < 0:
            raise ValueError(f"num_shots must be >= 0, got {self.num_shots}")

    @property
    def is_pass_at_k(self) -> bool:
        """True when multiple samples are drawn and pass@k is computed."""
        return self.sampling_config.n_sampling > 1

    @classmethod
    def from_dict(cls, name: str, category: BenchmarkCategory, d: dict) -> "Benchmark":
        """Build from a tasks.yaml task block."""
        sampling = SamplingConfig.from_dict({
            **d["sampling"],
            "k_list": d.get("k_list", [1]),
        })
        return cls(
            name=name,
            category=category,
            sampling_config=sampling,
            dataset=DatasetConfig.from_dict(d["dataset"]),
            num_shots=d.get("num_shots", 0),
            prompt_type=d.get("prompt_type", "default"),
        )
