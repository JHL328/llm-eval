from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class SamplingConfig:
    """Immutable value object describing how to sample from a model.

    Attributes:
        temperature:  Generation temperature. 0 = greedy decoding.
        top_p:        Nucleus sampling threshold. 1.0 = disabled.
        n_sampling:   Number of independent samples to draw per problem.
        max_tokens:   Maximum number of new tokens to generate.
        stop:         Optional list of stop strings.
        seed:         Random seed for reproducibility.
        k_list:       Values of k for pass@k reporting. Empty for accuracy tasks.
    """

    temperature: float
    top_p: float
    n_sampling: int
    max_tokens: int
    stop: List[str] = field(default_factory=list)
    seed: int = 42
    k_list: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not (0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.n_sampling < 1:
            raise ValueError(f"n_sampling must be >= 1, got {self.n_sampling}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if any(k > self.n_sampling for k in self.k_list):
            raise ValueError(
                f"All k in k_list must be <= n_sampling ({self.n_sampling}), "
                f"got k_list={self.k_list}"
            )

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0

    @classmethod
    def greedy(cls, max_tokens: int = 4096) -> "SamplingConfig":
        """Convenience constructor for greedy decoding (knowledge benchmarks)."""
        return cls(
            temperature=0,
            top_p=1.0,
            n_sampling=1,
            max_tokens=max_tokens,
            k_list=[1],
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SamplingConfig":
        """Build from a tasks.yaml sampling block."""
        return cls(
            temperature=d["temperature"],
            top_p=d["top_p"],
            n_sampling=d["n_sampling"],
            max_tokens=d["max_tokens"],
            stop=d.get("stop", []),
            seed=d.get("seed", 42),
            k_list=d.get("k_list", []),
        )
