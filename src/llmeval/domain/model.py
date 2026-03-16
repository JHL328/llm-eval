from dataclasses import dataclass
from enum import Enum


class ModelType(str, Enum):
    BASE = "base"   # raw pretrained checkpoint — no chat template
    SFT  = "sft"    # instruction-tuned — apply chat template


@dataclass(frozen=True)
class Model:
    """Immutable entity representing a model to evaluate.

    Attributes:
        name:  Human-readable identifier, used as output directory name.
        path:  Absolute path to the checkpoint on the cluster.
        type:  BASE or SFT — controls whether a chat template is applied.
        gpus:  Number of GPUs for vLLM tensor parallelism.
    """

    name: str
    path: str
    type: ModelType
    gpus: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Model name must not be empty")
        if not self.path:
            raise ValueError("Model path must not be empty")
        if self.gpus < 1:
            raise ValueError(f"gpus must be >= 1, got {self.gpus}")

    @property
    def is_sft(self) -> bool:
        return self.type == ModelType.SFT

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        """Build from a models.yaml entry."""
        return cls(
            name=d["name"],
            path=d["path"],
            type=ModelType(d.get("type", "base")),
            gpus=d.get("gpus", 1),
        )
