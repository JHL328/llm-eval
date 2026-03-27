"""vLLM inference wrapper used inside SLURM job scripts."""

from typing import List, Optional

from llmeval.domain.sampling_config import SamplingConfig


class VLLMRunner:
    """Thin wrapper around vLLM's LLM class.

    This class is instantiated inside individual SLURM job scripts (one per
    model × benchmark). It is NOT imported by the submitter or monitor —
    only by the benchmark evaluation scripts that run on the GPU nodes.

    Args:
        model_path:             Absolute path to the model checkpoint.
        tensor_parallel_size:   Number of GPUs (matches model.gpus).
        gpu_memory_utilization: Fraction of GPU memory for vLLM KV cache.
        trust_remote_code:      Passed to vLLM for models with custom code.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        trust_remote_code: bool = True,
        max_model_len: Optional[int] = None,
    ) -> None:
        # Deferred import — vllm is only available on GPU nodes
        from vllm import LLM, SamplingParams as VLLMSamplingParams  # noqa: F401

        llm_kwargs: dict = dict(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self._llm = LLM(**llm_kwargs)
        self._VLLMSamplingParams = VLLMSamplingParams

    def generate(
        self,
        prompts: List[str],
        sampling_config: SamplingConfig,
        use_tqdm: bool = True,
    ) -> List[List[str]]:
        """Generate text for a list of prompts.

        Args:
            prompts:         Input prompts.
            sampling_config: SamplingConfig domain object.
            use_tqdm:        Show vLLM progress bar.

        Returns:
            List of length len(prompts), each element is a list of
            n_sampling generated strings.
        """
        params = self._build_sampling_params(sampling_config)
        outputs = self._llm.generate(prompts, params, use_tqdm=use_tqdm)
        return [
            [output.text for output in req_output.outputs]
            for req_output in outputs
        ]

    def _build_sampling_params(self, cfg: SamplingConfig):
        kwargs = dict(
            n=cfg.n_sampling,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )
        if cfg.is_greedy:
            kwargs["temperature"] = 0
            kwargs["top_p"] = 1.0
        else:
            kwargs["temperature"] = cfg.temperature
            kwargs["top_p"] = cfg.top_p

        if cfg.stop:
            kwargs["stop"] = cfg.stop

        return self._VLLMSamplingParams(**kwargs)

    @classmethod
    def from_model(cls, model, max_model_len: Optional[int] = None, **kwargs) -> "VLLMRunner":
        """Build directly from a Model domain object."""
        return cls(
            model_path=model.path,
            tensor_parallel_size=model.gpus,
            max_model_len=max_model_len,
            **kwargs,
        )
