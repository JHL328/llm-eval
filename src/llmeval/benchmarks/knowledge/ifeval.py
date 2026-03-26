"""IFEval benchmark — wraps lm-evaluation-harness.

IFEval does NOT use our vLLM inference runner.  It delegates evaluation
to ``lm_eval`` CLI (from third_party/lm-evaluation-harness).  The job
submitter detects ``dataset.source == "lm_harness"`` and generates an
``lm_eval``-based SLURM script instead of the standard vLLM worker script.

This module provides:
  IFEvalBenchmark.lm_eval_args()  — the extra CLI args needed for ifeval
  IFEvalBenchmark.parse_result()  — extract EvalResult from lm_eval JSON output
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from llmeval.benchmarks.base import BaseBenchmark
from llmeval.domain.eval_result import EvalResult
from llmeval.domain.model import Model


class IFEvalBenchmark(BaseBenchmark):
    """IFEval — lm-harness backend, 0-shot, pass@1.

    load_dataset / build_prompt / check_answer are NOT used at inference time.
    The job submitter calls lm_eval directly.
    """

    # lm_eval task name
    LM_EVAL_TASK = "ifeval"

    # Metrics extracted from lm_eval results JSON
    PRIMARY_METRIC = "prompt_level_strict_acc,none"
    REPORTED_METRICS = [
        "prompt_level_strict_acc,none",
        "inst_level_strict_acc,none",
        "prompt_level_loose_acc,none",
        "inst_level_loose_acc,none",
    ]

    def lm_eval_args(self, model_path: str, output_dir: str, model_type: str = "base") -> str:
        """Return the lm_eval CLI invocation string for this benchmark."""
        args = (
            f"lm_eval --model vllm "
            f"--model_args pretrained={model_path},tensor_parallel_size=1,"
            f"dtype=bfloat16,gpu_memory_utilization=0.95,max_model_len=8192,trust_remote_code=True "
            f"--tasks {self.LM_EVAL_TASK} "
            f"--output_path {output_dir} "
            f"--gen_kwargs temperature=0,top_p=1,max_gen_toks=4096 "
            f"--batch_size auto "
            f"--log_samples "
            f"--num_fewshot 0 "
            f"--trust_remote_code"
        )
        if model_type == "sft":
            args += " \\\n  --apply_chat_template"
        return args

    def parse_result(self, result_json_path: Path, model: Model) -> Optional[EvalResult]:
        """Parse lm_eval's results_*.json and return an EvalResult."""
        import json

        if not result_json_path.exists():
            return None

        with open(result_json_path) as f:
            data = json.load(f)

        task_results = data.get("results", {}).get(self.LM_EVAL_TASK, {})
        if not task_results:
            return None

        metrics = {
            k.split(",")[0]: float(task_results[k])
            for k in self.REPORTED_METRICS
            if k in task_results
        }
        accuracy = metrics.get("prompt_level_strict_acc", 0.0)
        n_total = len(data.get("samples", {}).get(self.LM_EVAL_TASK, []))

        return EvalResult(
            model_name=model.name,
            benchmark_name=self.benchmark.name,
            metrics=metrics,
            n_total=n_total,
            n_correct=round(accuracy * n_total),
        )

    # ------------------------------------------------------------------
    # BaseBenchmark interface — not used for lm_harness tasks
    # ------------------------------------------------------------------

    def load_dataset(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("IFEval uses lm_eval; call lm_eval_args() instead.")

    def build_prompt(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError("IFEval uses lm_eval; call lm_eval_args() instead.")

    def check_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        raise NotImplementedError("IFEval uses lm_eval; call lm_eval_args() instead.")
