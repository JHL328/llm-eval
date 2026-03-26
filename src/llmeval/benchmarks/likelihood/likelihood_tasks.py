"""Likelihood benchmarks — wraps lm-evaluation-harness.

Tasks: arc_easy, arc_challenge, hellaswag, piqa, winogrande,
       triviaqa, nq_open, commonsense_qa, openbookqa, social_iqa,
       truthfulqa_mc2, drop

These tasks use lm_eval's likelihood scoring, NOT our vLLM inference runner.
The job submitter detects ``dataset.source == "lm_harness"`` and generates
an ``lm_eval``-based SLURM script.

This module provides:
  LikelihoodBenchmark.lm_eval_args()  — CLI args for the lm_eval call
  LikelihoodBenchmark.parse_result()  — extract EvalResult from lm_eval JSON
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from llmeval.benchmarks.base import BaseBenchmark
from llmeval.domain.eval_result import EvalResult
from llmeval.domain.model import Model

# Primary metric field per task (from evaluate_likelihood.py TASK_METRIC)
_TASK_METRIC: Dict[str, str] = {
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc_norm,none",
    "triviaqa": "exact_match,remove_whitespace",
    "nq_open": "exact_match,remove_whitespace",
    "commonsense_qa": "acc_norm,none",
    "openbookqa": "acc_norm,none",
    "social_iqa": "acc_norm,none",
    "truthfulqa_mc2": "acc_norm,none",
    "drop": "f1,none",
}

# Short metric name used in EvalResult.metrics dict
_METRIC_SHORT: Dict[str, str] = {
    "acc_norm,none": "acc_norm",
    "exact_match,remove_whitespace": "exact_match",
    "f1,none": "f1",
}


class LikelihoodBenchmark(BaseBenchmark):
    """Likelihood-scoring benchmark — lm-harness backend, pass@1 only.

    load_dataset / build_prompt / check_answer are NOT used at inference time.
    The job submitter calls lm_eval directly via lm_eval_args().
    """

    def lm_eval_args(self, model_path: str, output_dir: str, model_type: str = "base") -> str:
        """Return the lm_eval CLI invocation string for this task."""
        task_name = self.benchmark.dataset.name   # e.g. "arc_easy"
        num_fewshot = self.benchmark.num_shots
        args = (
            f"lm_eval --model vllm "
            f"--model_args pretrained={model_path},tensor_parallel_size=1,"
            f"dtype=bfloat16,gpu_memory_utilization=0.95,trust_remote_code=True "
            f"--tasks {task_name} "
            f"--output_path {output_dir} "
            f"--batch_size auto "
            f"--log_samples "
            f"--num_fewshot {num_fewshot} "
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

        task_name = self.benchmark.dataset.name
        task_results = data.get("results", {}).get(task_name, {})
        if not task_results:
            return None

        metric_key = _TASK_METRIC.get(task_name, "acc_norm,none")
        short_name = _METRIC_SHORT.get(metric_key, metric_key.split(",")[0])
        score = float(task_results.get(metric_key, 0.0))
        # lm_eval does not reliably expose dataset size in results JSON;
        # store 0 to indicate "unknown" — downstream display handles None.
        n_total = 0

        return EvalResult(
            model_name=model.name,
            benchmark_name=self.benchmark.name,
            metrics={short_name: score, "accuracy": score},
            n_total=n_total,
            n_correct=round(score * n_total),
        )

    # ------------------------------------------------------------------
    # BaseBenchmark interface — not used for lm_harness tasks
    # ------------------------------------------------------------------

    def load_dataset(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("Likelihood tasks use lm_eval; call lm_eval_args() instead.")

    def build_prompt(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError("Likelihood tasks use lm_eval; call lm_eval_args() instead.")

    def check_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        raise NotImplementedError("Likelihood tasks use lm_eval; call lm_eval_args() instead.")
