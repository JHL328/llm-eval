"""SubmitEvaluationUseCase — build EvalJobs and submit them to SLURM."""

import os
from typing import List, Optional

from llmeval.domain.benchmark import Benchmark, DatasetSource
from llmeval.domain.eval_job import EvalJob
from llmeval.domain.model import Model
from llmeval.infrastructure.config_loader import ConfigLoader
from llmeval.infrastructure.result_store import ResultStore
from llmeval.infrastructure.slurm.job_submitter import JobSubmitter

# Tasks that run in the isolated code conda env (has evalplus installed)
_CODE_TASKS = {"humaneval", "mbpp"}


class SubmitEvaluationUseCase:
    """Build (model × benchmark) EvalJobs and submit them to SLURM.

    Args:
        config_loader: Loaded ConfigLoader instance.
        result_store:  ResultStore for skip-if-done checks.
        dry_run:       Write SLURM scripts but do not call sbatch.
    """

    def __init__(
        self,
        config_loader: ConfigLoader,
        result_store: ResultStore,
        dry_run: bool = False,
    ) -> None:
        self.cfg = config_loader
        self.store = result_store
        self.dry_run = dry_run

        repo_root = str(config_loader.repo_root)
        slurm_defaults = config_loader.slurm_defaults()

        self._submitter_primary = JobSubmitter(
            repo_root=repo_root,
            cluster_cfg=slurm_defaults,
            conda_env=config_loader.conda_env("primary"),
        )
        self._submitter_code = JobSubmitter(
            repo_root=repo_root,
            cluster_cfg=slurm_defaults,
            conda_env=config_loader.conda_env("code"),
        )

    def execute(
        self,
        task_names: List[str],
        model_names: Optional[List[str]] = None,
        model_type: Optional[str] = None,
    ) -> List[EvalJob]:
        """Submit jobs for all (task, model) pairs.

        Args:
            task_names:  Task names to evaluate (keys from tasks.yaml).
            model_names: Optional allowlist of model names.
            model_type:  Optional filter — "base" or "sft".

        Returns:
            List of EvalJob objects (RUNNING status, or PENDING for dry_run).
        """
        models = self.cfg.load_models(model_type=model_type)
        if model_names:
            models = [m for m in models if m.name in model_names]

        output_root = self.cfg.output_root()
        submitted: List[EvalJob] = []

        for task_name in task_names:
            benchmark = self.cfg.load_benchmark(task_name)
            for model in models:
                if self.store.is_done(task_name, model.name) and not self.dry_run:
                    print(f"[skip] {model.name}/{task_name} — result already exists")
                    continue

                output_dir = os.path.join(output_root, task_name, model.name)
                job = EvalJob(model=model, benchmark=benchmark, output_dir=output_dir)
                eval_command = self._build_eval_command(job, benchmark, model)
                submitter = (
                    self._submitter_code
                    if task_name in _CODE_TASKS
                    else self._submitter_primary
                )

                try:
                    job_id = submitter.write_and_submit(
                        job, eval_command, dry_run=self.dry_run
                    )
                    if job_id:
                        print(f"[submit] {model.name}/{task_name} → SLURM job {job_id}")
                    submitted.append(job)
                except RuntimeError as e:
                    print(f"[error] {model.name}/{task_name}: {e}")

        return submitted

    def _build_eval_command(
        self, job: EvalJob, benchmark: Benchmark, model: Model
    ) -> str:
        """Return the shell command embedded in the SLURM job script."""
        if benchmark.dataset.source == DatasetSource.LM_HARNESS:
            return self._lm_harness_command(job, benchmark, model)
        return self._vllm_run_command(job, benchmark, model)

    def _vllm_run_command(
        self, job: EvalJob, benchmark: Benchmark, model: Model
    ) -> str:
        gpus = benchmark.slurm_overrides.get("gpus", model.gpus)
        return (
            f"python -m llmeval.interfaces.cli.run_job"
            f" --task {benchmark.name}"
            f" --model-path {model.path}"
            f" --model-name {model.name}"
            f" --model-type {model.type.value}"
            f" --gpus {gpus}"
            f" --output-dir {job.output_dir}"
        )

    def _lm_harness_command(
        self, job: EvalJob, benchmark: Benchmark, model: Model
    ) -> str:
        from llmeval.benchmarks.knowledge.ifeval import IFEvalBenchmark
        from llmeval.benchmarks.likelihood.likelihood_tasks import LikelihoodBenchmark

        if benchmark.name == "ifeval":
            instance = IFEvalBenchmark(benchmark, self.cfg.repo_root)
        else:
            instance = LikelihoodBenchmark(benchmark, self.cfg.repo_root)

        return instance.lm_eval_args(
            model_path=model.path,
            output_dir=job.output_dir,
            model_type=model.type.value,
        )
