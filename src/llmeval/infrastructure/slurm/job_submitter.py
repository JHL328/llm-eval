"""Renders Jinja2 SLURM templates and submits jobs via sbatch."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from llmeval.domain.eval_job import EvalJob


_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "slurm_job.sh.j2"


class JobSubmitter:
    """Renders a SLURM job script and submits it with sbatch.

    Args:
        repo_root:    Absolute path to the llm-eval repo root.
        cluster_cfg:  The `slurm` block from cluster.yaml (defaults).
        conda_env:    Full path to the conda env to activate.
    """

    def __init__(self, repo_root: str, cluster_cfg: dict, conda_env: str) -> None:
        self.repo_root = repo_root
        self.cluster_cfg = cluster_cfg
        self.conda_env = conda_env
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            keep_trailing_newline=True,
        )

    def render_script(self, job: EvalJob, eval_command: str) -> str:
        """Render the SLURM job script for the given EvalJob.

        Args:
            job:          The EvalJob entity (provides output_dir, model, benchmark).
            eval_command: The shell command that runs the evaluation.

        Returns:
            Rendered script as a string.
        """
        # Merge cluster defaults with per-task overrides from benchmark.slurm_overrides
        task_slurm = job.benchmark.slurm_overrides
        gpus      = task_slurm.get("gpus",          job.model.gpus)
        mem       = task_slurm.get("mem",            self.cluster_cfg.get("mem", "800G"))
        time      = task_slurm.get("time",           self.cluster_cfg.get("default_time", "12:00:00"))
        cpus      = task_slurm.get("cpus_per_task",  self.cluster_cfg.get("cpus_per_task", 96))
        partition = task_slurm.get("partition",      self.cluster_cfg.get("partition", "main"))

        template = self._jinja_env.get_template(_TEMPLATE_NAME)
        return template.render(
            job_name=f"{job.benchmark.name}_{job.model.name}",
            output_dir=job.output_dir,
            gpus=gpus,
            cpus_per_task=cpus,
            time=time,
            partition=partition,
            mem=mem,
            account=self.cluster_cfg.get("account", ""),
            qos=self.cluster_cfg.get("qos", ""),
            conda_env_path=self.conda_env,
            repo_root=self.repo_root,
            eval_command=eval_command,
        )

    def write_and_submit(
        self, job: EvalJob, eval_command: str, dry_run: bool = False
    ) -> Optional[str]:
        """Write the job script to disk and submit it.

        Args:
            job:          The EvalJob to submit.
            eval_command: Shell command that runs the evaluation.
            dry_run:      If True, write the script but do not call sbatch.

        Returns:
            SLURM job ID string if submitted, None if dry_run.
        """
        os.makedirs(job.output_dir, exist_ok=True)

        script = self.render_script(job, eval_command)
        script_path = job.job_script_path
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)

        if dry_run:
            print(f"[dry-run] Script written to {script_path}")
            print(script)
            return None

        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed for {job.model.name}/{job.benchmark.name}:\n"
                f"{result.stderr.strip()}"
            )

        # sbatch stdout: "Submitted batch job <id>"
        token = result.stdout.strip().split()[-1]
        if not token.isdigit():
            raise RuntimeError(
                f"Unexpected sbatch output (could not parse job ID): {result.stdout.strip()!r}"
            )
        job.mark_submitted(token)
        return token
