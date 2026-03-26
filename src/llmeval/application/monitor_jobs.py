"""MonitorJobsUseCase — polls SLURM until all submitted jobs finish."""

from typing import List

from llmeval.domain.eval_job import EvalJob
from llmeval.infrastructure.slurm.job_monitor import wait_for_jobs


class MonitorJobsUseCase:
    """Block until all submitted SLURM jobs have finished."""

    def execute(self, jobs: List[EvalJob], poll_interval: int = 30) -> None:
        """Wait for all jobs, updating each job's status on completion.

        Args:
            jobs:          Jobs returned by SubmitEvaluationUseCase.execute().
            poll_interval: Seconds between squeue polls.
        """
        wait_for_jobs(jobs, poll_interval=poll_interval)
