from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from llmeval.domain.benchmark import Benchmark
from llmeval.domain.model import Model


class JobStatus(str, Enum):
    PENDING   = "pending"    # created, not yet submitted
    RUNNING   = "running"    # sbatch submitted, SLURM job active
    COMPLETED = "completed"  # result.json written successfully
    FAILED    = "failed"     # fail.flag written or job exited non-zero


@dataclass
class EvalJob:
    """Mutable entity tracking one model × one benchmark SLURM job.

    Attributes:
        model:      The model being evaluated.
        benchmark:  The benchmark being run.
        output_dir: Absolute path to this job's output directory.
        slurm_job_id: SLURM job ID after submission, None before.
        status:     Current lifecycle state.
    """

    model: Model
    benchmark: Benchmark
    output_dir: str
    slurm_job_id: Optional[str] = field(default=None)
    status: JobStatus = field(default=JobStatus.PENDING)

    @property
    def result_path(self) -> str:
        return f"{self.output_dir}/result.json"

    @property
    def fail_flag_path(self) -> str:
        return f"{self.output_dir}/fail.flag"

    @property
    def slurm_out_path(self) -> str:
        return f"{self.output_dir}/slurm.out"

    @property
    def slurm_err_path(self) -> str:
        return f"{self.output_dir}/slurm.err"

    @property
    def job_script_path(self) -> str:
        return f"{self.output_dir}/job.sh"

    def mark_submitted(self, slurm_job_id: str) -> None:
        self.slurm_job_id = slurm_job_id
        self.status = JobStatus.RUNNING

    def mark_completed(self) -> None:
        self.status = JobStatus.COMPLETED

    def mark_failed(self) -> None:
        self.status = JobStatus.FAILED
