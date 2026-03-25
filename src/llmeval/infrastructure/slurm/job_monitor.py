"""Simple squeue-based SLURM job monitor (evaluate_gsm8k.py pattern)."""

import subprocess
import time
from typing import List

from llmeval.domain.eval_job import EvalJob, JobStatus


_POLL_INTERVAL = 30  # seconds between squeue polls
_SQUEUE_TIMEOUT = 10  # seconds before squeue call is considered hung


def _query_running_ids(job_ids: List[str]) -> set:
    """Return the set of job IDs still active in SLURM (running or pending)."""
    ids_str = ",".join(job_ids)
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", ids_str, "-o", "%i"],
            capture_output=True,
            text=True,
            timeout=_SQUEUE_TIMEOUT,
        )
        if result.returncode != 0:
            err = result.stderr.strip()
            # Known transient errors — treat as "still running" to avoid false completion
            if "slurm_load_jobs error" in err or "Socket timed out" in err:
                return set(job_ids)
            # Jobs not found → all completed
            return set()
        lines = result.stdout.strip().splitlines()
        return {line.strip() for line in lines if line.strip()}
    except subprocess.TimeoutExpired:
        return set(job_ids)


def wait_for_jobs(jobs: List[EvalJob], poll_interval: int = _POLL_INTERVAL) -> None:
    """Block until all submitted SLURM jobs have finished.

    Polls squeue every `poll_interval` seconds. Updates each EvalJob's status
    to COMPLETED or FAILED based on the presence of result.json / fail.flag.

    Args:
        jobs:          List of EvalJob objects that have been submitted.
        poll_interval: Seconds between squeue polls.
    """
    import os

    pending = [j for j in jobs if j.slurm_job_id is not None]
    if not pending:
        return

    active_ids = [j.slurm_job_id for j in pending]
    print(f"\n⏳ Monitoring {len(active_ids)} SLURM jobs...")

    while active_ids:
        try:
            running = _query_running_ids(active_ids)
            # Jobs that left the queue — check result/fail flag
            finished_ids = [jid for jid in active_ids if jid not in running]
            for job in pending:
                if job.slurm_job_id in finished_ids:
                    if os.path.exists(job.result_path):
                        job.mark_completed()
                    else:
                        job.mark_failed()

            active_ids = list(running)

            if active_ids:
                print(
                    f"  {len(active_ids)} job(s) still running/pending. "
                    f"Next check in {poll_interval}s..."
                )
                time.sleep(poll_interval)

        except Exception as e:
            print(f"⚠️  Exception in monitor loop: {e}. Retrying in {poll_interval}s...")
            time.sleep(poll_interval)

    completed = sum(1 for j in pending if j.status == JobStatus.COMPLETED)
    failed    = sum(1 for j in pending if j.status == JobStatus.FAILED)
    print(f"\n✅ All jobs finished — {completed} completed, {failed} failed.")
