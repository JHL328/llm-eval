"""Atomic JSON read/write for result.json files."""

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from llmeval.domain.eval_result import EvalResult


class ResultStore:
    """Handles atomic reads and writes of per-model result.json files,
    and aggregated passk.json files.

    Uses fcntl file locking so concurrent SLURM jobs writing to the same
    aggregated file do not corrupt it.
    """

    RESULT_FILENAME = "result.json"
    FAIL_FLAG = "fail.flag"
    PASSK_FILENAME = "passk.json"

    def __init__(self, output_root: str) -> None:
        self.output_root = Path(output_root)

    # ------------------------------------------------------------------
    # Per-model result.json
    # ------------------------------------------------------------------

    def result_path(self, task_name: str, model_name: str) -> Path:
        return self.output_root / task_name / model_name / self.RESULT_FILENAME

    def fail_flag_path(self, task_name: str, model_name: str) -> Path:
        return self.output_root / task_name / model_name / self.FAIL_FLAG

    def is_completed(self, task_name: str, model_name: str) -> bool:
        return self.result_path(task_name, model_name).exists()

    def is_failed(self, task_name: str, model_name: str) -> bool:
        return self.fail_flag_path(task_name, model_name).exists()

    def is_done(self, task_name: str, model_name: str) -> bool:
        """True if the job already has a result or a fail flag (skip re-submission)."""
        return self.is_completed(task_name, model_name) or self.is_failed(task_name, model_name)

    def read_result(self, task_name: str, model_name: str) -> Optional[EvalResult]:
        path = self.result_path(task_name, model_name)
        if not path.exists():
            return None
        with open(path) as f:
            return EvalResult.from_dict(json.load(f))

    def write_result(self, result: EvalResult, task_name: str) -> None:
        """Write result atomically using a temp file + os.replace (POSIX atomic)."""
        path = self.result_path(task_name, result.model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # Aggregated passk.json  (one file per task, all models)
    # ------------------------------------------------------------------

    def passk_path(self, task_name: str) -> Path:
        return self.output_root / task_name / self.PASSK_FILENAME

    def update_passk_atomically(
        self, task_name: str, model_name: str, metrics: Dict[str, Any]
    ) -> None:
        """Atomically append or update one model's metrics in passk.json.

        Safe to call from concurrent SLURM jobs.
        """
        path = self.passk_path(task_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            try:
                all_results = json.load(f)
            except (json.JSONDecodeError, ValueError):
                all_results = {}
            all_results[model_name] = metrics
            f.seek(0)
            f.truncate()
            json.dump(all_results, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

    def read_passk(self, task_name: str) -> Dict[str, Any]:
        path = self.passk_path(task_name)
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)
