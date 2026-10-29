"""Sweep coordinator: one ``implementation.py`` run per (model, dataset param) cell.

Adapted from ``papers/contextuality_inductive_bias/utils/run_sweep.py``. A
sweep is the ``sweep`` block of one experiment config (``models`` x
``dataset_params``); every cell is one planned run with its own output
directory::

    outdir/sweeps/<sweep_id>/
        sweep.log, sweep_status.json, summary.csv, summary.json
        runs/<run_id>/attempt<k>/run_<timestamp>/   (run.log, run_status.json, metrics.json, ...)
        runs/<run_id>/attempt<k>/worker.out

Each planned run gets at most two attempts (initial + one automatic retry).
Only this coordinator writes ``sweep.log`` and ``sweep_status.json``; workers
write their own ``run.log``. Grid-search workers report progress through
``progress.json`` (fits done / total), copied into the run's sweep entry.
The 5 retrain seeds live inside each run, so a sweep has one repetition per
cell; the summary is descriptive (no candidate is "selected").

Usage (from the repository root)::

    python3 papers/better_than_classical/utils/run_sweep.py \\
        papers/better_than_classical/configs/<config>.json --sweep-id <id> --max-workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR.parent)]

from better_than_classical.lib.experiment_logging import (  # noqa: E402
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)

logger = logging.getLogger("better_than_classical.sweep")
MAX_ATTEMPTS = 2
PROGRESS_POLL_SECONDS = 20
# One thread per worker: parallelism comes from running cells side by side.
WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}
SUMMARY_COLUMNS = (
    "model",
    "dataset",
    "param",
    "test_acc_mean",
    "test_acc_std",
    "train_acc_mean",
    "best_mean_cv_accuracy",
    "released_test_acc_mean",
    "released_test_acc_std",
    "difference_of_means",
    "n_score_seeds_failed",
    "n_failed_grid_points",
    "n_not_converged_cv_fits",
    "total_fit_seconds",
)


def paper_code_fingerprint() -> str:
    """SHA-256 over the paper's code, CLI schema, configs and requirements."""
    digest = hashlib.sha256()
    patterns = (
        "lib/*.py",
        "lib/qml_benchmarks/*.py",
        "lib/qml_benchmarks/models/*.py",
        "utils/*.py",
        "cli.json",
        "configs/*.json",
        "requirements.txt",
    )
    files = sorted({p for pattern in patterns for p in PROJECT_DIR.glob(pattern)})
    for path in files:
        digest.update(str(path.relative_to(PROJECT_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _configure_logging(sweep_dir: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)sZ [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (
        logging.FileHandler(sweep_dir / "sweep.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


class Coordinator:
    """Plan, launch, track and summarise one sweep."""

    def __init__(
        self,
        config_path: Path,
        sweep_id: str,
        sweep_root: Path,
        max_workers: int,
        models: list[str] | None,
        dataset_params: list[int] | None,
    ) -> None:
        self.config_path = config_path
        self.config = json.loads(config_path.read_text())
        sweep_block = self.config["sweep"]
        self.models = models or list(sweep_block["models"])
        self.dataset_params = dataset_params or [
            int(d) for d in sweep_block["dataset_params"]
        ]
        self.sweep_id = sweep_id
        self.sweep_dir = sweep_root / sweep_id
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.runs: list[dict] = []
        for model in self.models:
            for param in self.dataset_params:
                self.runs.append(
                    {
                        "run_id": f"{sweep_id}__{model}__{param}",
                        "candidate": {
                            "model": model,
                            "dataset": self.config["dataset"]["name"],
                            "param": param,
                        },
                        "seed": self.config.get("seed", 42),
                        "repetition": 0,
                        "status": "PLANNED",
                        "run_dir": None,
                        "run_log": None,
                        "run_status": None,
                        "config_path": None,
                        "metrics_path": None,
                        "attempts": [],
                    }
                )
        self.status = {
            "schema_version": 1,
            "sweep_id": sweep_id,
            "status": "RUNNING",
            "started_at": None,
            "completed_at": None,
            "code": {
                **git_state(REPO_ROOT),
                "paper_code_sha256": paper_code_fingerprint(),
            },
            "config_path": str(config_path.relative_to(REPO_ROOT)),
            "stage": self.config.get("experiment", {}).get("stage"),
            "expected_runs": len(self.runs),
            "runs": self.runs,
            "summary_path": None,
            "selected_candidates": [],
            "selection": "none (descriptive sweep: every cell is reported)",
            "max_workers": max_workers,
            "error": None,
        }

    def _write_status(self, log_progress: bool = True) -> None:
        with self.lock:
            write_json(self.sweep_dir / "sweep_status.json", self.status)
            if log_progress and self.status["status"] == "RUNNING":
                logger.info("SWEEP_PROGRESS | %s", sweep_progress(self.status))

    def _command(self, run: dict, outdir: Path) -> list[str]:
        return [
            sys.executable,
            str(REPO_ROOT / "implementation.py"),
            "--paper",
            "better_than_classical",
            "--config",
            str(self.config_path.relative_to(PROJECT_DIR)),
            "--outdir",
            str(outdir),
            "--sweep-id",
            self.sweep_id,
            "--run-id",
            run["run_id"],
            "--repetition",
            str(run["repetition"]),
            "--model",
            run["candidate"]["model"],
            "--dataset-param",
            str(run["candidate"]["param"]),
        ]

    def _track_progress(
        self, run: dict, outdir: Path, process: subprocess.Popen
    ) -> None:
        while True:
            try:
                process.wait(timeout=PROGRESS_POLL_SECONDS)
                return
            except subprocess.TimeoutExpired:
                pass
            progress_files = list(outdir.glob("run_*/progress.json"))
            if not progress_files:
                continue
            try:
                progress = json.loads(progress_files[0].read_text())
            except (OSError, ValueError):
                continue  # file being replaced; read it next time
            with self.lock:
                run["progress"] = progress
            self._write_status()

    def _execute(self, run: dict) -> None:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            outdir = self.sweep_dir / "runs" / run["run_id"] / f"attempt{attempt}"
            outdir.mkdir(parents=True, exist_ok=True)
            command = self._command(run, outdir)
            with self.lock:
                run["status"] = "RUNNING"
            event = "CANDIDATE_STARTED" if attempt == 1 else "CANDIDATE_RETRY_STARTED"
            logger.info(
                "%s | run_id=%s | candidate=%s | attempt=%d",
                event,
                run["run_id"],
                run["candidate"],
                attempt,
            )
            self._write_status()
            started = time.time()
            with (outdir / "worker.out").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, **WORKER_ENV},
                )
                self._track_progress(run, outdir, process)
            elapsed = time.time() - started
            run_dirs = sorted(outdir.glob("run_*"))
            outcome = self._check_run(run_dirs)
            record = {
                "attempt": attempt,
                "command": " ".join(command),
                "exit_code": process.returncode,
                "elapsed_s": round(elapsed, 1),
                "run_dir": str(run_dirs[0].relative_to(self.sweep_dir))
                if run_dirs
                else None,
                "outcome": outcome,
            }
            with self.lock:
                run["attempts"].append(record)
            if outcome == "COMPLETED":
                run_dir = run_dirs[0]
                run_status = json.loads((run_dir / "run_status.json").read_text())
                with self.lock:
                    run.update(
                        status="COMPLETED",
                        run_dir=str(run_dir.relative_to(self.sweep_dir)),
                        run_log=str((run_dir / "run.log").relative_to(self.sweep_dir)),
                        run_status=str(
                            (run_dir / "run_status.json").relative_to(self.sweep_dir)
                        ),
                        config_path=str(
                            (run_dir / run_status["config_path"]).relative_to(
                                self.sweep_dir
                            )
                        ),
                        metrics_path=str(
                            (run_dir / run_status["metrics_path"]).relative_to(
                                self.sweep_dir
                            )
                        ),
                    )
                    run.pop("progress", None)
                logger.info(
                    "CANDIDATE_COMPLETED | run_id=%s | metrics=%s | elapsed_s=%.1f",
                    run["run_id"],
                    run["metrics_path"],
                    elapsed,
                )
                self._write_status()
                return
            if attempt < MAX_ATTEMPTS:
                logger.warning(
                    "CANDIDATE_ATTEMPT_FAILED | run_id=%s | attempt=%d | outcome=%s | exit=%s",
                    run["run_id"],
                    attempt,
                    outcome,
                    process.returncode,
                )
        with self.lock:
            run["status"] = "FAILED"
            run.pop("progress", None)
        logger.error(
            "CANDIDATE_FAILED | run_id=%s | attempts=%d", run["run_id"], MAX_ATTEMPTS
        )
        self._write_status()

    @staticmethod
    def _check_run(run_dirs: list[Path]) -> str:
        if len(run_dirs) != 1:
            return f"INVALID (found {len(run_dirs)} run directories)"
        status_path = run_dirs[0] / "run_status.json"
        if not status_path.is_file():
            return "INVALID (no run_status.json)"
        run_status = json.loads(status_path.read_text())
        if run_status.get("status") != "COMPLETED":
            return f"FAILED (run status {run_status.get('status')})"
        try:
            json.loads((run_dirs[0] / run_status["metrics_path"]).read_text())
        except (OSError, ValueError) as exc:
            return f"INVALID (metrics unreadable: {exc})"
        return "COMPLETED"

    def _summarize(self) -> Path:
        """Write summary.csv / summary.json from every completed run's metrics.json."""
        rows = []
        for run in self.runs:
            row = {"run_id": run["run_id"], "status": run["status"], **run["candidate"]}
            if run["status"] == "COMPLETED":
                metrics = json.loads((self.sweep_dir / run["metrics_path"]).read_text())
                row.update({k: metrics[k] for k in SUMMARY_COLUMNS if k in metrics})
            rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(self.sweep_dir / "summary.csv", index=False)
        summary = {
            "sweep_id": self.sweep_id,
            "expected_runs": len(self.runs),
            "completed_runs": int((table["status"] == "COMPLETED").sum()),
            "coverage": "complete"
            if (table["status"] == "COMPLETED").all()
            else "partial",
            "rows": table.to_dict(orient="records"),
        }
        write_json(self.sweep_dir / "summary.json", summary)
        return self.sweep_dir / "summary.json"

    def run(self) -> int:
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        if (self.sweep_dir / "sweep_status.json").exists():
            raise FileExistsError(f"Sweep directory already used: {self.sweep_dir}")
        _configure_logging(self.sweep_dir)
        self.status["started_at"] = utc_now()
        self._write_status(log_progress=False)
        logger.info(
            "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | models=%s | dataset_params=%s | "
            "workers=%d | commit=%s | dirty=%s | paper_code_sha256=%s",
            self.sweep_id,
            len(self.runs),
            self.models,
            self.dataset_params,
            self.max_workers,
            self.status["code"]["commit"],
            self.status["code"]["dirty"],
            self.status["code"]["paper_code_sha256"],
        )
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                list(pool.map(self._execute, self.runs))
            summary_path = self._summarize()
            self.status["summary_path"] = str(summary_path.relative_to(self.sweep_dir))
            fingerprint_end = paper_code_fingerprint()
            if fingerprint_end != self.status["code"]["paper_code_sha256"]:
                logger.warning(
                    "CODE_CHANGED_DURING_SWEEP | start=%s | end=%s",
                    self.status["code"]["paper_code_sha256"],
                    fingerprint_end,
                )
                self.status["code"]["paper_code_sha256_end"] = fingerprint_end
            completed = sum(run["status"] == "COMPLETED" for run in self.runs)
            self.status["completed_at"] = utc_now()
            if completed == len(self.runs):
                self.status["status"] = "COMPLETED"
                self._write_status(log_progress=False)
                logger.info(
                    "SWEEP_COMPLETED | summary=%s | selected=none (descriptive sweep) | completed=%d/%d",
                    self.status["summary_path"],
                    completed,
                    len(self.runs),
                )
                return 0
            if not completed:
                self.status.update(
                    status="FAILED",
                    error={
                        "type": "AllRunsFailed",
                        "message": "No planned run completed",
                    },
                )
                self._write_status(log_progress=False)
                logger.error(
                    "SWEEP_FAILED | sweep_id=%s | no planned run completed",
                    self.sweep_id,
                )
                return 1
            self.status["status"] = "PARTIAL"
            self._write_status(log_progress=False)
            logger.warning(
                "SWEEP_PARTIAL | summary=%s | completed=%d/%d",
                self.status["summary_path"],
                completed,
                len(self.runs),
            )
            return 1
        except Exception as exc:
            self.status.update(
                status="FAILED",
                completed_at=utc_now(),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            self._write_status(log_progress=False)
            logger.exception("SWEEP_FAILED | sweep_id=%s", self.sweep_id)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "config", type=Path, help="Experiment config with a 'sweep' block"
    )
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument(
        "--models", nargs="+", default=None, help="Subset of the config's models"
    )
    parser.add_argument(
        "--dataset-params",
        nargs="+",
        type=int,
        default=None,
        help="Subset of the params",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=PROJECT_DIR / "outdir" / "sweeps",
        help="Parent directory of sweep folders",
    )
    args = parser.parse_args(argv)
    return Coordinator(
        args.config.resolve(),
        args.sweep_id,
        args.sweep_root.resolve(),
        args.max_workers,
        args.models,
        args.dataset_params,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
