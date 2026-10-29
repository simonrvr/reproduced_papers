"""CLI schema and an end-to-end default run (needs the acquired data)."""

from __future__ import annotations

import json

import pytest
from common import REPO_ROOT, build_project_cli_parser, load_runtime_ready_config

DATA_ROOT = REPO_ROOT / "data"
HOSTED = DATA_ROOT / "better_than_classical" / "pennylane_datasets"


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_flags_are_parsed():
    parser, _ = build_project_cli_parser()
    args = parser.parse_args(
        [
            "--model",
            "SVC",
            "--dataset",
            "mnist_pca",
            "--dataset-param",
            "6",
            "--sweep-id",
            "s",
        ]
    )
    assert (args.model, args.dataset, args.dataset_param, args.sweep_id) == (
        "SVC",
        "mnist_pca",
        6,
        "s",
    )


@pytest.mark.skipif(
    not HOSTED.exists(),
    reason="acquired datasets not present (see LOG.md Data Acquisition Log)",
)
def test_default_config_runs_end_to_end(tmp_path, monkeypatch):
    from better_than_classical.lib import runner

    cfg = load_runtime_ready_config()
    cfg["data_root"] = str(DATA_ROOT)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["metrics_path"] == "metrics.json"
    assert metrics["model"] == "SVC" and metrics["n_grid_points"] == 16
    assert 0.9 <= metrics["test_acc_mean"] <= 1.0
    for name in (
        "run.log",
        "cv_results.csv",
        "scores.csv",
        "best_hyperparameters.json",
        "progress.json",
    ):
        assert (run_dir / name).exists(), name
    log = (run_dir / "run.log").read_text()
    assert log.count("TRAIN_FIT_COMPLETED") == 80
    assert log.count("RUN_COMPLETED") == 1
