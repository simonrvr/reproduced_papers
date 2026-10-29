"""Shared-runtime wiring: the paper is discovered and its runner is called."""

from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

import runtime_lib.runtime as runtime_module
from runtime_lib import run_from_project


def test_runtime_calls_runner_with_cli_overrides(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}

    def fake_import_callable(name: str):
        assert name == "lib.runner.train_and_evaluate"

        def _runner(cfg, run_dir: Path):
            recorded["cfg"] = cfg
            recorded["run_dir"] = run_dir

        return _runner

    monkeypatch.setattr(runtime_module, "import_callable", fake_import_callable)
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--model",
                "LogisticRegression",
                "--dataset-param",
                "4",
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)

    assert recorded["run_dir"] == run_dir
    cfg = recorded["cfg"]
    assert cfg["model"]["name"] == "LogisticRegression"
    assert cfg["dataset"]["param"] == 4
    assert (run_dir / "config_snapshot.json").exists()
