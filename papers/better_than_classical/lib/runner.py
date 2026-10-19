"""Runtime entry point for the better_than_classical reproduction."""

from __future__ import annotations

import logging
from pathlib import Path


def train_and_evaluate(cfg, run_dir: Path) -> None:
    """Run one configured experiment and write its artifacts to ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration produced by the shared runtime.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.
    """
    # TODO: Phase 02 replaces this scaffold artifact with the benchmark
    # pipeline (dataset generation, hyperparameter search, model evaluation).
    logger = logging.getLogger(__name__)
    artifact = run_dir / "done.txt"
    artifact.write_text(
        "Scaffold run complete. The benchmark pipeline is not implemented yet.\n",
        encoding="utf-8",
    )
    logger.info("Wrote scaffold artifact: %s", artifact)
