"""Runtime entry point for the better_than_classical reproduction.

One run executes one experiment kind (``cfg["experiment"]["kind"]``):

- ``grid_search``: one (model, dataset) cell of the benchmark with the
  paper's protocol (stages S3-S7): 5-fold CV grid search, selection on mean
  CV accuracy, retraining with seeds 0-4, test accuracy.
- ``score_best``: port validation (stage S2): retrain with the authors'
  released best hyperparameters and compare test accuracy with their released
  5-seed scores.
- ``reanalysis``: experiment E0 (stage S1), recompute the paper's aggregates
  from the released results.
- ``positivity_bias``: experiment E12 (stage S1), the Fig. 3 simulation.

Every run writes ``run.log``, ``run_status.json``, ``config_snapshot.json``
(written by the shared runtime) and ``metrics.json``, plus experiment tables.
Grid-search runs also keep ``progress.json`` (fits done / total) for the sweep
coordinator.
"""

from __future__ import annotations

import ast
import json
import logging
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_PAPER_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PAPER_DIR.parents[1]
_LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}


def _require(cfg: dict[str, Any], dotted: str) -> Any:
    """Return a required config value or fail with the missing key's path."""
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing required config key: {dotted}")
        node = node[key]
    return node


def _configure_device(device: str) -> str:
    """Point JAX at the configured device and return the device actually used.

    The models are JAX/PennyLane estimators; ``cpu`` keeps JAX's default.
    ``cuda`` / ``cuda:<i>`` / ``gpu`` select a GPU and fail clearly when none
    is visible to JAX.
    """
    import jax

    if device == "cpu":
        return str(jax.devices("cpu")[0])
    if device in ("gpu", "cuda") or device.startswith("cuda:"):
        index = int(device.split(":")[1]) if ":" in device else 0
        gpu = jax.devices("gpu")[index]
        jax.config.update("jax_default_device", gpu)
        return str(gpu)
    raise ValueError(f"Unsupported device {device!r} (use cpu, gpu, cuda or cuda:<i>)")


def _references_dir(cfg: dict[str, Any]) -> Path:
    from .data import paper_data_dir

    return paper_data_dir(cfg) / "sources" / "qml-benchmarks"


def _selection_block(kind: str) -> dict[str, str]:
    """Return the ``selection`` block of ``run_status.json``.

    The logging contract only accepts ``maximize`` / ``minimize``; runs that
    select nothing state it in ``name``.
    """
    if kind == "grid_search":
        return {
            "name": "mean_cv_accuracy",
            "direction": "maximize",
            "split": "train (StratifiedKFold(5), no shuffle)",
            "checkpoint": "final parameters at convergence; selected point retrained on the full train split",
        }
    if kind == "score_best":
        return {
            "name": "none in this run (authors' released best hyperparameters, chosen on their CV accuracy)",
            "direction": "maximize",
            "split": "released selection on train CV; this run scores the test split",
            "checkpoint": "final parameters at convergence",
        }
    return {
        "name": f"none ({kind} is an analysis run without model selection)",
        "direction": "maximize",
        "split": "n/a",
        "checkpoint": "n/a",
    }


def _write_table(run_dir: Path, name: str, table: pd.DataFrame) -> str:
    path = run_dir / name
    table.to_csv(path, index=False)
    logger.info("ARTIFACT_WRITTEN | path=%s | rows=%d", name, len(table))
    return name


# --------------------------------------------------------------------------- #
# grid_search                                                                  #
# --------------------------------------------------------------------------- #


def _run_grid_search(
    cfg: dict[str, Any], run_dir: Path, status: dict[str, Any]
) -> dict[str, Any]:
    from .data import load_dataset
    from .grid_search import run_grid_search, score_with_seeds, select_best
    from .hyperparameters import expand_grid, model_grid

    model_name = _require(cfg, "model.name")
    search_cfg = _require(cfg, "grid_search")
    grid_source = _require(search_cfg, "grid")
    if grid_source == "paper":
        grid = model_grid(model_name)
    elif isinstance(grid_source, dict):
        grid = {key: list(values) for key, values in grid_source.items()}
    else:
        raise ValueError(
            "grid_search.grid must be 'paper' or an explicit {name: [values]} grid"
        )
    fixed_params = dict(_require(search_cfg, "fixed_params"))
    search_seed = int(_require(search_cfg, "search_seed"))
    score_seeds = [int(s) for s in _require(search_cfg, "score_seeds")]

    data = load_dataset(cfg)
    status["dataset"] = data.identity()
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | dataset=%s | train=%s | test=%s",
        status["dataset"]["name"],
        data.X_train.shape,
        data.X_test.shape,
    )

    n_points = len(expand_grid(grid))
    logger.info(
        "TRAINING_STARTED | model=%s | grid=%s | grid_points=%d | folds=5 | fits=%d | fixed_params=%s",
        model_name,
        "paper" if grid_source == "paper" else "override",
        n_points,
        5 * n_points,
        fixed_params,
    )

    def record_progress(done: int, total: int) -> None:
        write_json(
            run_dir / "progress.json", {"current": done, "total": total, "unit": "fits"}
        )

    cv_results = run_grid_search(
        model_name,
        grid,
        data.X_train,
        data.y_train,
        search_seed,
        fixed_params,
        record_progress,
    )
    _write_table(run_dir, "cv_results.csv", cv_results)
    best_index, best_params = select_best(cv_results, grid)
    best_cv = float(cv_results.loc[best_index, "mean_test_accuracy"])
    write_json(
        run_dir / "best_hyperparameters.json",
        {"params": {k: repr(v) for k, v in best_params.items()}},
    )
    logger.info(
        "SELECTION_COMPLETED | point=%d | params=%s | mean_cv_accuracy=%.6f",
        best_index + 1,
        best_params,
        best_cv,
    )

    scores = score_with_seeds(
        model_name,
        best_params,
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
        score_seeds,
        fixed_params,
    )
    _write_table(run_dir, "scores.csv", scores)

    return {
        "experiment": "grid_search",
        "model": model_name,
        "dataset": data.name,
        "param": data.param,
        "grid": "paper" if grid_source == "paper" else "override",
        "fixed_params": {k: repr(v) for k, v in fixed_params.items()},
        "n_grid_points": n_points,
        "n_failed_grid_points": int(cv_results["mean_test_accuracy"].isna().sum()),
        "n_not_converged_cv_fits": int(cv_results["n_not_converged"].sum()),
        "best_params": {k: repr(v) for k, v in best_params.items()},
        "best_mean_cv_accuracy": best_cv,
        "test_acc_mean": float(scores["test_acc"].mean()),
        "test_acc_std": float(scores["test_acc"].std(ddof=1)),
        "train_acc_mean": float(scores["train_acc"].mean()),
        "n_score_seeds": len(score_seeds),
        "n_score_seeds_failed": int(scores["test_acc"].isna().sum()),
        "total_fit_seconds": float(
            cv_results["mean_fit_seconds"].sum() * 5 + scores["fit_seconds"].sum()
        ),
        "artifacts": ["cv_results.csv", "scores.csv", "best_hyperparameters.json"],
    }


# --------------------------------------------------------------------------- #
# score_best (port validation)                                                 #
# --------------------------------------------------------------------------- #


def _cast_like_grid(model_name: str, raw: dict[str, str]) -> dict[str, Any]:
    """Cast released best-hyperparameter strings to the types of the model grid."""
    from .hyperparameters import model_grid

    grid = model_grid(model_name)
    params: dict[str, Any] = {}
    for key, text in raw.items():
        if key not in grid:
            raise KeyError(
                f"Released hyperparameter {key!r} is not in the {model_name} grid"
            )
        example = grid[key][0]
        if isinstance(example, bool):
            params[key] = text == "True"
        elif isinstance(example, int):
            params[key] = int(float(text))
        elif isinstance(example, float):
            params[key] = float(text)
        elif isinstance(example, tuple):
            params[key] = ast.literal_eval(text)
        else:
            params[key] = text
    return params


def _run_score_best(
    cfg: dict[str, Any], run_dir: Path, status: dict[str, Any]
) -> dict[str, Any]:
    from .data import load_dataset
    from .grid_search import score_with_seeds
    from .reanalysis import released_file

    model_name = _require(cfg, "model.name")
    benchmark = _require(cfg, "score_best.released_benchmark")
    score_seeds = [int(s) for s in _require(cfg, "score_best.score_seeds")]
    fixed_params = dict(_require(cfg, "score_best.fixed_params"))
    results_root = _references_dir(cfg) / "paper" / "results"
    param = int(_require(cfg, "dataset.param"))

    best_path = released_file(
        results_root, benchmark, model_name, param, "_GridSearchCV-best-hyperparams.csv"
    )
    released_path = released_file(
        results_root,
        benchmark,
        model_name,
        param,
        "_GridSearchCV-best-hyperparams-results.csv",
    )
    best_table = pd.read_csv(best_path)
    raw = dict(
        zip(
            best_table["hyperparameter"].astype(str),
            best_table["best_value"].astype(str),
        )
    )
    params = _cast_like_grid(model_name, raw)
    released = pd.read_csv(released_path, index_col=0)

    data = load_dataset(cfg)
    status["dataset"] = data.identity()
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | dataset=%s | train=%s | test=%s",
        status["dataset"]["name"],
        data.X_train.shape,
        data.X_test.shape,
    )
    logger.info(
        "TRAINING_STARTED | model=%s | released_params=%s | seeds=%s | source=%s",
        model_name,
        params,
        score_seeds,
        best_path.name,
    )
    scores = score_with_seeds(
        model_name,
        params,
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
        score_seeds,
        fixed_params,
    )
    logger.info("TRAIN_RETRAINS_COMPLETED | seeds=%d", len(score_seeds))
    _write_table(run_dir, "scores.csv", scores)
    _write_table(run_dir, "released_scores.csv", released.reset_index(names="seed_row"))

    ours, theirs = scores["test_acc"].to_numpy(), released["test_acc"].to_numpy()
    return {
        "experiment": "score_best",
        "model": model_name,
        "dataset": data.name,
        "param": data.param,
        "released_benchmark": benchmark,
        "released_params": {k: repr(v) for k, v in params.items()},
        "fixed_params": {k: repr(v) for k, v in fixed_params.items()},
        "test_acc_mean": float(np.nanmean(ours)),
        "test_acc_std": float(np.nanstd(ours, ddof=1)),
        "released_test_acc_mean": float(theirs.mean()),
        "released_test_acc_std": float(theirs.std(ddof=1)),
        "difference_of_means": float(np.nanmean(ours) - theirs.mean()),
        "n_score_seeds_failed": int(np.isnan(ours).sum()),
        "artifacts": ["scores.csv", "released_scores.csv"],
    }


# --------------------------------------------------------------------------- #
# reanalysis (E0)                                                              #
# --------------------------------------------------------------------------- #


def _run_reanalysis(
    cfg: dict[str, Any], run_dir: Path, status: dict[str, Any]
) -> dict[str, Any]:
    from .reanalysis import (
        CORRELATION_SCRIPT_BENCHMARKS,
        accuracy_curves,
        family_models,
        grid_tables,
        load_released_scores,
        rank_family,
        summarize_correlations,
        summarize_ranks,
    )

    results_root = _references_dir(cfg) / "paper" / "results"
    extra_models = list(_require(cfg, "reanalysis.extra_models"))
    all_models = sorted(
        set(
            family_models("qnn", True)
            + family_models("kernel", True)
            + family_models("cnn", False)
            + extra_models
        )
    )
    scores = load_released_scores(results_root, all_models)
    status["dataset"] = {
        "name": "qml-benchmarks released results",
        "splits": {
            "released_rows": int(len(scores)),
            "cells": int(scores.groupby(["benchmark", "param", "model"]).ngroups),
        },
        "preprocessing": "none (authors' per-seed train/test accuracies)",
        "subset": "files selected as in paper/plots/plot_rankings_total.py (two_curves- prefix only)",
        "source": str(results_root.relative_to(_REPO_ROOT))
        if results_root.is_relative_to(_REPO_ROOT)
        else str(results_root),
    }
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | released_rows=%d | cells=%s",
        len(scores),
        status["dataset"]["splits"]["cells"],
    )
    _write_table(run_dir, "released_scores.csv", scores)

    logger.info("EVALUATION_STARTED | analysis=rankings")
    ranking_summary = {}
    rank_tables = []
    for family in ("qnn", "kernel", "cnn"):
        for with_separable in (False, True) if family != "cnn" else (False,):
            models = family_models(family, with_separable)
            for restrict in (False, True):
                ranked = rank_family(scores, models, restrict)
                summary = summarize_ranks(ranked, models)
                label = f"{family}{'_with_separable' if with_separable else ''}{'_complete_cells' if restrict else ''}"
                summary.insert(0, "ranking", label)
                rank_tables.append(summary)
                ranked.insert(0, "ranking", label)
                ranked.to_csv(run_dir / f"ranks_{label}.csv", index=False)
                ranking_summary[label] = {
                    "order": summary["model"].tolist(),
                    "mean_rank_pct": [
                        round(v, 6) for v in summary["mean_rank_pct"].tolist()
                    ],
                    "n_datasets": summary["n_datasets"].tolist(),
                    "rank_1_counts": summary["rank_1"].tolist(),
                    "first": summary["model"].iloc[0],
                }
    _write_table(run_dir, "rank_summary.csv", pd.concat(rank_tables, ignore_index=True))
    logger.info(
        "EVALUATION_COMPLETED | analysis=rankings | first_by_family=%s",
        {k: v["first"] for k, v in ranking_summary.items()},
    )

    logger.info("EVALUATION_STARTED | analysis=accuracy_curves_and_ablations")
    curves = accuracy_curves(scores)
    _write_table(run_dir, "accuracy_curves.csv", curves)
    ablation_models = {
        "entanglement_ablation_fig13": [
            "DressedQuantumCircuitClassifier",
            "DressedQuantumCircuitClassifierSeparable",
            "DataReuploadingClassifier",
            "DataReuploadingClassifierSeparable",
            "QuantumBoltzmannMachine",
            "QuantumBoltzmannMachineSeparable",
        ],
        "drc_ablation_fig15": [
            "DataReuploadingClassifier",
            "DataReuploadingClassifierNoScaling",
            "DataReuploadingClassifierNoCost",
            "DataReuploadingClassifierNoTrainableEmbedding",
        ],
    }
    ablation_summary = {}
    for name, models in ablation_models.items():
        table = curves[
            (curves["benchmark"] == "mnist_pca")
            & curves["model"].isin(models)
            & curves["param"].between(2, 9)
        ]
        _write_table(run_dir, f"{name}.csv", table)
        pivot = table.pivot(index="param", columns="model", values="test_mean")
        ablation_summary[name] = {
            model: [
                round(v, 4) if np.isfinite(v) else None for v in pivot[model].tolist()
            ]
            for model in pivot.columns
        }
        ablation_summary[name]["param"] = pivot.index.tolist()
    logger.info("EVALUATION_COMPLETED | analysis=accuracy_curves_and_ablations")

    logger.info("EVALUATION_STARTED | analysis=grid_search_tables")
    grids = grid_tables(results_root, all_models)
    _write_table(run_dir, "grid_search_summary.csv", grids)
    correlations_script = summarize_correlations(grids, CORRELATION_SCRIPT_BENCHMARKS)
    correlations_all = summarize_correlations(grids, tuple(grids["benchmark"].unique()))
    _write_table(
        run_dir,
        "hyperparameter_correlations_released_script_scope.csv",
        correlations_script,
    )
    _write_table(
        run_dir, "hyperparameter_correlations_all_benchmarks.csv", correlations_all
    )
    logger.info(
        "EVALUATION_COMPLETED | analysis=grid_search_tables | grid_files=%d",
        grids.groupby(["benchmark", "param", "model"]).ngroups,
    )

    return {
        "experiment": "reanalysis",
        "released_rows": int(len(scores)),
        "rankings": ranking_summary,
        "ablations_mnist_pca_test_mean": ablation_summary,
        "correlations_released_scope": {
            row.hyperparameter: {
                "spearman_mean": round(row.spearman_mean, 4),
                "n": int(row.n),
            }
            for row in correlations_script.itertuples()
        },
        "artifacts": sorted(p.name for p in run_dir.glob("*.csv")),
    }


# --------------------------------------------------------------------------- #
# positivity_bias (E12)                                                        #
# --------------------------------------------------------------------------- #


def _run_positivity_bias(
    cfg: dict[str, Any], run_dir: Path, status: dict[str, Any]
) -> dict[str, Any]:
    from .positivity_bias import simulate_published_means

    sim = _require(cfg, "positivity_bias")
    seeds = list(
        range(int(sim["first_seed"]), int(sim["first_seed"]) + int(sim["repetitions"]))
    )
    status["dataset"] = {
        "name": "simulated performances",
        "splits": {"repetitions": len(seeds)},
        "preprocessing": "none",
        "subset": "none",
    }
    write_json(run_dir / "run_status.json", status)
    logger.info("DATASET_READY | repetitions=%d", len(seeds))
    logger.info("EVALUATION_STARTED | analysis=positivity_bias")
    rows = []
    for seed in seeds:
        quantum, classical = simulate_published_means(
            seed,
            int(sim["n_researchers"]),
            int(sim["n_designs"]),
            float(sim["quantum_mean"]),
            float(sim["quantum_std"]),
            float(sim["classical_mean"]),
            float(sim["classical_std"]),
        )
        rows.append(
            {
                "seed": seed,
                "published_quantum_mean": quantum,
                "published_classical_mean": classical,
            }
        )
    table = pd.DataFrame(rows)
    _write_table(run_dir, "positivity_bias.csv", table)
    metrics = {
        "experiment": "positivity_bias",
        "settings": sim,
        "published_quantum_mean": float(table["published_quantum_mean"].mean()),
        "published_quantum_std": float(table["published_quantum_mean"].std(ddof=1)),
        "published_classical_mean": float(table["published_classical_mean"].mean()),
        "published_classical_std": float(table["published_classical_mean"].std(ddof=1)),
        "fraction_quantum_above_classical": float(
            (table["published_quantum_mean"] > table["published_classical_mean"]).mean()
        ),
        "paper_values": {"quantum": 0.74, "classical": 0.65},
        "artifacts": ["positivity_bias.csv"],
    }
    logger.info(
        "EVALUATION_COMPLETED | analysis=positivity_bias | quantum=%.4f+-%.4f | classical=%.4f+-%.4f",
        metrics["published_quantum_mean"],
        metrics["published_quantum_std"],
        metrics["published_classical_mean"],
        metrics["published_classical_std"],
    )
    return metrics


_EXPERIMENTS = {
    "grid_search": _run_grid_search,
    "score_best": _run_score_best,
    "reanalysis": _run_reanalysis,
    "positivity_bias": _run_positivity_bias,
}


def _candidate(cfg: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind in ("grid_search", "score_best"):
        return {
            "model": cfg["model"]["name"],
            "dataset": cfg["dataset"]["name"],
            "param": cfg["dataset"]["param"],
        }
    return {"experiment": kind}


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Run one configured experiment and write its artifacts to ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration produced by the shared runtime. Requires
        ``experiment.kind`` and the block of that kind (see module docstring
        and ``configs/``).
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Raises
    ------
    KeyError
        If a required config key is missing or the experiment kind is unknown.
    """
    run_dir = Path(run_dir)
    level = _LOG_LEVELS[str(cfg.get("logging", {}).get("level", "info")).lower()]
    configure_logging(run_dir / "run.log", level=level)
    kind = _require(cfg, "experiment.kind")
    if kind not in _EXPERIMENTS:
        raise KeyError(
            f"Unknown experiment.kind {kind!r}; known: {sorted(_EXPERIMENTS)}"
        )

    jax_device = _configure_device(str(_require(cfg, "device")))
    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        write_json(config_path, cfg)
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": cfg.get("run_id") or run_dir.name,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": _candidate(cfg, kind),
        "seed": cfg.get("seed"),
        "repetition": cfg.get("repetition"),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(_REPO_ROOT),
        "dataset": {
            "name": "pending",
            "splits": {},
            "preprocessing": "pending",
            "subset": "pending",
        },
        "selection": _selection_block(kind),
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(config_path),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
        "experiment": kind,
        "stage": cfg.get("experiment", {}).get("stage"),
        "python": platform.python_version(),
        "jax_device": jax_device,
    }
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | experiment=%s | candidate=%s | seed=%s",
        status["run_id"],
        kind,
        json.dumps(status["candidate"]),
        status["seed"],
    )
    try:
        metrics = _EXPERIMENTS[kind](cfg, run_dir, status)
        write_json(run_dir / "metrics.json", metrics)
        logger.info("METRICS_WRITTEN | path=metrics.json")
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(run_dir / "metrics.json"),
        )
        write_json(run_dir / "run_status.json", status)
        logger.info("RUN_COMPLETED | run_id=%s", status["run_id"])
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise
