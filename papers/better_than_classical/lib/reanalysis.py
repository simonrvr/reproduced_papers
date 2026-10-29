"""Re-analysis of the authors' released results (experiment E0).

Reads ``paper/results/`` of the qml-benchmarks snapshot and recomputes the
aggregates behind the paper's figures with independent code:

- per-family rank statistics (Fig. 10; Fig. 24 with separable models), using
  the released rule: mean test accuracy over the 5 retrain seeds per
  (dataset, model), ``rank(method="min", ascending=False)`` within the family
  per dataset, models ordered by mean ``rank(pct=True)``;
- the same ranking restricted to datasets on which every family member has a
  result (added robustness check);
- accuracy curves per benchmark (Figs. 11, 12, 22, 23);
- the entanglement-removal and data re-uploading ablations (Figs. 13, 15);
- grid-search accuracy ranges (Fig. 7) and hyperparameter-accuracy
  correlations (Fig. 8; Spearman as in the released script and Pearson as
  stated in the paper text).

This checks the paper's analysis of its own numbers. It does not retrain any
model and is not independent evidence for the claims.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .models import FAMILIES, SEPARABLE_BY_FAMILY

logger = logging.getLogger(__name__)

# benchmark -> (results folder, filename stem before the parameter, stem after it)
# Mirrors the `datasets` table of paper/plots/plot_rankings_total.py; the
# parameter is an integer except for image benchmarks ("16x16", "16_x_16").
RELEASED_BENCHMARKS = {
    "mnist_pca": ("mnist_pca", "mnist_3-5_", "d"),
    "mnist_pca_minus": ("mnist_pca-", "mnist_3-5_", "d-250"),
    "mnist_cg": ("mnist_cg", "mnist_pixels_3-5_", ""),
    "bars_and_stripes": ("bars_and_stripes", "bars_and_stripes_", "_0.5noise"),
    "linearly_separable": ("linearly_separable", "linearly_separable_", "d"),
    "hidden_manifold": ("hidden_manifold", "hidden_manifold-6manifold-", "d"),
    "hidden_manifold_diff": (
        "hidden_manifold_diff",
        "hidden_manifold-10d-",
        "manifold",
    ),
    "two_curves": ("two_curves", "two_curves-5degree-0.1offset-", "d"),
    "two_curves_diff": ("two_curves_diff", "two_curves-10d-", "degree"),
    "hyperplanes_diff": ("hyperplanes_diff", "hyperplanes-10d-from3d-", "n"),
}
_RESULTS_SUFFIX = "_GridSearchCV-best-hyperparams-results.csv"
_GRID_SUFFIX = "_GridSearchCV.csv"
_IMAGE_BENCHMARKS = {"mnist_cg": "{n}x{n}", "bars_and_stripes": "{n}_x_{n}"}
# The released correlation script (Fig. 8) points "TWO CURVES" at the
# two_curves_diff folder with the two_curves prefix, so it reads no file; it
# also omits the image benchmarks.
CORRELATION_SCRIPT_BENCHMARKS = (
    "mnist_pca",
    "mnist_pca_minus",
    "linearly_separable",
    "hidden_manifold",
    "hidden_manifold_diff",
    "two_curves_diff",
    "hyperplanes_diff",
)


def _param_token(benchmark: str, n: int) -> str:
    return _IMAGE_BENCHMARKS.get(benchmark, "{n}").format(n=n)


def released_file(
    results_root: Path, benchmark: str, model: str, n: int, suffix: str
) -> Path:
    """Return the path of one released result file (it may not exist)."""
    folder, stem, after = RELEASED_BENCHMARKS[benchmark]
    name = f"{model}_{stem}{_param_token(benchmark, n)}{after}{suffix}"
    return results_root / folder / model / name


def load_released_scores(
    results_root: Path, models: list[str], params=range(2, 33)
) -> pd.DataFrame:
    """Load every released 5-seed train/test accuracy file for the given models.

    Returns
    -------
    pandas.DataFrame
        Columns ``benchmark, param, model, seed_row, train_acc, test_acc``.
    """
    frames = []
    for benchmark in RELEASED_BENCHMARKS:
        for model in models:
            for n in params:
                path = released_file(results_root, benchmark, model, n, _RESULTS_SUFFIX)
                if not path.exists():
                    continue
                table = pd.read_csv(path, index_col=0)
                frames.append(
                    pd.DataFrame(
                        {
                            "benchmark": benchmark,
                            "param": n,
                            "model": model,
                            "seed_row": table.index.to_numpy(),
                            "train_acc": table["train_acc"].to_numpy(),
                            "test_acc": table["test_acc"].to_numpy(),
                        }
                    )
                )
    if not frames:
        raise FileNotFoundError(f"No released results found under {results_root}")
    return pd.concat(frames, ignore_index=True)


def rank_family(
    scores: pd.DataFrame, family_models: list[str], restrict_to_complete: bool
) -> pd.DataFrame:
    """Rank the models of one family per dataset, as in plot_rankings_total.py.

    Parameters
    ----------
    scores : pandas.DataFrame
        Output of :func:`load_released_scores` (or our own runs, same columns).
    family_models : list of str
        Models competing in this family.
    restrict_to_complete : bool
        If True, keep only datasets on which every model of the family has a
        result (added robustness check; the paper ranks among available models).

    Returns
    -------
    pandas.DataFrame
        One row per (benchmark, param, model) with ``mean_test_acc``, ``rank``
        and ``rank_pct``.
    """
    family = scores[scores["model"].isin(family_models)]
    means = (
        family.groupby(["benchmark", "param", "model"], as_index=False)["test_acc"]
        .mean()
        .rename(columns={"test_acc": "mean_test_acc"})
    )
    if restrict_to_complete:
        counts = means.groupby(["benchmark", "param"])["model"].transform("nunique")
        means = means[counts == len(family_models)].copy()
    grouped = means.groupby(["benchmark", "param"])["mean_test_acc"]
    means["rank"] = grouped.rank(method="min", ascending=False)
    means["rank_pct"] = grouped.rank(method="min", ascending=False, pct=True)
    return means


def summarize_ranks(ranked: pd.DataFrame, family_models: list[str]) -> pd.DataFrame:
    """Count how often each model takes each rank and order by mean ``rank_pct``.

    Returns
    -------
    pandas.DataFrame
        One row per model with ``n_datasets``, ``rank_<k>`` counts,
        ``mean_rank_pct`` and ``order`` (1 = best, the paper's top row).
    """
    rows = []
    for model in family_models:
        model_ranks = ranked[ranked["model"] == model]
        row = {"model": model, "n_datasets": len(model_ranks)}
        for k in range(1, len(family_models) + 1):
            row[f"rank_{k}"] = int((model_ranks["rank"] == k).sum())
        row["mean_rank_pct"] = (
            float(model_ranks["rank_pct"].mean()) if len(model_ranks) else np.nan
        )
        rows.append(row)
    summary = (
        pd.DataFrame(rows)
        .sort_values("mean_rank_pct", na_position="last")
        .reset_index(drop=True)
    )
    summary["order"] = np.arange(1, len(summary) + 1)
    return summary


def accuracy_curves(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, min and max of train/test accuracy over the 5 seeds per cell."""
    return (
        scores.groupby(["benchmark", "model", "param"])
        .agg(
            test_mean=("test_acc", "mean"),
            test_std=("test_acc", "std"),
            test_min=("test_acc", "min"),
            test_max=("test_acc", "max"),
            train_mean=("train_acc", "mean"),
            train_std=("train_acc", "std"),
            n_seeds=("test_acc", "count"),
        )
        .reset_index()
    )


def _hyperparameter_columns(table: pd.DataFrame) -> list[str]:
    ignore = {"param_batch_size", "param_max_vmap"}
    return [c for c in table.columns if c.startswith("param_") and c not in ignore]


def _rank_column(values: pd.Series) -> pd.Series:
    """Average ranks; strings rank lexicographically, as numpy sorting does."""
    if pd.api.types.is_numeric_dtype(values):
        return values.rank(method="average")
    return values.astype(str).rank(method="average")


def grid_tables(
    results_root: Path, models: list[str], params=range(2, 33)
) -> pd.DataFrame:
    """Per-grid-search summaries: accuracy range and hyperparameter correlations.

    Returns
    -------
    pandas.DataFrame
        Long table with one row per (benchmark, param, model, hyperparameter):
        ``min_cv_accuracy``, ``max_cv_accuracy`` (Fig. 7, from
        ``mean_test_accuracy`` over the grid), ``spearman`` (released Fig. 8
        computation: rows with NaN dropped, average ranks, strings ranked
        lexicographically) and ``pearson`` (paper text; numeric
        hyperparameters only).
    """
    rows = []
    for benchmark in RELEASED_BENCHMARKS:
        for model in models:
            for n in params:
                path = released_file(results_root, benchmark, model, n, _GRID_SUFFIX)
                if not path.exists() or path.stat().st_size == 0:
                    continue
                table = pd.read_csv(path)
                # Single-metric searches name the column mean_test_score (the
                # released plot_hyperparameter_accuracy_ranges.py reads both).
                if "mean_test_accuracy" not in table.columns:
                    table = table.rename(
                        columns={"mean_test_score": "mean_test_accuracy"}
                    )
                accuracy = table["mean_test_accuracy"]
                base = {
                    "benchmark": benchmark,
                    "param": n,
                    "model": model,
                    "min_cv_accuracy": float(accuracy.min()),
                    "max_cv_accuracy": float(accuracy.max()),
                    "n_grid_points": len(table),
                    "n_failed_points": int(accuracy.isna().sum()),
                }
                hp_columns = _hyperparameter_columns(table)
                complete = table[hp_columns + ["mean_test_accuracy"]].dropna()
                if not hp_columns:
                    rows.append(
                        {
                            **base,
                            "hyperparameter": None,
                            "spearman": np.nan,
                            "pearson": np.nan,
                        }
                    )
                for column in hp_columns:
                    spearman = pearson = np.nan
                    if len(complete) > 2 and complete[column].nunique() > 1:
                        spearman = float(
                            np.corrcoef(
                                _rank_column(complete[column]),
                                complete["mean_test_accuracy"].rank(method="average"),
                            )[0, 1]
                        )
                        if pd.api.types.is_numeric_dtype(complete[column]):
                            pearson = float(
                                np.corrcoef(
                                    complete[column], complete["mean_test_accuracy"]
                                )[0, 1]
                            )
                    rows.append(
                        {
                            **base,
                            "hyperparameter": column.removeprefix("param_"),
                            "spearman": spearman,
                            "pearson": pearson,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_correlations(
    grid_summary: pd.DataFrame, benchmarks: tuple[str, ...], max_param: int = 20
) -> pd.DataFrame:
    """Mean and std of correlations per hyperparameter (the Fig. 8 bars)."""
    subset = grid_summary[
        grid_summary["benchmark"].isin(benchmarks)
        & (grid_summary["param"] <= max_param)
        & grid_summary["hyperparameter"].notna()
    ]
    return (
        subset.groupby("hyperparameter")
        .agg(
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            n=("spearman", "count"),
        )
        .reset_index()
        .sort_values("spearman_mean")
    )


def family_models(family: str, with_separable: bool) -> list[str]:
    """Return the models of a ranking family, optionally with its separable model."""
    models = list(FAMILIES[family])
    if with_separable and family in SEPARABLE_BY_FAMILY:
        models.append(SEPARABLE_BY_FAMILY[family])
    return models
