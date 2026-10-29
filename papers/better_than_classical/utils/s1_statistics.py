"""Stage S1 statistics on the re-analysis of the authors' released results.

Reads a ``reanalysis`` run directory (``released_scores.csv``,
``grid_search_summary.csv``) and, optionally, a ``positivity_bias`` run
directory, and writes ``s1_statistics.json`` plus CSV tables to the
re-analysis run directory and to ``results/``:

- ``s1_paired_vs_baseline.csv``: every model against its family's classical
  baseline (MLP, SVC, CNN) and against the family's separable model, paired
  over the datasets where both have results (mean test accuracy over the 5
  seeds per dataset): number of datasets, median / mean difference, fraction
  of datasets where the reference is at least as good, Wilcoxon signed-rank
  p-value (two-sided; fewer than 6 pairs: not tested) and a seeded bootstrap
  95% CI of the mean difference.
- ``s1_ablation_differences.csv``: Fig. 13 / Fig. 15 differences per d with
  the standard error from the 5 seeds, and how many d exceed 2 SE.
- ``s1_grid_ranges.csv``: Fig. 7 ranges (min / max CV accuracy over the grid).

Usage
-----
python utils/s1_statistics.py outdir/run_<reanalysis> [--positivity-run outdir/run_<pb>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PAPER_DIR = Path(__file__).resolve().parents[1]
BASELINES = {
    "qnn": "MLPClassifier",
    "kernel": "SVC",
    "cnn": "ConvolutionalNeuralNetwork",
}
FAMILY_MODELS = {
    "qnn": [
        "DressedQuantumCircuitClassifier",
        "DataReuploadingClassifier",
        "QuantumBoltzmannMachine",
        "IQPVariationalClassifier",
        "QuantumMetricLearner",
        "CircuitCentricClassifier",
        "TreeTensorClassifier",
        "SeparableVariationalClassifier",
    ],
    "kernel": [
        "ProjectedQuantumKernel",
        "IQPKernelClassifier",
        "QuantumKitchenSinks",
        "SeparableKernelClassifier",
    ],
    "cnn": ["QuanvolutionalNeuralNetwork", "WeiNet"],
}
SEPARABLE = {
    "qnn": "SeparableVariationalClassifier",
    "kernel": "SeparableKernelClassifier",
}
MIN_PAIRS_FOR_TEST = 6
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 0


def paired_comparison(per_dataset: pd.DataFrame, reference: str, model: str) -> dict:
    """Compare ``reference`` with ``model`` over datasets where both have results.

    The difference is ``reference - model`` in mean test accuracy, so a
    positive value means the reference is better.
    """
    pivot = per_dataset.pivot_table(
        index=["benchmark", "param"], columns="model", values="test_acc"
    )
    if reference not in pivot or model not in pivot:
        return {"n_datasets": 0}
    pairs = pivot[[reference, model]].dropna()
    diff = (pairs[reference] - pairs[model]).to_numpy()
    row = {
        "n_datasets": len(diff),
        "median_diff": float(np.median(diff)),
        "mean_diff": float(np.mean(diff)),
        "frac_reference_at_least_as_good": float(np.mean(diff >= 0)),
    }
    if len(diff) >= MIN_PAIRS_FOR_TEST and np.any(diff != 0):
        row["wilcoxon_p"] = float(wilcoxon(diff).pvalue)
    else:
        row["wilcoxon_p"] = np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = rng.choice(diff, size=(BOOTSTRAP_SAMPLES, len(diff)), replace=True).mean(
        axis=1
    )
    row["mean_diff_ci_low"], row["mean_diff_ci_high"] = (
        float(v) for v in np.percentile(boot, [2.5, 97.5])
    )
    return row


def ablation_differences(scores: pd.DataFrame, pairs: dict[str, tuple[str, str]]):
    """Per-d differences (original - variant) on mnist pca with seed SEs."""
    rows = []
    mnist = scores[(scores["benchmark"] == "mnist_pca") & scores["param"].between(2, 9)]
    stats = mnist.groupby(["model", "param"])["test_acc"].agg(["mean", "std", "count"])
    for name, (original, variant) in pairs.items():
        for d in range(2, 10):
            if (original, d) not in stats.index or (variant, d) not in stats.index:
                continue
            a, b = stats.loc[(original, d)], stats.loc[(variant, d)]
            se = float(np.sqrt(a["std"] ** 2 / a["count"] + b["std"] ** 2 / b["count"]))
            diff = float(a["mean"] - b["mean"])
            rows.append(
                {
                    "comparison": name,
                    "original": original,
                    "variant": variant,
                    "d": d,
                    "original_mean": float(a["mean"]),
                    "variant_mean": float(b["mean"]),
                    "diff": diff,
                    "se": se,
                    "exceeds_2se": bool(abs(diff) > 2 * se),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reanalysis_run", type=Path)
    parser.add_argument("--positivity-run", type=Path, default=None)
    args = parser.parse_args()
    run = args.reanalysis_run
    results_dir = PAPER_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    scores = pd.read_csv(run / "released_scores.csv")
    per_dataset = scores.groupby(["benchmark", "param", "model"], as_index=False)[
        "test_acc"
    ].mean()

    paired_rows = []
    for family, models in FAMILY_MODELS.items():
        references = [BASELINES[family]]
        if family in SEPARABLE:
            references.append(SEPARABLE[family])
        for reference in references:
            for model in models:
                if model == reference:
                    continue
                row = paired_comparison(per_dataset, reference, model)
                paired_rows.append(
                    {"family": family, "reference": reference, "model": model, **row}
                )
    paired = pd.DataFrame(paired_rows)

    ablations = ablation_differences(
        scores,
        {
            "fig13_dqc": (
                "DressedQuantumCircuitClassifier",
                "DressedQuantumCircuitClassifierSeparable",
            ),
            "fig13_drc": (
                "DataReuploadingClassifier",
                "DataReuploadingClassifierSeparable",
            ),
            "fig13_qbm": (
                "QuantumBoltzmannMachine",
                "QuantumBoltzmannMachineSeparable",
            ),
            "fig15_nocost": (
                "DataReuploadingClassifier",
                "DataReuploadingClassifierNoCost",
            ),
            "fig15_noscaling": (
                "DataReuploadingClassifier",
                "DataReuploadingClassifierNoScaling",
            ),
            "fig15_notrainableembedding": (
                "DataReuploadingClassifier",
                "DataReuploadingClassifierNoTrainableEmbedding",
            ),
        },
    )
    ablation_summary = (
        ablations[ablations["d"] >= 5]
        .groupby("comparison")
        .agg(
            mean_diff_d5_9=("diff", "mean"),
            n_d=("d", "count"),
            n_exceeding_2se=("exceeds_2se", "sum"),
        )
        .reset_index()
    )

    grids = pd.read_csv(run / "grid_search_summary.csv")
    grid_ranges = (
        grids[
            grids["model"].isin(
                [
                    "MLPClassifier",
                    "DataReuploadingClassifier",
                    "CircuitCentricClassifier",
                    "IQPVariationalClassifier",
                ]
            )
            & grids["benchmark"].isin(
                ["linearly_separable", "mnist_pca", "hidden_manifold_diff"]
            )
        ]
        .drop_duplicates(["benchmark", "param", "model"])
        .groupby(["benchmark", "model"])
        .agg(
            lowest_min_cv=("min_cv_accuracy", "min"),
            median_min_cv=("min_cv_accuracy", "median"),
            median_max_cv=("max_cv_accuracy", "median"),
            highest_max_cv=("max_cv_accuracy", "max"),
        )
        .reset_index()
    )

    summary = {
        "source_run": run.name,
        "paired_vs_reference": paired.round(6).to_dict(orient="records"),
        "ablation_summary_d5_to_9": ablation_summary.round(6).to_dict(orient="records"),
    }
    if args.positivity_run is not None:
        summary["positivity_bias"] = json.loads(
            (args.positivity_run / "metrics.json").read_text()
        )
        summary["positivity_bias"]["source_run"] = args.positivity_run.name

    for directory in (run, results_dir):
        paired.to_csv(directory / "s1_paired_vs_baseline.csv", index=False)
        ablations.to_csv(directory / "s1_ablation_differences.csv", index=False)
        grid_ranges.to_csv(directory / "s1_grid_ranges.csv", index=False)
        (directory / "s1_statistics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    pd.set_option("display.width", 220)
    print(paired.round(4).to_string(index=False))
    print(ablation_summary.round(4).to_string(index=False))
    print(grid_ranges.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
