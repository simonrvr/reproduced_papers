"""Stage S1 figures: re-analysis of the authors' released results + Fig. 3.

Reads a ``reanalysis`` run directory (after ``utils/s1_statistics.py``) and a
``positivity_bias`` run directory and writes PNGs to the re-analysis run
directory and to ``results/``:

- ``s1_rankings.png``: Fig. 10 / Fig. 24 analogue, rank counts per family
  with separable models.
- ``s1_entanglement_and_drc_ablation.png``: Fig. 13 and Fig. 15 analogues
  (mean test accuracy over the 5 released seeds, band = +-1 SE).
- ``s1_paired_differences.png``: reference minus model, mean over datasets
  with bootstrap 95% CI (added; the paper reports no test).
- ``s1_hyperparameter_correlations.png``: Fig. 8 analogue.
- ``s1_positivity_bias.png``: Fig. 3 analogue over 1000 seeds.

Every figure is labelled with its result scope.

Usage
-----
python utils/plot_s1_reanalysis.py outdir/run_<reanalysis> outdir/run_<positivity_bias>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
# Reference categorical palette, slots 1-4 in fixed order (dataviz palette.md).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
# Diverging blue <-> gray <-> red (palette.md), for rank 1 (best) .. last.
DIVERGING_ENDS = ("#2a78d6", "#f0efec", "#e34948")
SCOPE_REANALYSIS = (
    "Scope: re-analysis of the authors' released results (not independent evidence)"
)
SCOPE_SIMULATION = "Scope: simulation of the paper's thought experiment (1000 seeds)"

SHORT = {
    "MLPClassifier": "MLP",
    "DressedQuantumCircuitClassifier": "DressedQC",
    "DataReuploadingClassifier": "DataReuploading",
    "QuantumBoltzmannMachine": "QBM",
    "SeparableVariationalClassifier": "SeparableVariational",
    "IQPVariationalClassifier": "IQPVariational",
    "QuantumMetricLearner": "MetricLearner",
    "CircuitCentricClassifier": "CircuitCentric",
    "TreeTensorClassifier": "TreeTensor",
    "SVC": "SVC",
    "SeparableKernelClassifier": "SeparableKernel",
    "ProjectedQuantumKernel": "ProjectedKernel",
    "IQPKernelClassifier": "IQPKernel",
    "QuantumKitchenSinks": "KitchenSinks",
    "ConvolutionalNeuralNetwork": "CNN",
    "QuanvolutionalNeuralNetwork": "Quanvolutional",
    "WeiNet": "WeiNet",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name, run_dir):
    results = PAPER_DIR / "results"
    results.mkdir(exist_ok=True)
    for directory in (run_dir, results):
        fig.savefig(directory / name, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _diverging(n):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rank", DIVERGING_ENDS)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_rankings(run_dir: Path):
    summary = pd.read_csv(run_dir / "rank_summary.csv")
    panels = [
        ("qnn_with_separable", "QNN family + SeparableVariational"),
        ("kernel_with_separable", "Kernel family + SeparableKernel"),
        ("cnn", "CNN family"),
    ]
    heights = [9, 5, 3]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 7.5),
        gridspec_kw={"height_ratios": heights},
        facecolor=SURFACE,
    )
    for ax, (label, title) in zip(axes, panels):
        table = summary[summary["ranking"] == label].sort_values(
            "order", ascending=False
        )
        rank_cols = [
            c for c in table.columns if c.startswith("rank_") and c != "rank_pct"
        ]
        rank_cols = sorted(rank_cols, key=lambda c: int(c.split("_")[1]))
        rank_cols = [c for c in rank_cols if table[c].notna().any()]
        colors = _diverging(len(rank_cols))
        left = np.zeros(len(table))
        names = [SHORT.get(m, m) for m in table["model"]]
        for color, col in zip(colors, rank_cols):
            values = table[col].fillna(0).to_numpy()
            ax.barh(
                names,
                values,
                left=left,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.5,
                label=col.replace("rank_", "rank "),
            )
            left += values
        for y, (total, pct) in enumerate(zip(left, table["mean_rank_pct"])):
            ax.text(
                total + 0.015 * left.max(),
                y,
                f"mean norm. rank {pct:.2f}",
                va="center",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.set_xlim(0, left.max() * 1.35)
    axes[-1].set_xlabel(
        "number of datasets with each rank (blue = 1st ... red = last)",
        color=TEXT_SECONDARY,
    )
    axes[0].legend(
        ncol=9, fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08)
    )
    fig.suptitle(
        "Rankings recomputed from released results (paper Figs. 10, 24)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=TEXT,
    )
    fig.text(
        0.01,
        -0.01,
        SCOPE_REANALYSIS + ". Rows ordered by mean normalised rank (lower is better).",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout()
    _save(fig, "s1_rankings.png", run_dir)


def plot_ablations(run_dir: Path):
    curves = pd.read_csv(run_dir / "accuracy_curves.csv")
    mnist = curves[(curves["benchmark"] == "mnist_pca") & curves["param"].between(2, 9)]
    panels = [
        (
            "Fig. 13: DressedQC",
            [
                "DressedQuantumCircuitClassifier",
                "DressedQuantumCircuitClassifierSeparable",
            ],
        ),
        (
            "Fig. 13: DataReuploading",
            ["DataReuploadingClassifier", "DataReuploadingClassifierSeparable"],
        ),
        (
            "Fig. 13: QBM",
            ["QuantumBoltzmannMachine", "QuantumBoltzmannMachineSeparable"],
        ),
        (
            "Fig. 15: DataReuploading ablation",
            [
                "DataReuploadingClassifier",
                "DataReuploadingClassifierNoCost",
                "DataReuploadingClassifierNoScaling",
                "DataReuploadingClassifierNoTrainableEmbedding",
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True, facecolor=SURFACE)
    for ax, (title, models) in zip(axes, panels):
        for color, model in zip(SERIES, models):
            data = mnist[mnist["model"] == model].sort_values("param")
            se = data["test_std"] / np.sqrt(data["n_seeds"])
            label = model.replace("Classifier", "").replace(
                "DressedQuantumCircuit", "DressedQC"
            )
            label = label.replace("QuantumBoltzmannMachine", "QBM")
            if title.startswith("Fig. 15"):
                label = (
                    label.replace("DataReuploading", "") or "DataReuploading (original)"
                )
            ax.fill_between(
                data["param"],
                data["test_mean"] - se,
                data["test_mean"] + se,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(
                data["param"],
                data["test_mean"],
                color=color,
                linewidth=2,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                label=label,
            )
        _style(ax)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel("number of features d (mnist pca)", color=TEXT_SECONDARY)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("test accuracy (mean of 5 seeds)", color=TEXT_SECONDARY)
    fig.text(
        0.01,
        -0.03,
        SCOPE_REANALYSIS + ". Bands: +-1 standard error over the 5 retrain seeds.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout()
    _save(fig, "s1_entanglement_and_drc_ablation.png", run_dir)


def plot_paired(run_dir: Path):
    paired = pd.read_csv(run_dir / "s1_paired_vs_baseline.csv")
    paired = paired[paired["n_datasets"] > 0].copy()
    paired["label"] = [
        f"{SHORT.get(r, r)} - {SHORT.get(m, m)}  (n={n})"
        for r, m, n in zip(paired["reference"], paired["model"], paired["n_datasets"])
    ]
    fig, ax = plt.subplots(figsize=(8, 7.5), facecolor=SURFACE)
    y = np.arange(len(paired))[::-1]
    is_baseline = paired["reference"].isin(
        ["MLPClassifier", "SVC", "ConvolutionalNeuralNetwork"]
    )
    for mask, color, legend in (
        (is_baseline, SERIES[0], "reference = classical baseline"),
        (~is_baseline, SERIES[1], "reference = separable (no-entanglement) model"),
    ):
        sub = paired[mask]
        ys = y[mask.to_numpy()]
        ax.errorbar(
            sub["mean_diff"],
            ys,
            xerr=[
                sub["mean_diff"] - sub["mean_diff_ci_low"],
                sub["mean_diff_ci_high"] - sub["mean_diff"],
            ],
            fmt="o",
            color=color,
            markersize=6,
            markeredgecolor=SURFACE,
            elinewidth=2,
            capsize=0,
            label=legend,
        )
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.set_yticks(y, paired["label"], fontsize=8)
    _style(ax)
    ax.set_xlabel(
        "mean test-accuracy difference (reference - model), bootstrap 95% CI\n"
        "> 0: reference better",
        color=TEXT_SECONDARY,
    )
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title(
        "Paired over datasets in the released results (added statistics)",
        loc="left",
        fontsize=11,
        color=TEXT,
    )
    fig.text(0.01, -0.02, SCOPE_REANALYSIS + ".", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout()
    _save(fig, "s1_paired_differences.png", run_dir)


def plot_correlations(run_dir: Path):
    table = pd.read_csv(
        run_dir / "hyperparameter_correlations_released_script_scope.csv"
    )
    shown = [
        "t",
        "learning_rate_init",
        "repeats",
        "temperature",
        "n_input_copies",
        "observable_type",
        "learning_rate",
        "gamma_factor",
        "alpha",
        "hidden_layer_sizes",
        "trotter_steps",
        "n_qfeatures",
        "visible_qubits",
        "n_layers",
        "C",
        "n_episodes",
    ]
    categorical = {
        "observable_type",
        "visible_qubits",
        "hidden_layer_sizes",
        "n_qfeatures",
    }
    table = table[table["hyperparameter"].isin(shown)].sort_values("spearman_mean")
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
    y = np.arange(len(table))
    colors = [
        SERIES[1] if h in categorical else SERIES[0] for h in table["hyperparameter"]
    ]
    ax.barh(
        y,
        2 * table["spearman_std"],
        left=table["spearman_mean"] - table["spearman_std"],
        color=colors,
        alpha=0.35,
        height=0.7,
        edgecolor=SURFACE,
    )
    ax.scatter(
        table["spearman_mean"], y, color=colors, s=30, zorder=3, edgecolor=SURFACE
    )
    labels = [
        f"{h} (categorical)" if h in categorical else h for h in table["hyperparameter"]
    ]
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.set_xlim(-1, 1)
    _style(ax)
    ax.set_xlabel(
        "Spearman correlation with mean CV accuracy (mean +- std over grids)",
        color=TEXT_SECONDARY,
    )
    ax.set_title(
        "Hyperparameter-accuracy correlation (paper Fig. 8)",
        loc="left",
        fontsize=11,
        color=TEXT,
    )
    fig.text(
        0.01,
        -0.04,
        SCOPE_REANALYSIS + ". Same file scope as the released script.\n"
        "Orange: categorical values ranked alphabetically, so their sign is arbitrary.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout()
    _save(fig, "s1_hyperparameter_correlations.png", run_dir)


def plot_positivity(pb_dir: Path):
    table = pd.read_csv(pb_dir / "positivity_bias.csv")
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    bins = np.linspace(0.6, 0.78, 61)
    for color, column, label in (
        (SERIES[0], "published_classical_mean", "published classical mean"),
        (
            SERIES[1],
            "published_quantum_mean",
            "published quantum mean (best of 20 designs)",
        ),
    ):
        ax.hist(
            table[column],
            bins=bins,
            color=color,
            alpha=0.75,
            label=label,
            edgecolor=SURFACE,
        )
        ax.axvline(table[column].mean(), color=color, linewidth=2, ymax=0.72)
    _style(ax)
    ax.set_xlabel(
        "mean performance reported by 100 researchers\n(true means: classical 0.65, quantum 0.55; vertical lines: means over repetitions)",
        color=TEXT_SECONDARY,
    )
    ax.set_ylabel("repetitions", color=TEXT_SECONDARY)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)  # headroom so the legend clears the bars
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(
        "Positivity bias flips the published ordering (paper Fig. 3)",
        loc="left",
        fontsize=11,
        color=TEXT,
    )
    fig.text(0.01, -0.06, SCOPE_SIMULATION + ".", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout()
    _save(fig, "s1_positivity_bias.png", pb_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reanalysis_run", type=Path)
    parser.add_argument("positivity_run", type=Path)
    args = parser.parse_args()
    plot_rankings(args.reanalysis_run)
    plot_ablations(args.reanalysis_run)
    plot_paired(args.reanalysis_run)
    plot_correlations(args.reanalysis_run)
    plot_positivity(args.positivity_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
