"""Estimate the grid-search cost of the original study from released results.

Reads the authors' released ``*_GridSearchCV.csv`` tables (one row per grid
point, with per-fold fit and score times measured on the Cedar cluster) and
sums ``(mean_fit_time + mean_score_time) * 5 folds`` over the grid, per
(benchmark, model, dataset file). Used in Phase 01 to assign compute tiers.

Usage
-----
python utils/estimate_grid_cost.py \
    ../../data/better_than_classical/sources/qml-benchmarks out.csv
"""

import argparse
import glob
import os
import re

import pandas as pd

N_FOLDS = 5


def estimate_costs(reference_root):
    """Return one row per released grid-search table with its cost in hours.

    Parameters
    ----------
    reference_root : str
        Root of the qml-benchmarks snapshot (contains ``paper/results``).

    Returns
    -------
    pandas.DataFrame
        Columns ``bench``, ``model``, ``file``, ``first_int``, ``n_grid``,
        ``cost_h`` and ``max_fit_s``. ``first_int`` is the first integer in
        the dataset part of the filename (the dimension d for most
        benchmarks, but the fixed d = 10 or the manifold size for DIFF and
        hidden-manifold files).
    """
    rows = []
    pattern = os.path.join(reference_root, "paper/results/*/*/*_GridSearchCV.csv")
    for path in sorted(glob.glob(pattern)):
        bench = path.split("/results/")[1].split("/")[0]
        model = path.split("/")[-2]
        dataset_part = os.path.basename(path).replace(model, "")
        match = re.search(r"(\d+)", dataset_part.replace("3-5", ""))
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        cost_s = ((table["mean_fit_time"] + table["mean_score_time"]) * N_FOLDS).sum()
        rows.append(
            {
                "bench": bench,
                "model": model,
                "file": os.path.basename(path),
                "first_int": int(match.group(1)) if match else -1,
                "n_grid": len(table),
                "cost_h": cost_s / 3600,
                "max_fit_s": table["mean_fit_time"].max(),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference_root")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    costs = estimate_costs(args.reference_root)
    costs.to_csv(args.output_csv, index=False)
    summary = costs.groupby(["bench", "model"]).agg(
        n_files=("file", "count"), grid=("n_grid", "max"), total_h=("cost_h", "sum")
    )
    print(summary.round(2).to_string())
    print(f"TOTAL sequential hours: {costs.cost_h.sum():.1f}")


if __name__ == "__main__":
    main()
