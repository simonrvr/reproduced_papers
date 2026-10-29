"""Stage S2: compare port-validation sweeps with the authors' released scores.

For every completed run of the given sweep directories, reads ``scores.csv``
(our 5 retrain seeds) and ``released_scores.csv`` (the authors' 5 seeds for the
same model, data and hyperparameters) and applies the agreement rule declared
in LOG.md before launch:

    agree  <=>  |mean_ours - mean_released| <= max(2 * sqrt(s_o^2/5 + s_r^2/5), 0.01)

It also counts seeds whose test accuracy is identical to the released value
for the same ``random_state`` (bit-for-bit reproduction of that seed).
Writes ``s2_port_validation.csv`` / ``.json`` to every sweep directory and to
``results/``, and prints the table.

Usage
-----
python utils/s2_compare.py outdir/sweeps/s2_pv_mnist_pca [outdir/sweeps/s2_pv_mnist_pca_minus ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_DIR = Path(__file__).resolve().parents[1]
ABSOLUTE_TOLERANCE = 0.01
EXACT_MATCH_ATOL = 1e-9


def compare_sweep(sweep_dir: Path) -> pd.DataFrame:
    """Return one comparison row per planned run of a sweep."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    rows = []
    for run in status["runs"]:
        row = {
            "sweep_id": status["sweep_id"],
            **run["candidate"],
            "run_status": run["status"],
        }
        if run["status"] != "COMPLETED":
            rows.append(row)
            continue
        run_dir = sweep_dir / run["run_dir"]
        ours = pd.read_csv(run_dir / "scores.csv").sort_values("seed")
        released = pd.read_csv(run_dir / "released_scores.csv").sort_values("seed_row")
        o, r = ours["test_acc"].to_numpy(), released["test_acc"].to_numpy()
        tolerance = max(
            2 * np.sqrt(np.nanvar(o, ddof=1) / len(o) + np.var(r, ddof=1) / len(r)),
            ABSOLUTE_TOLERANCE,
        )
        diff = float(np.nanmean(o) - r.mean())
        same_length = len(o) == len(r)
        row.update(
            ours_mean=float(np.nanmean(o)),
            ours_std=float(np.nanstd(o, ddof=1)),
            released_mean=float(r.mean()),
            released_std=float(r.std(ddof=1)),
            diff=diff,
            tolerance=float(tolerance),
            agree=bool(abs(diff) <= tolerance),
            n_seeds_failed=int(np.isnan(o).sum()),
            n_seeds_identical=int(np.sum(np.abs(o - r) < EXACT_MATCH_ATOL))
            if same_length
            else None,
            ours_train_mean=float(np.nanmean(ours["train_acc"])),
            released_train_mean=float(released["train_acc"].mean()),
            mean_fit_seconds=float(ours["fit_seconds"].mean()),
            mean_steps=float(ours["steps"].mean())
            if ours["steps"].notna().any()
            else None,
            run_dir=str(run_dir.relative_to(PAPER_DIR)),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    tables = []
    for sweep_dir in (p.resolve() for p in args.sweep_dirs):
        table = compare_sweep(sweep_dir)
        table.to_csv(sweep_dir / "s2_port_validation.csv", index=False)
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    results = PAPER_DIR / "results"
    results.mkdir(exist_ok=True)
    combined.to_csv(results / "s2_port_validation.csv", index=False)
    summary = {
        "sweeps": [
            str(d.relative_to(PAPER_DIR))
            for d in (p.resolve() for p in args.sweep_dirs)
        ],
        "planned_runs": int(len(combined)),
        "completed_runs": int((combined["run_status"] == "COMPLETED").sum()),
        "cells_agreeing": int(
            combined.get("agree", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "rule": "agree if |mean_ours - mean_released| <= max(2*SE_diff, 0.01)",
        "rows": combined.to_dict(orient="records"),
    }
    (results / "s2_port_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    pd.set_option("display.width", 220)
    columns = [
        "sweep_id",
        "model",
        "param",
        "ours_mean",
        "ours_std",
        "released_mean",
        "released_std",
        "diff",
        "tolerance",
        "agree",
        "n_seeds_identical",
        "n_seeds_failed",
        "mean_fit_seconds",
    ]
    print(
        combined[[c for c in columns if c in combined]].round(4).to_string(index=False)
    )
    print(
        f"agreeing cells: {summary['cells_agreeing']}/{summary['completed_runs']} completed "
        f"({summary['planned_runs']} planned)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
