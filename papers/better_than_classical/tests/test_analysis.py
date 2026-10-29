"""Ranking re-analysis and the positivity-bias simulation."""

from __future__ import annotations

from pathlib import Path

import common  # noqa: F401  (puts papers/ on sys.path)
import numpy as np
import pandas as pd
import pytest
from better_than_classical.lib.positivity_bias import simulate_published_means
from better_than_classical.lib.reanalysis import (
    family_models,
    load_released_scores,
    rank_family,
    summarize_ranks,
)

RESULTS = (
    Path(__file__).resolve().parents[3]
    / "data/better_than_classical/sources/qml-benchmarks/paper/results"
)


def test_rank_family_uses_min_ranks_and_percentages():
    scores = pd.DataFrame(
        {
            "benchmark": ["b"] * 6,
            "param": [2] * 6,
            "model": ["A", "A", "B", "B", "C", "C"],
            "test_acc": [0.9, 0.9, 0.8, 0.8, 0.9, 0.9],
        }
    )
    ranked = rank_family(scores, ["A", "B", "C"], restrict_to_complete=False).set_index(
        "model"
    )
    assert (
        ranked.loc["A", "rank"] == 1
        and ranked.loc["C", "rank"] == 1
        and ranked.loc["B", "rank"] == 3
    )
    assert ranked.loc["B", "rank_pct"] == pytest.approx(1.0)
    summary = summarize_ranks(ranked.reset_index(), ["A", "B", "C"])
    assert summary["model"].tolist()[-1] == "B"


def test_positivity_bias_reverses_the_ordering():
    quantum, classical = zip(
        *[
            simulate_published_means(s, 100, 20, 0.55, 0.1, 0.65, 0.07)
            for s in range(50)
        ]
    )
    assert np.mean(quantum) == pytest.approx(0.74, abs=0.01)
    assert np.mean(classical) == pytest.approx(0.65, abs=0.01)


@pytest.mark.skipif(
    not RESULTS.exists(), reason="released qml-benchmarks results not present"
)
def test_released_rankings_put_classical_models_first():
    for family, first in (
        ("qnn", "MLPClassifier"),
        ("kernel", "SVC"),
        ("cnn", "ConvolutionalNeuralNetwork"),
    ):
        models = family_models(family, with_separable=False)
        scores = load_released_scores(RESULTS, models)
        summary = summarize_ranks(rank_family(scores, models, False), models)
        assert summary["model"].iloc[0] == first
