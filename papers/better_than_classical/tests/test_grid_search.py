"""Grid-search protocol: CV splits, NaN handling, selection and scoring."""

from __future__ import annotations

import common  # noqa: F401  (puts papers/ on sys.path)
import numpy as np
import pandas as pd
import pytest
from better_than_classical.lib.grid_search import (
    run_grid_search,
    score_with_seeds,
    select_best,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def _data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = np.where(X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.normal(size=n) > 0, 1.0, -1.0)
    return X, y


def test_cv_scores_match_sklearn_gridsearchcv():
    X, y = _data()
    grid = {"C": [0.1, 1.0, 10.0]}
    ours = run_grid_search(
        "LogisticRegression",
        grid,
        X,
        y,
        search_seed=0,
        fixed_params={"max_iter": 10000},
    )
    reference = GridSearchCV(
        LogisticRegression(max_iter=10000, random_state=0), grid, cv=5
    ).fit(X, y)
    np.testing.assert_allclose(
        ours["mean_test_accuracy"], reference.cv_results_["mean_test_score"]
    )
    assert (
        ours["rank_test_accuracy"].tolist()
        == reference.cv_results_["rank_test_score"].tolist()
    )
    best_index, best_params = select_best(ours, grid)
    assert best_index == reference.best_index_ and best_params == reference.best_params_


def test_failed_fold_excludes_point_and_all_failed_raises():
    table = pd.DataFrame({"mean_test_accuracy": [np.nan, 0.7, 0.7]})
    assert select_best(table, {"C": [1, 2, 3]}) == (1, {"C": 2})
    with pytest.raises(RuntimeError):
        select_best(
            pd.DataFrame({"mean_test_accuracy": [np.nan, np.nan]}), {"C": [1, 2]}
        )


def test_invalid_grid_point_scores_nan_instead_of_crashing():
    X, y = _data()
    table = run_grid_search(
        "LogisticRegression", {"C": [-1.0, 1.0]}, X, y, search_seed=0, fixed_params={}
    )
    assert (
        np.isnan(table.loc[0, "mean_test_accuracy"])
        and "InvalidParameterError" in table.loc[0, "fold_errors"]
    )
    assert np.isfinite(table.loc[1, "mean_test_accuracy"])


def test_scores_one_row_per_seed():
    X, y = _data()
    scores = score_with_seeds(
        "SVC", {"C": 1.0, "gamma": 0.1}, X[:90], y[:90], X[90:], y[90:], [0, 1, 2], {}
    )
    assert scores["seed"].tolist() == [0, 1, 2]
    assert scores["test_acc"].between(0, 1).all() and scores["converged"].all()


def test_quantum_model_fit_records_steps():
    X, y = _data(n=40)
    table = run_grid_search(
        "DataReuploadingClassifier",
        {"learning_rate": [0.1], "n_layers": [1], "observable_type": ["single"]},
        X,
        y,
        search_seed=42,
        fixed_params={"max_steps": 3, "convergence_interval": None},
    )
    assert table.loc[0, "mean_steps"] == 3 and table.loc[0, "n_not_converged"] == 0
