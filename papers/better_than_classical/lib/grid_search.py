"""Hyperparameter search and final scoring with the protocol of the original study.

Reproduces, for one (model, dataset) cell, what ``scripts/run_hyperparameter_search.py``
and ``scripts/score_with_best_hyperparameters.py`` of qml-benchmarks do:

1. Full grid search with scikit-learn's default 5-fold CV for classifiers
   (``StratifiedKFold(5)``, no shuffling), scored by accuracy. A fold whose
   fit raises (in particular ``ConvergenceWarning`` when the loss has not
   flattened after ``max_steps``) scores NaN, which makes the grid point's
   mean NaN and excludes it, as ``GridSearchCV(error_score=nan)`` does.
2. Selection of the grid point with the highest mean CV accuracy; ties go to
   the first point in ``ParameterGrid`` order (``GridSearchCV.best_index_``).
3. Retraining of the selected setting on the full training split with
   ``random_state`` 0..4 and scoring on the test split. The released script
   would crash on a non-converged retrain; here such a seed is recorded as
   ``converged = False`` with NaN accuracies.

The CV loop is written out instead of calling ``GridSearchCV`` so that every
fit is logged and counted, and because scikit-learn >= 1.6 no longer
recognises the vendored ``BaseEstimator, ClassifierMixin`` classes as
classifiers (``GridSearchCV`` would silently use unstratified ``KFold``).
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

from .hyperparameters import expand_grid
from .models import build_model

logger = logging.getLogger(__name__)

N_FOLDS = 5


@dataclass
class FitOutcome:
    """Result of fitting one model once."""

    accuracy: float
    train_accuracy: float
    converged: bool
    n_steps: int | None
    fit_seconds: float
    error: str | None


def fit_and_score(
    model_name: str,
    params: dict[str, Any],
    random_state: int,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> FitOutcome:
    """Fit one model and score it, turning fit failures into NaN outcomes.

    Parameters
    ----------
    model_name : str
        Registry name.
    params : dict
        Hyperparameters.
    random_state : int
        Model seed.
    X_fit, y_fit : numpy.ndarray
        Training data for this fit.
    X_eval, y_eval : numpy.ndarray
        Data on which accuracy is reported.

    Returns
    -------
    FitOutcome
        Accuracies (NaN when the fit failed), convergence flag, optimiser
        steps (``None`` for models without a gradient loop) and wall time.
    """
    model = build_model(model_name, params, random_state)
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            # Deprecation chatter from scikit-learn/PennyLane is not evidence.
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=DeprecationWarning)
            model.fit(X_fit, y_fit)
            accuracy = float(model.score(X_eval, y_eval))
            train_accuracy = float(model.score(X_fit, y_fit))
        error = None
        # converged_ is None for models without (or with a disabled) convergence check.
        converged = getattr(model, "converged_", None) is not False
    except ConvergenceWarning as exc:
        accuracy = train_accuracy = float("nan")
        error = f"ConvergenceWarning: {exc}"
        converged = False
    except Exception as exc:  # noqa: BLE001 - GridSearchCV(error_score=nan) semantics
        accuracy = train_accuracy = float("nan")
        error = f"{type(exc).__name__}: {exc}"
        converged = False
        logger.warning(
            "FIT_FAILED | model=%s | params=%s | error=%s", model_name, params, error
        )
    n_steps = getattr(model, "n_steps_", None)
    return FitOutcome(
        accuracy=accuracy,
        train_accuracy=train_accuracy,
        converged=converged,
        n_steps=None if n_steps is None else int(n_steps),
        fit_seconds=time.perf_counter() - start,
        error=error,
    )


def run_grid_search(
    model_name: str,
    grid: dict[str, list[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    search_seed: int,
    fixed_params: dict[str, Any],
    on_fit_completed: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Evaluate every grid point with stratified 5-fold CV.

    Parameters
    ----------
    model_name : str
        Registry name.
    grid : dict
        Hyperparameter grid ``{name: [values]}``.
    X_train, y_train : numpy.ndarray
        Training split (the test split is never seen here).
    search_seed : int
        ``random_state`` of every model during the search (the released
        search used the model default, 42).
    fixed_params : dict
        Constructor arguments applied to every grid point (empty for paper
        runs; smoke configs use it to cap ``max_steps``).
    on_fit_completed : callable, optional
        Called as ``on_fit_completed(done, total)`` after each fit. Default
        value is None.

    Returns
    -------
    pandas.DataFrame
        One row per grid point: ``param_<name>`` columns, ``split<k>_accuracy``,
        ``mean_test_accuracy`` / ``std_test_accuracy`` (NaN if any fold
        failed, as in ``GridSearchCV.cv_results_``), ``rank_test_accuracy``,
        fit-time, step and convergence summaries, and fold errors.
    """
    points = expand_grid(grid)
    folds = list(StratifiedKFold(n_splits=N_FOLDS).split(X_train, y_train))
    total_fits = len(points) * len(folds)
    rows = []
    done = 0
    for point_index, params in enumerate(points):
        fold_outcomes = []
        for fold_index, (fit_idx, val_idx) in enumerate(folds):
            outcome = fit_and_score(
                model_name,
                {**fixed_params, **params},
                search_seed,
                X_train[fit_idx],
                y_train[fit_idx],
                X_train[val_idx],
                y_train[val_idx],
            )
            fold_outcomes.append(outcome)
            done += 1
            logger.info(
                "TRAIN_FIT_COMPLETED | fit=%d/%d | point=%d/%d | fold=%d/%d | cv_accuracy=%.6f | "
                "converged=%s | steps=%s | seconds=%.2f",
                done,
                total_fits,
                point_index + 1,
                len(points),
                fold_index + 1,
                len(folds),
                outcome.accuracy,
                outcome.converged,
                outcome.n_steps,
                outcome.fit_seconds,
            )
            if on_fit_completed is not None:
                on_fit_completed(done, total_fits)
        accuracies = np.array([o.accuracy for o in fold_outcomes])
        row = {f"param_{key}": value for key, value in params.items()}
        row["params"] = repr(params)
        for fold_index, accuracy in enumerate(accuracies):
            row[f"split{fold_index}_accuracy"] = accuracy
        # np.mean propagates NaN: a grid point with a failed fold is excluded.
        row["mean_test_accuracy"] = float(np.mean(accuracies))
        row["std_test_accuracy"] = float(np.std(accuracies))
        row["mean_fit_seconds"] = float(np.mean([o.fit_seconds for o in fold_outcomes]))
        steps = [o.n_steps for o in fold_outcomes if o.n_steps is not None]
        row["mean_steps"] = float(np.mean(steps)) if steps else float("nan")
        row["n_not_converged"] = sum(not o.converged for o in fold_outcomes)
        row["fold_errors"] = (
            " || ".join(o.error for o in fold_outcomes if o.error) or ""
        )
        rows.append(row)
        logger.info(
            "GRID_POINT_COMPLETED | point=%d/%d | params=%s | mean_cv_accuracy=%.6f",
            point_index + 1,
            len(points),
            params,
            row["mean_test_accuracy"],
        )
    table = pd.DataFrame(rows)
    # GridSearchCV ranks NaN means last; rank 1 is the best mean accuracy.
    table["rank_test_accuracy"] = (
        table["mean_test_accuracy"]
        .fillna(-np.inf)
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return table


def select_best(
    cv_results: pd.DataFrame, grid: dict[str, list[Any]]
) -> tuple[int, dict[str, Any]]:
    """Select the grid point with the highest mean CV accuracy.

    Ties go to the first point in grid order. Raises if every point failed.

    Returns
    -------
    tuple
        ``(row_index, params)`` of the selected grid point.

    Raises
    ------
    RuntimeError
        If no grid point has a finite mean CV accuracy.
    """
    means = cv_results["mean_test_accuracy"].to_numpy()
    if not np.isfinite(means).any():
        raise RuntimeError(
            "Every grid point failed or did not converge in at least one fold"
        )
    best_index = int(np.nanargmax(np.where(np.isfinite(means), means, -np.inf)))
    return best_index, expand_grid(grid)[best_index]


def score_with_seeds(
    model_name: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seeds: list[int],
    fixed_params: dict[str, Any],
) -> pd.DataFrame:
    """Retrain on the full training split for each seed and score on test.

    Returns
    -------
    pandas.DataFrame
        One row per seed with ``train_acc``, ``test_acc``, ``converged``,
        ``steps``, ``fit_seconds`` and ``error`` (the released
        ``*-best-hyperparams-results.csv`` has the first two columns).
    """
    rows = []
    for seed in seeds:
        logger.info(
            "EVALUATION_STARTED | split=test | seed=%d | params=%s", seed, params
        )
        outcome = fit_and_score(
            model_name,
            {**fixed_params, **params},
            seed,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        rows.append(
            {
                "seed": seed,
                "train_acc": outcome.train_accuracy,
                "test_acc": outcome.accuracy,
                "converged": outcome.converged,
                "steps": outcome.n_steps,
                "fit_seconds": outcome.fit_seconds,
                "error": outcome.error or "",
            }
        )
        logger.info(
            "EVALUATION_COMPLETED | split=test | seed=%d | test_accuracy=%.6f | train_accuracy=%.6f | "
            "converged=%s | seconds=%.2f",
            seed,
            outcome.accuracy,
            outcome.train_accuracy,
            outcome.converged,
            outcome.fit_seconds,
        )
    return pd.DataFrame(rows)
