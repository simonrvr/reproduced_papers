"""Hyperparameter grids for every model (paper Appendix C).

The original grids are read unchanged from the vendored
``qml_benchmarks/hyperparameter_settings.py``. This module only converts them
to ``{name: [values]}`` and adds grids for the models introduced by this
reproduction, each copied from its parent model so that a variant gets the
same tuning budget as the model it is compared with.
"""

from __future__ import annotations

import itertools
from typing import Any

from .models import parse_hyperparameter_value
from .qml_benchmarks.hyperparameter_settings import hyper_parameter_settings

# Variants added here (or released without their own grid) -> parent grid.
_PARENT_GRID = {
    "DataReuploadingClassifierNoCost": "DataReuploadingClassifier",
    "DataReuploadingClassifierNoScaling": "DataReuploadingClassifier",
    "DataReuploadingClassifierNoTrainableEmbedding": "DataReuploadingClassifier",
    "DataReuploadingClassifierNoCZ": "DataReuploadingClassifier",
    "QuantumBoltzmannMachineIndependentZZ": "QuantumBoltzmannMachine",
    "WeiNetBoundCircuits": "WeiNet",
    "WeiNetCoherentFilter": "WeiNet",
}

# LogisticRegression floor (Fair Baseline Plan, LOG.md): L2 penalty, C grid.
_ADDED_GRIDS = {
    "LogisticRegression": {"C": [0.1, 1.0, 10.0, 100.0]},
}


def model_grid(model_name: str) -> dict[str, list[Any]]:
    """Return the full hyperparameter grid of a model.

    Parameters
    ----------
    model_name : str
        Registry name of the model.

    Returns
    -------
    dict
        Mapping from constructor argument to the list of candidate values,
        with ``hidden_layer_sizes`` parsed to tuples.

    Raises
    ------
    KeyError
        If no grid is defined for the model.
    """
    if model_name in _ADDED_GRIDS:
        return {key: list(values) for key, values in _ADDED_GRIDS[model_name].items()}
    source_name = _PARENT_GRID.get(model_name, model_name)
    if source_name not in hyper_parameter_settings:
        raise KeyError(f"No hyperparameter grid for model {model_name!r}")
    return {
        key: [parse_hyperparameter_value(key, value) for value in spec["val"]]
        for key, spec in hyper_parameter_settings[source_name].items()
    }


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a grid into the list of parameter dictionaries.

    The order matches scikit-learn's ``ParameterGrid`` (keys sorted
    alphabetically, last key varying fastest), so grid-point indices are
    comparable with the released ``GridSearchCV`` tables.
    """
    keys = sorted(grid)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[k] for k in keys))
    ]
