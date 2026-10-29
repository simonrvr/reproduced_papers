"""Dropping diff_method=None from the kernel models' QNodes changes no output.

PennyLane 0.45 executes ``diff_method=None`` QNodes through a host callback
inside ``jax.jit`` (hundreds of times slower), so the port uses the default
differentiation method. These tests pin that both settings give identical
kernel matrices / features (PORTING.md).
"""

from __future__ import annotations

import common  # noqa: F401  (puts papers/ on sys.path)
import numpy as np
import pytest
from better_than_classical.lib.models import build_model

RELEASED = {
    "SeparableKernelClassifier": (
        {"interface": "jax", "diff_method": None},
        {"encoding_layers": 2},
    ),
    "IQPKernelClassifier": (
        {"interface": "jax-jit", "diff_method": None},
        {"repeats": 2},
    ),
    "ProjectedQuantumKernel": (
        {"interface": "jax-jit", "diff_method": None},
        {"trotter_steps": 2},
    ),
}


def _data(n_features, n=12):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n, n_features))
    return X, np.where(X[:, 0] > 0, 1, -1)


@pytest.mark.parametrize("name", sorted(RELEASED))
def test_kernel_matrix_identical_to_released_qnode_setting(name):
    released_kwargs, params = RELEASED[name]
    X, y = _data(3)
    ported = build_model(name, params, 0).fit(X, y)
    released = build_model(name, {**params, "qnode_kwargs": released_kwargs}, 0).fit(
        X, y
    )
    Xs = ported.transform(X)
    # Compiled vs callback execution may differ by float rounding (~1e-16).
    np.testing.assert_allclose(
        np.asarray(ported.precompute_kernel(Xs, Xs)),
        np.asarray(released.precompute_kernel(Xs, Xs)),
        rtol=0,
        atol=1e-12,
    )


def test_kitchen_sinks_features_identical_to_released_qnode_setting():
    X, y = _data(4, n=20)
    params = {"n_episodes": 5, "n_qfeatures": "full"}
    ported = build_model("QuantumKitchenSinks", params, 0).fit(X, y)
    released = build_model(
        "QuantumKitchenSinks",
        {**params, "qnode_kwargs": {"interface": "jax", "diff_method": None}},
        0,
    ).fit(X, y)
    np.testing.assert_array_equal(
        np.asarray(ported.transform(X)), np.asarray(released.transform(X))
    )
    np.testing.assert_array_equal(ported.predict(X), released.predict(X))
