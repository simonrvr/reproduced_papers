"""Dataset loader (needs the acquired data)."""

from __future__ import annotations

import numpy as np
import pytest
from better_than_classical.lib.data import build_dataloader, load_dataset
from common import REPO_ROOT

DATA_ROOT = REPO_ROOT / "data"
pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "better_than_classical" / "pennylane_datasets").exists(),
    reason="acquired datasets not present (see LOG.md Data Acquisition Log)",
)


def _cfg(name, param):
    return {
        "data_root": str(DATA_ROOT),
        "dataset": {"name": name, "param": param, "root": None, "subsample_seed": 0},
    }


def test_linearly_separable_split_shapes():
    X, y = next(iter(build_dataloader(_cfg("linearly_separable", 3), "train")))
    assert X.shape == (240, 3) and set(np.unique(y)) == {-1.0, 1.0}


def test_mnist_pca_minus_uses_same_images_for_every_dimension():
    low = load_dataset(_cfg("mnist_pca_minus", 2))
    high = load_dataset(_cfg("mnist_pca_minus", 6))
    np.testing.assert_array_equal(low.y_train, high.y_train)
    # PCA components are nested (up to the randomized SVD solver used per d).
    np.testing.assert_allclose(low.X_train, high.X_train[:, :2], atol=1e-2)


def test_unknown_dataset_fails_clearly():
    with pytest.raises(KeyError):
        load_dataset(_cfg("not_a_dataset", 2))
