"""Benchmark datasets of Bowles, Ahmed & Schuld (2024), loaded from local copies.

Sources (acquired and verified in Phase 01, see LOG.md "Data Acquisition Log"):

- ``<data_root>/better_than_classical/pennylane_datasets/*.h5``: the authors'
  hosted PennyLane datasets, i.e. the exact data behind the released results.
- ``<data_root>/better_than_classical/sources/qml-benchmarks/paper/plots/
  datasets-for-plots/mnist_pca-/``: the authors' exact 250/250 mnist pca-
  samples, available only for d = 2 and d = 10.

Every split is a pair ``(X, y)`` with ``X`` of shape ``(n_samples, n_features)``
(images are flattened row-major, as in the released code) and labels in
``{-1, +1}``. Models apply their own scaling inside ``fit`` (paper Sec. II C),
so no preprocessing happens here.

The PennyLane HDF5 files store every value as its own scalar, which makes reads
slow (about 9 s for one mnist pca split), so each split is cached once as
``.npz`` under ``<data_root>/better_than_classical/cache/``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PAPER_NAME = "better_than_classical"
_PAPER_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PAPER_DIR.parents[1]
MNIST_PCA_MINUS_SIZE = 250

# name -> (HDF5 stem, train attribute, test attribute, meaning of `param`)
_HOSTED_DATASETS = {
    "linearly_separable": ("LinearlySeparable", "train", "test", "n_features"),
    "mnist_pca": ("MNISTPCA", "train", "test", "n_features"),
    "hidden_manifold": ("HiddenManifold", "train", "test", "n_features"),
    "hidden_manifold_diff": (
        "HiddenManifold",
        "diff_train",
        "diff_test",
        "manifold_dim",
    ),
    "two_curves": ("TwoCurves", "train", "test", "n_features"),
    "two_curves_diff": ("TwoCurves", "diff_train", "diff_test", "degree"),
    "hyperplanes_diff": ("Hyperplanes", "diff_train", "diff_test", "n_hyperplanes"),
    "bars_and_stripes": ("BarsAndStripes", "train", "test", "image_width"),
}
DERIVED_DATASETS = ("mnist_pca_minus", "mnist_pca_minus_released")
DATASET_NAMES = tuple(_HOSTED_DATASETS) + DERIVED_DATASETS


@dataclass(frozen=True)
class BenchmarkData:
    """Train and test splits of one benchmark dataset plus its provenance."""

    name: str
    param: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    source: str
    subset: str

    def identity(self) -> dict[str, Any]:
        """Return the ``dataset`` block required by ``run_status.json``."""
        return {
            "name": f"{self.name}[{self.param}]",
            "splits": {
                "train": {
                    "n_samples": int(len(self.y_train)),
                    "n_features": int(self.X_train.shape[1]),
                },
                "test": {
                    "n_samples": int(len(self.y_test)),
                    "n_features": int(self.X_test.shape[1]),
                },
            },
            "preprocessing": "none in the loader; each model scales inputs inside fit (paper Sec. II C)",
            "subset": self.subset,
            "source": self.source,
        }


def paper_data_dir(cfg: dict[str, Any]) -> Path:
    """Return the paper's data directory.

    ``dataset.root`` wins when set; otherwise ``<data_root>/better_than_classical``
    where ``data_root`` comes from ``--data-root`` / ``DATA_DIR`` through the
    shared runtime (``cfg["data_root"]``).

    Raises
    ------
    KeyError
        If neither ``dataset.root`` nor ``data_root`` / ``DATA_DIR`` is set.
    """
    root = cfg["dataset"].get("root")
    if root:
        return Path(root)
    data_root = cfg.get("data_root") or os.environ.get("DATA_DIR")
    if not data_root:
        raise KeyError(
            "Set cfg['data_root'], DATA_DIR or dataset.root to locate the data"
        )
    data_root = Path(data_root)
    # TODO: remove once runtime_lib.find_repo_root no longer stops at the paper
    # folder's own implementation.py. The shared runtime chdirs into the paper
    # before resolving its default, so it sets data_root (and DATA_DIR) to
    # papers/better_than_classical/data instead of the repository data root of
    # AGENTS.md. Only that exact, non-existent folder is redirected (same
    # workaround as papers/contextuality_inductive_bias/lib/data.py); any other
    # explicit --data-root / DATA_DIR is used as given.
    mis_resolved_default = _PAPER_DIR / "data"
    if (
        data_root.resolve() == mis_resolved_default.resolve()
        and not mis_resolved_default.exists()
    ):
        logger.warning(
            "DATA_ROOT_REDIRECTED | from=%s | to=%s | reason=runtime default resolved inside the paper folder",
            data_root,
            _REPO_ROOT / "data",
        )
        data_root = _REPO_ROOT / "data"
    return data_root / PAPER_NAME


def _read_hosted_split(h5_path: Path, attribute: str, key: str):
    import pennylane as qml

    dataset = qml.data.Dataset.open(str(h5_path), "r")
    split = getattr(dataset, attribute)
    if key not in split:
        raise KeyError(
            f"{h5_path.name}: no {attribute}[{key!r}]; available {sorted(split.keys())}"
        )
    return np.asarray(split[key]["inputs"], dtype=np.float64), np.asarray(
        split[key]["labels"], dtype=np.float64
    )


def _load_hosted(data_dir: Path, name: str, param: int) -> BenchmarkData:
    stem, train_attr, test_attr, _ = _HOSTED_DATASETS[name]
    h5_path = data_dir / "pennylane_datasets" / f"{stem}_{stem}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Missing hosted dataset {h5_path} (see LOG.md Data Acquisition Log)"
        )
    cache_path = data_dir / "cache" / f"{name}_{param}.npz"
    if cache_path.exists():
        arrays = np.load(cache_path)
        X_train, y_train, X_test, y_test = (
            arrays[k] for k in ("X_train", "y_train", "X_test", "y_test")
        )
    else:
        logger.info(
            "Converting %s[%d] from %s to %s", name, param, h5_path.name, cache_path
        )
        X_train, y_train = _read_hosted_split(h5_path, train_attr, str(param))
        X_test, y_test = _read_hosted_split(h5_path, test_attr, str(param))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
    return BenchmarkData(
        name,
        param,
        X_train,
        y_train,
        X_test,
        y_test,
        source=f"pennylane_datasets/{h5_path.name}:{train_attr}/{test_attr}[{param}]",
        subset="full",
    )


def _load_mnist_pca_minus(
    data_dir: Path, param: int, subsample_seed: int
) -> BenchmarkData:
    """Seeded 250/250 subsample of the hosted mnist pca split (documented deviation).

    The released code draws the subsample with ``random.choices`` (with
    replacement, unseeded, separately for every d). Here the same seeded
    indices, drawn without replacement, are used for every d, so datasets of
    different dimension contain the same images.
    """
    full = _load_hosted(data_dir, "mnist_pca", param)
    rng = np.random.default_rng(subsample_seed)
    train_idx = rng.choice(len(full.y_train), size=MNIST_PCA_MINUS_SIZE, replace=False)
    test_idx = rng.choice(len(full.y_test), size=MNIST_PCA_MINUS_SIZE, replace=False)
    return BenchmarkData(
        "mnist_pca_minus",
        param,
        full.X_train[train_idx],
        full.y_train[train_idx],
        full.X_test[test_idx],
        full.y_test[test_idx],
        source=full.source,
        subset=f"{MNIST_PCA_MINUS_SIZE}/{MNIST_PCA_MINUS_SIZE} without replacement, seed {subsample_seed}",
    )


def _load_mnist_pca_minus_released(data_dir: Path, param: int) -> BenchmarkData:
    folder = (
        data_dir / "sources/qml-benchmarks/paper/plots/datasets-for-plots/mnist_pca-"
    )
    splits = {}
    for split in ("train", "test"):
        path = folder / f"mnist_3-5_{param}d-250_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No released mnist pca- sample for d={param}: {path}"
            )
        table = np.loadtxt(path, delimiter=",")
        splits[split] = (table[:, :-1], table[:, -1])
    return BenchmarkData(
        "mnist_pca_minus_released",
        param,
        *splits["train"],
        *splits["test"],
        source=f"sources/qml-benchmarks/.../mnist_pca-/mnist_3-5_{param}d-250_*.csv",
        subset="authors' exact 250/250 sample",
    )


def load_dataset(cfg: dict[str, Any]) -> BenchmarkData:
    """Load the dataset named in ``cfg['dataset']``.

    Parameters
    ----------
    cfg : dict
        Resolved config with ``dataset.name`` and ``dataset.param`` (the
        varied quantity: number of features, manifold dimension, degree,
        number of hyperplanes or image width). ``mnist_pca_minus`` also needs
        ``dataset.subsample_seed``.

    Returns
    -------
    BenchmarkData
        Train/test arrays and provenance.

    Raises
    ------
    KeyError
        If a required config key is missing or the dataset name is unknown.
    """
    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"]
    param = int(dataset_cfg["param"])
    data_dir = paper_data_dir(cfg)
    if name in _HOSTED_DATASETS:
        data = _load_hosted(data_dir, name, param)
    elif name == "mnist_pca_minus":
        data = _load_mnist_pca_minus(
            data_dir, param, int(dataset_cfg["subsample_seed"])
        )
    elif name == "mnist_pca_minus_released":
        data = _load_mnist_pca_minus_released(data_dir, param)
    else:
        raise KeyError(f"Unknown dataset {name!r}; known: {DATASET_NAMES}")
    for split, labels in (("train", data.y_train), ("test", data.y_test)):
        if not set(np.unique(labels)) <= {-1.0, 1.0}:
            raise ValueError(f"{name}[{param}] {split} labels are not in {{-1, +1}}")
    return data


def build_dataloader(cfg: dict[str, Any], split: str):
    """Return the split as a one-batch iterable ``[(X, y)]``.

    The benchmark models are scikit-learn estimators that draw their own
    mini-batches inside ``fit`` (batch size is a model hyperparameter), so the
    loader hands over the whole split as a single batch.

    Parameters
    ----------
    cfg : dict
        Resolved config (see :func:`load_dataset`).
    split : {"train", "test"}
        Which split to return.

    Returns
    -------
    list of tuple
        ``[(X, y)]`` with ``X`` of shape ``(n_samples, n_features)``.
    """
    data = load_dataset(cfg)
    if split == "train":
        return [(data.X_train, data.y_train)]
    if split == "test":
        return [(data.X_test, data.y_test)]
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")
