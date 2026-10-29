"""Check one batch of every dataset used by the local staged plan (Phase 02).

Usage
-----
DATA_DIR=../../data python utils/check_data_loader.py
"""

import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PAPER_DIR), str(PAPER_DIR.parents[1])]

from lib.data import build_dataloader  # noqa: E402

# (dataset, param, expected train shape, expected test shape)
EXPECTED = [
    ("mnist_pca", 2, (11552, 2), (1902, 2)),
    ("mnist_pca", 8, (11552, 8), (1902, 8)),
    ("mnist_pca_minus", 6, (250, 6), (250, 6)),
    ("mnist_pca_minus_released", 2, (250, 2), (250, 2)),
    ("mnist_pca_minus_released", 10, (250, 10), (250, 10)),
    ("linearly_separable", 2, (240, 2), (60, 2)),
    ("linearly_separable", 8, (240, 8), (60, 8)),
    ("bars_and_stripes", 4, (1000, 16), (200, 16)),
    ("bars_and_stripes", 8, (1000, 64), (200, 64)),
]


def main() -> int:
    for name, param, train_shape, test_shape in EXPECTED:
        cfg = {
            "dataset": {"name": name, "param": param, "root": None, "subsample_seed": 0}
        }
        X, y = next(iter(build_dataloader(cfg, split="train")))
        X_test, y_test = next(iter(build_dataloader(cfg, split="test")))
        assert X.shape == train_shape, (name, param, X.shape)
        assert X_test.shape == test_shape, (name, param, X_test.shape)
        assert set(y) <= {-1.0, 1.0} and set(y_test) <= {-1.0, 1.0}
        balance = float((y == 1).mean())
        print(
            f"OK {name}[{param}] train {X.shape} test {X_test.shape} frac(+1)={balance:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
