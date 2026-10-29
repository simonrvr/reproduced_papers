"""The qml.broadcast replacement reproduces PennyLane 0.39 wire patterns."""

from __future__ import annotations

import common  # noqa: F401  (puts papers/ on sys.path)
import pytest
from better_than_classical.lib.qml_benchmarks.compat import _pattern_wire_pairs


@pytest.mark.parametrize(
    ("pattern", "n_wires", "expected"),
    [
        ("double", 5, [[0, 1], [2, 3]]),
        ("double_odd", 5, [[1, 2], [3, 4]]),
        ("double_odd", 2, []),
        ("chain", 4, [[0, 1], [1, 2], [2, 3]]),
        ("ring", 1, []),
        ("ring", 2, [[0, 1]]),
        ("ring", 4, [[0, 1], [1, 2], [2, 3], [3, 0]]),
        ([[2, 0], [3, 1]], 4, [[2, 0], [3, 1]]),
    ],
)
def test_broadcast_patterns_match_pennylane_039(pattern, n_wires, expected):
    assert _pattern_wire_pairs(n_wires, pattern) == expected


def test_jax_core_aliases_are_restored():
    import better_than_classical.lib.qml_benchmarks  # noqa: F401
    import jax.core

    assert callable(jax.core.is_concrete)
    assert callable(jax.core.concrete_or_error)
