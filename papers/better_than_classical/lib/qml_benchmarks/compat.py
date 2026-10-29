"""Compatibility shims for running the 2024 qml-benchmarks code on current libraries.

The reference code (XanaduAI/qml-benchmarks, commit 95e5a07) targets
PennyLane ~0.34. ``qml.broadcast`` was removed in PennyLane 0.40; this module
re-implements the wire patterns the vendored models use, copied from
PennyLane 0.39 ``pennylane/templates/broadcast.py`` (Apache-2.0), so the
gate sequences are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def _pattern_wire_pairs(n_wires: int, pattern: str | Sequence) -> list[list[int]]:
    """Return the wire sequence of a PennyLane 0.39 ``qml.broadcast`` pattern.

    Parameters
    ----------
    n_wires : int
        Number of wires, labelled ``0 .. n_wires - 1``.
    pattern : str or sequence
        One of ``"double"``, ``"double_odd"``, ``"chain"``, ``"ring"``, or an
        explicit sequence of wire tuples (PennyLane's custom pattern).

    Returns
    -------
    list of list of int
        Wires for each application of the broadcast unitary, in order.

    Raises
    ------
    ValueError
        If the pattern is not one used by the vendored models.
    """
    if not isinstance(pattern, str):
        return [list(wires) for wires in pattern]
    if pattern == "double":
        return [[i, i + 1] for i in range(0, n_wires - 1, 2)]
    if pattern == "double_odd":
        return [[i, i + 1] for i in range(1, n_wires - 1, 2)]
    if pattern == "chain":
        return [[i, i + 1] for i in range(n_wires - 1)]
    if pattern == "ring":
        if n_wires in (0, 1):
            return []
        if n_wires == 2:
            # PennyLane deviation kept: for 2 wires the ring equals the chain.
            return [[0, 1]]
        return [[i, (i + 1) % n_wires] for i in range(n_wires)]
    raise ValueError(f"Unsupported broadcast pattern: {pattern!r}")


def broadcast(unitary: Callable, wires: Sequence[int], pattern: str | Sequence) -> None:
    """Apply a parameter-free two-qubit gate in a ``qml.broadcast`` pattern.

    Parameters
    ----------
    unitary : callable
        PennyLane operation class, for example ``qml.CZ`` or ``qml.CNOT``.
    wires : sequence of int
        Contiguous wire labels starting at 0, as used by the reference models.
    pattern : str or sequence
        Pattern accepted by :func:`_pattern_wire_pairs`.
    """
    wires = list(wires)
    if wires != list(range(len(wires))):
        raise ValueError("compat.broadcast expects wires labelled 0..n-1")
    for gate_wires in _pattern_wire_pairs(len(wires), pattern):
        unitary(wires=gate_wires)


def restore_jax_core_aliases() -> None:
    """Re-expose two JAX functions that PennyLane 0.45.1 still imports.

    PennyLane 0.45.1 calls ``jax.core.is_concrete`` (``pennylane/math/utils.py``)
    and ``jax.core.concrete_or_error`` (``pennylane/math/single_dispatch.py``),
    which JAX 0.11 removed from its public namespace. The implementations
    still exist as ``jax._src.core.is_concrete`` and
    ``jax._src.core.concrete_or_error``; the removed public names were
    aliases of exactly these functions, so restoring the aliases does not
    change behaviour.

    TODO: remove once the environment's PennyLane no longer references the
    removed ``jax.core`` names (or JAX is pinned below 0.11).
    """
    import jax.core
    from jax._src import core as jax_core_internal

    for name in ("is_concrete", "concrete_or_error"):
        if name not in vars(jax.core):
            setattr(jax.core, name, getattr(jax_core_internal, name))
