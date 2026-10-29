"""Positivity-bias thought experiment (paper Sec. I A, Fig. 3).

Each of ``n_researchers`` draws one classical baseline score from
N(0.65, 0.07) and evaluates ``n_designs`` quantum designs drawn from
N(0.55, 0.10), then publishes the design with the largest quantum-minus-
classical gap (with one classical score per researcher, the best quantum
design). The paper reports published means of 0.74 (quantum) vs 0.65
(classical). The released ``paper/plots/plot_positivity_bias.py`` runs this
once without a seed; here it is repeated over seeds to give the spread.
"""

from __future__ import annotations

import numpy as np


def simulate_published_means(
    seed: int,
    n_researchers: int,
    n_designs: int,
    quantum_mean: float,
    quantum_std: float,
    classical_mean: float,
    classical_std: float,
) -> tuple[float, float]:
    """Run one 100-researcher experiment and return the published means.

    Parameters
    ----------
    seed : int
        Seed of this repetition.
    n_researchers : int
        Number of researchers who each publish one comparison.
    n_designs : int
        Quantum designs tried per researcher before publishing the best.
    quantum_mean, quantum_std : float
        "True" quantum performance distribution.
    classical_mean, classical_std : float
        "True" classical performance distribution.

    Returns
    -------
    tuple of float
        ``(published quantum mean, published classical mean)``.
    """
    rng = np.random.default_rng(seed)
    classical = rng.normal(classical_mean, classical_std, size=n_researchers)
    quantum = rng.normal(quantum_mean, quantum_std, size=(n_researchers, n_designs))
    # Published design: smallest classical - quantum gap (released code sorts by it).
    best_design = np.argmin(classical[:, None] - quantum, axis=1)
    published_quantum = quantum[np.arange(n_researchers), best_design]
    return float(published_quantum.mean()), float(classical.mean())
