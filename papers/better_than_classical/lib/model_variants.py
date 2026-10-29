"""Control variants added by this reproduction (not part of the original study).

Each class subclasses a vendored qml-benchmarks model and changes exactly one
thing, so a difference in accuracy can be attributed to that change:

- ``DataReuploadingClassifierNoCZ``: removes only the CZ entanglers from the
  data re-uploading circuit. The released ``DataReuploadingClassifierSeparable``
  also merges the data and trainable rotations into one gate per layer, which
  confounds "no entanglement" with "different embedding" (LOG.md, R7).
- ``QuantumBoltzmannMachineIndependentZZ``: gives every ZZ coupling its own
  parameter vector, as in Eq. (C8) of the paper. The released model reuses one
  vector for all couplings (``count`` is never incremented).
- ``WeiNetBoundCircuits``: binds each of the 9 circuits to its own row/column
  shift. In the released model the QNode closures are late-bound, so all 9
  circuits apply the same shift.
- ``WeiNetCoherentFilter``: applies the classical filter as a coherent sum of
  the 9 shift unitaries (post-selected linear combination of unitaries, the
  convolution of Wei et al.), instead of a classical mixture. A classical
  mixture of permutations followed by Z-basis readout only sees permuted
  squared amplitudes, so it cannot see the sign pattern that separates bars
  from stripes.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

from .qml_benchmarks.model_utils import chunk_vmapped_fn
from .qml_benchmarks.models.data_reuploading import DataReuploadingClassifier
from .qml_benchmarks.models.quantum_boltzmann_machine import (
    QuantumBoltzmannMachine,
    sigmaX,
    sigmaZ,
    tensor_ops,
)
from .qml_benchmarks.models.weinet import WeiNet


class DataReuploadingClassifierNoCZ(DataReuploadingClassifier):
    """Data re-uploading classifier with the CZ entanglers removed.

    The circuit is the released ``DataReuploadingClassifier`` circuit with the
    ``double`` / ``double_odd`` CZ layers deleted; the separate data rotation
    ``Rot(x * omega)`` and trainable rotation ``Rot(theta)`` are kept, as is
    the final ``Rot(x * omega + theta)`` layer, the loss and the prediction.
    """

    def construct_model(self):
        """Construct the entanglement-free circuit and its vectorised forward pass."""
        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            for layer in range(self.n_layers):
                x_idx = 0
                for i in range(self.n_qubits_):
                    angles = (
                        x[x_idx : x_idx + 3]
                        * params["omegas"][layer, x_idx : x_idx + 3]
                    )
                    qml.Rot(*angles, wires=i)
                    angles = params["thetas"][i, layer, :]
                    qml.Rot(*angles, wires=i)
                    x_idx += 3

            x_idx = 0
            for i in range(self.n_qubits_):
                angles = (
                    x[x_idx : x_idx + 3]
                    * params["omegas"][self.n_layers, x_idx : x_idx + 3]
                    + params["thetas"][i, self.n_layers, :]
                )
                qml.Rot(*angles, wires=i)
                x_idx += 3

            return [qml.expval(qml.PauliZ(wires=[i])) for i in range(self.n_qubits_)]

        self.circuit = circuit

        def circuit_as_array(params, x):
            return jnp.array(circuit(params, x))

        if self.jit:
            circuit_as_array = jax.jit(circuit_as_array)
        self.forward = jax.vmap(circuit_as_array, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)
        return self.forward


class QuantumBoltzmannMachineIndependentZZ(QuantumBoltzmannMachine):
    """Quantum Boltzmann machine with one parameter vector per ZZ coupling.

    Identical to the released model except that the Hamiltonian
    ``H(theta, x) = sum_j Z_j theta_j^Z.x + sum_j X_j theta_j^X.x
    + sum_{j<k} Z_j Z_k theta_jk.x`` (paper Eq. C8) uses a distinct
    ``theta_jk`` for every pair, which the released code allocates but never
    indexes.
    """

    def construct_model(self):
        """Construct the exact Gibbs-state forward pass."""
        singles = list(itertools.combinations(np.arange(self.n_qubits), 1))
        doubles = list(itertools.combinations(np.arange(self.n_qubits), 2))
        self.n_params_ = 2 * len(singles) + len(doubles)

        if self.observable_type == "sum":
            obs = (
                sum(
                    tensor_ops([sigmaZ], (i,), self.n_qubits)
                    for i in range(self.n_visible)
                )
                / self.n_visible
            )
        elif self.observable_type == "product":
            obs = 1.0
            for _ in range(self.n_visible):
                obs = jnp.kron(obs, sigmaZ)
        else:
            raise ValueError(f"Unknown observable_type {self.observable_type!r}")

        def gibbs_state(thetas, x):
            H = jnp.zeros([2**self.n_qubits, 2**self.n_qubits])
            count = 0
            for idxs in singles:
                H = H + tensor_ops([sigmaZ], idxs, self.n_qubits) * jnp.dot(
                    thetas[count], x
                )
                count += 1
                H = H + tensor_ops([sigmaX], idxs, self.n_qubits) * jnp.dot(
                    thetas[count], x
                )
                count += 1
            for idxs in doubles:
                H = H + tensor_ops([sigmaZ, sigmaZ], idxs, self.n_qubits) * jnp.dot(
                    thetas[count], x
                )
                count += 1
            state = jax.scipy.linalg.expm(-H / self.temperature, max_squarings=32)
            return state / jnp.trace(state)

        def model(thetas, x):
            return jnp.trace(jnp.matmul(gibbs_state(thetas, x), obs))

        if self.jit:
            model = jax.jit(model)
        self.forward = jax.vmap(model, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)
        return self.forward


def _shift_matrix(size, shift):
    """Return the cyclic shift permutation ``e_i -> e_{(i + shift) mod size}``.

    Matches the released WeiNet's ``V[i, (i + k) % height] = 1`` construction.
    """
    matrix = np.zeros([size, size])
    for i in range(size):
        matrix[i, (i + shift) % size] = 1.0
    return matrix


# Row/column offsets of the 3x3 filter taps, in the released WeiNet order.
_WEINET_OFFSETS = (-1, 0, 1)


class WeiNetBoundCircuits(WeiNet):
    """WeiNet with each of the 9 circuits bound to its own shift.

    Circuit ``(nu, mu)`` applies ``S_row(k_nu) (x) S_col(k_mu)``. Everything
    else (amplitude embedding, classical mixture over circuits through
    ``softmax(s)``, Z/ZZ readout, loss) is the released model.
    """

    def construct_models(self):
        """Construct the 9 QNodes, one per (row shift, column shift)."""
        operators = _weinet_readout_operators(self.n_qubits_)
        wires = range(self.n_qubits_)
        dev = qml.device(self.dev_type, wires=wires)
        circuits = []
        for row_shift in _WEINET_OFFSETS:
            for col_shift in _WEINET_OFFSETS:
                unitary = jnp.array(
                    np.kron(
                        _shift_matrix(self.height_, row_shift),
                        _shift_matrix(self.width_, col_shift),
                    )
                )

                # Default argument binds this iteration's unitary to the QNode.
                @qml.qnode(dev, **self.qnode_kwargs)
                def circuit(x, unitary=unitary):
                    qml.AmplitudeEmbedding(
                        jnp.reshape(x, -1), wires=wires, normalize=True, pad_with=0.0
                    )
                    qml.QubitUnitary(unitary, wires=wires)
                    return [qml.expval(op) for op in operators]

                self.circuit = circuit
                circuits.append(jax.jit(circuit) if self.jit else circuit)
        self.circuits = circuits


class WeiNetCoherentFilter(WeiNet):
    """WeiNet with the filter applied coherently (post-selected LCU).

    The work register ends in ``sum_{nu,mu} f[nu,mu] S_row(k_nu) S_col(k_mu)|x>``
    renormalised, i.e. the amplitude encoding of the circular 2D
    cross-correlation of the image with the chosen filter. ``<Z_i>`` and
    ``<Z_i Z_j>`` on the non-pooled qubits (the released readout operators) are
    computed exactly from the computational-basis probabilities, and a
    trainable linear readout (plus bias via the identity operator) gives the
    logit. There is no trainable mixture: the filter is fixed by
    ``filter_name``.
    """

    def construct_unitaries(self):
        """Build the fixed linear map of the coherent filter."""
        conv = np.zeros([self.height_ * self.width_] * 2)
        for nu, row_shift in enumerate(_WEINET_OFFSETS):
            for mu, col_shift in enumerate(_WEINET_OFFSETS):
                conv += float(self.filter[nu, mu]) * np.kron(
                    _shift_matrix(self.height_, row_shift),
                    _shift_matrix(self.width_, col_shift),
                )
        self.filter_map_ = jnp.array(conv)

    def construct_models(self):
        """Precompute the readout sign table (one row per operator)."""
        n_qubits = self.n_qubits_
        # PennyLane wire order: wire 0 is the most significant bit of the index.
        bits = (
            np.arange(2**n_qubits)[:, None] >> (n_qubits - 1 - np.arange(n_qubits))
        ) & 1
        z_signs = 1 - 2 * bits
        rows = []
        kept = [i for i in range(n_qubits - 1) if i != n_qubits // 2 - 1]
        for i in kept:
            rows.append(z_signs[:, i])
            for j in kept:
                if j > i:
                    rows.append(z_signs[:, i] * z_signs[:, j])
        rows.append(np.ones(2**n_qubits))
        self.readout_signs_ = jnp.array(np.stack(rows))

    def forward_fn(self, params, x):
        """Return the readout logit for one flattened, padded image."""
        state = self.filter_map_ @ jnp.reshape(x, -1)
        probs = state**2 / jnp.sum(state**2)
        expvals = self.readout_signs_ @ probs
        return jnp.sum(params["weights"] * expvals)

    def initialize_params(self):
        """Initialise only the linear readout weights (the filter is fixed)."""
        n_expvals = (
            (self.n_qubits_ - 2) + (self.n_qubits_ - 2) * (self.n_qubits_ - 3) // 2 + 1
        )
        self.params_ = {
            "weights": jax.random.normal(self.generate_key(), shape=(n_expvals,))
            / n_expvals
        }


def _weinet_readout_operators(n_qubits):
    """Return the released WeiNet readout operators (Z, ZZ on kept qubits, identity)."""
    operators = []
    for i in range(n_qubits - 1):
        if i != int(n_qubits / 2) - 1:
            operators.append(qml.PauliZ(wires=i))
            for j in range(i + 1, n_qubits - 1):
                if j != int(n_qubits / 2) - 1:
                    operators.append(qml.PauliZ(wires=i) @ qml.PauliZ(wires=j))
    operators.append(qml.Identity(wires=0))
    return operators
