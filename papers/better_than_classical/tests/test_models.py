"""Model registry, grids and the control variants added by this reproduction."""

from __future__ import annotations

import common  # noqa: F401  (puts papers/ on sys.path)
import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest
from better_than_classical.lib.hyperparameters import expand_grid, model_grid
from better_than_classical.lib.models import ALL_MODEL_NAMES, build_model

# Grid sizes of the paper's Appendix C (including the fixed batch_size / max_vmap).
PAPER_GRID_SIZES = {
    "CircuitCentricClassifier": 27,
    "DataReuploadingClassifier": 36,
    "DressedQuantumCircuitClassifier": 12,
    "IQPVariationalClassifier": 36,
    "QuantumBoltzmannMachine": 27,
    "TreeTensorClassifier": 3,
    "IQPKernelClassifier": 12,
    "ProjectedQuantumKernel": 108,
    "QuantumKitchenSinks": 8,
    "WeiNet": 9,
    "MLPClassifier": 27,
    "SVC": 16,
    "ConvolutionalNeuralNetwork": 9,
    "SeparableVariationalClassifier": 12,
    "SeparableKernelClassifier": 16,
}


@pytest.mark.parametrize(("name", "size"), sorted(PAPER_GRID_SIZES.items()))
def test_grid_sizes_match_paper(name, size):
    assert len(expand_grid(model_grid(name))) == size


def test_variants_reuse_parent_grid():
    assert model_grid("DataReuploadingClassifierNoCZ") == model_grid(
        "DataReuploadingClassifier"
    )
    assert model_grid("QuantumBoltzmannMachineIndependentZZ") == model_grid(
        "QuantumBoltzmannMachine"
    )
    assert model_grid("WeiNetCoherentFilter") == model_grid("WeiNet")


def test_mlp_hidden_layer_sizes_are_tuples():
    assert set(model_grid("MLPClassifier")["hidden_layer_sizes"]) == {
        (100,),
        (10, 10, 10, 10),
        (50, 10, 5),
    }


@pytest.mark.parametrize("name", ALL_MODEL_NAMES)
def test_every_registered_model_builds(name):
    grid_point = expand_grid(model_grid(name))[0]
    assert build_model(name, grid_point, random_state=0) is not None


def _toy_data(n_features=4, n_samples=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    return X, np.where(X[:, 0] + X[:, 1] > 0, 1, -1)


def test_drc_no_cz_has_no_entangling_gates_but_keeps_both_rotations():
    X, y = _toy_data(n_features=6)
    model = build_model(
        "DataReuploadingClassifierNoCZ", {"n_layers": 2, "max_steps": 1}, 0
    )
    model.initialize(6)
    model.scaler = None
    tape = qml.workflow.construct_tape(model.circuit)(
        model.params_, model.transform(X)[0]
    )
    names = [op.name for op in tape.operations]
    assert "CZ" not in names
    # 2 qubits x (2 layers x 2 Rot + final Rot) = 10 rotations, as in the parent without CZs.
    assert names.count("Rot") == 10
    parent = build_model(
        "DataReuploadingClassifier", {"n_layers": 2, "max_steps": 1}, 0
    )
    parent.initialize(6)
    parent.scaler = None
    parent_names = [
        op.name
        for op in qml.workflow.construct_tape(parent.circuit)(
            parent.params_, parent.transform(X)[0]
        ).operations
    ]
    assert [n for n in parent_names if n != "CZ"] == names
    assert "CZ" in parent_names


def _qbm_zz_gradients(name):
    X, y = _toy_data(n_features=3)
    model = build_model(name, {"visible_qubits": "single"}, 0)
    model.initialize(3, classes=[-1, 1])
    model.scaler = None
    Xs = model.transform(X)
    grads = jax.grad(lambda t: jnp.sum(model.forward(t, Xs)))(model.params_["thetas"])
    return np.asarray(grads[2 * 3 :])  # rows reserved for the 3 ZZ couplings


def test_released_qbm_uses_only_the_first_zz_row_and_corrected_uses_all():
    released = _qbm_zz_gradients("QuantumBoltzmannMachine")
    corrected = _qbm_zz_gradients("QuantumBoltzmannMachineIndependentZZ")
    assert np.abs(released[0]).sum() > 0 and np.allclose(released[1:], 0.0)
    assert all(np.abs(row).sum() > 0 for row in corrected)


def test_coherent_weinet_readout_matches_pennylane_expectations():
    model = build_model("WeiNetCoherentFilter", {"filter_name": "edge_detect"}, 0)
    model.initialize(16, classes=[-1, 1])
    rng = np.random.default_rng(1)
    image = rng.normal(size=16)
    state = np.asarray(model.filter_map_ @ image)
    state = state / np.linalg.norm(state)

    dev = qml.device("default.qubit", wires=4)
    from better_than_classical.lib.model_variants import _weinet_readout_operators

    @qml.qnode(dev)
    def expectations():
        qml.StatePrep(state, wires=range(4))
        return [qml.expval(op) for op in _weinet_readout_operators(4)]

    expected = np.array(expectations())
    got = np.asarray(model.readout_signs_ @ (state**2))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_bound_weinet_circuits_differ_while_released_ones_coincide():
    X, _ = _toy_data(n_features=16, n_samples=4)
    released = build_model("WeiNet", {}, 0)
    released.initialize(16, classes=[-1, 1])
    bound = build_model("WeiNetBoundCircuits", {}, 0)
    bound.initialize(16, classes=[-1, 1])
    x = released.transform(X)[0]
    released_outputs = [np.asarray(c(x)) for c in released.circuits]
    bound_outputs = [np.asarray(c(x)) for c in bound.circuits]
    assert all(np.allclose(o, released_outputs[0]) for o in released_outputs)
    assert not all(np.allclose(o, bound_outputs[0]) for o in bound_outputs)
