"""Model registry: names used in configs and released results -> estimators.

Quantum models come from the vendored qml-benchmarks code
(``lib/qml_benchmarks``) or from the control variants in
``lib/model_variants.py``. The classical baselines are plain scikit-learn
estimators configured as in qml-benchmarks ``models/__init__.py``
(``MLPClassifier(max_iter=3000)``, ``SVC(kernel="rbf")``), plus the
``LogisticRegression`` floor added by this reproduction.
"""

from __future__ import annotations

import ast
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .model_variants import (
    DataReuploadingClassifierNoCZ,
    QuantumBoltzmannMachineIndependentZZ,
    WeiNetBoundCircuits,
    WeiNetCoherentFilter,
)
from .qml_benchmarks.models.circuit_centric import CircuitCentricClassifier
from .qml_benchmarks.models.convolutional_neural_network import (
    ConvolutionalNeuralNetwork,
)
from .qml_benchmarks.models.data_reuploading import (
    DataReuploadingClassifier,
    DataReuploadingClassifierNoCost,
    DataReuploadingClassifierNoScaling,
    DataReuploadingClassifierNoTrainableEmbedding,
    DataReuploadingClassifierSeparable,
)
from .qml_benchmarks.models.dressed_quantum_circuit import (
    DressedQuantumCircuitClassifier,
    DressedQuantumCircuitClassifierSeparable,
)
from .qml_benchmarks.models.iqp_kernel import IQPKernelClassifier
from .qml_benchmarks.models.iqp_variational import IQPVariationalClassifier
from .qml_benchmarks.models.projected_quantum_kernel import ProjectedQuantumKernel
from .qml_benchmarks.models.quantum_boltzmann_machine import (
    QuantumBoltzmannMachine,
    QuantumBoltzmannMachineSeparable,
)
from .qml_benchmarks.models.quantum_kitchen_sinks import QuantumKitchenSinks
from .qml_benchmarks.models.separable import (
    SeparableKernelClassifier,
    SeparableVariationalClassifier,
)
from .qml_benchmarks.models.tree_tensor import TreeTensorClassifier
from .qml_benchmarks.models.weinet import WeiNet

QUANTUM_MODEL_CLASSES = {
    cls.__name__: cls
    for cls in (
        CircuitCentricClassifier,
        DataReuploadingClassifier,
        DataReuploadingClassifierNoCost,
        DataReuploadingClassifierNoScaling,
        DataReuploadingClassifierNoTrainableEmbedding,
        DataReuploadingClassifierSeparable,
        DataReuploadingClassifierNoCZ,
        DressedQuantumCircuitClassifier,
        DressedQuantumCircuitClassifierSeparable,
        IQPVariationalClassifier,
        TreeTensorClassifier,
        QuantumBoltzmannMachine,
        QuantumBoltzmannMachineSeparable,
        QuantumBoltzmannMachineIndependentZZ,
        SeparableVariationalClassifier,
        IQPKernelClassifier,
        ProjectedQuantumKernel,
        QuantumKitchenSinks,
        SeparableKernelClassifier,
        WeiNet,
        WeiNetBoundCircuits,
        WeiNetCoherentFilter,
        ConvolutionalNeuralNetwork,
    )
}

CLASSICAL_MODEL_NAMES = ("MLPClassifier", "SVC", "LogisticRegression")

# Models added by this reproduction; everything else is in the original study.
ADDED_MODEL_NAMES = (
    "LogisticRegression",
    "DataReuploadingClassifierNoCZ",
    "QuantumBoltzmannMachineIndependentZZ",
    "WeiNetBoundCircuits",
    "WeiNetCoherentFilter",
)

# Model families used for ranking (paper Fig. 10 / 24, plot_rankings_total.py).
FAMILIES = {
    "qnn": [
        "MLPClassifier",
        "CircuitCentricClassifier",
        "DataReuploadingClassifier",
        "DressedQuantumCircuitClassifier",
        "IQPVariationalClassifier",
        "QuantumMetricLearner",
        "QuantumBoltzmannMachine",
        "TreeTensorClassifier",
    ],
    "kernel": [
        "IQPKernelClassifier",
        "ProjectedQuantumKernel",
        "QuantumKitchenSinks",
        "SVC",
    ],
    "cnn": ["ConvolutionalNeuralNetwork", "WeiNet", "QuanvolutionalNeuralNetwork"],
}
SEPARABLE_BY_FAMILY = {
    "qnn": "SeparableVariationalClassifier",
    "kernel": "SeparableKernelClassifier",
}

ALL_MODEL_NAMES = tuple(QUANTUM_MODEL_CLASSES) + CLASSICAL_MODEL_NAMES


def parse_hyperparameter_value(name: str, value: Any) -> Any:
    """Convert a hyperparameter value read from a CSV or grid into its Python type.

    ``hidden_layer_sizes`` is stored as a string such as ``"(100,)"`` in the
    released grids and best-hyperparameter files; it must be a tuple for
    scikit-learn. Other values are returned unchanged.
    """
    if name == "hidden_layer_sizes" and isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, tuple):
            raise ValueError(f"hidden_layer_sizes must parse to a tuple, got {value!r}")
        return parsed
    return value


def build_model(name: str, params: dict[str, Any], random_state: int):
    """Instantiate a model by name with hyperparameters and a seed.

    Parameters
    ----------
    name : str
        Registry name (class name as used in the released results).
    params : dict
        Hyperparameters for the estimator constructor.
    random_state : int
        Seed passed as ``random_state`` (model initialisation and, for the
        quantum models, JAX keys).

    Returns
    -------
    sklearn.base.BaseEstimator
        Unfitted estimator.

    Raises
    ------
    KeyError
        If ``name`` is not a registered model.
    """
    params = {key: parse_hyperparameter_value(key, val) for key, val in params.items()}
    if name in QUANTUM_MODEL_CLASSES:
        return QUANTUM_MODEL_CLASSES[name](**params, random_state=random_state)
    if name == "MLPClassifier":
        return MLPClassifier(**{"max_iter": 3000, **params}, random_state=random_state)
    if name == "SVC":
        return SVC(kernel="rbf", **params, random_state=random_state)
    if name == "LogisticRegression":
        return LogisticRegression(
            **{"max_iter": 10000, **params}, random_state=random_state
        )
    raise KeyError(f"Unknown model {name!r}; known: {sorted(ALL_MODEL_NAMES)}")
