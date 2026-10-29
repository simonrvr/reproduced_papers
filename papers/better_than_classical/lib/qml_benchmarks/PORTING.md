# Vendored qml-benchmarks code

Source: https://github.com/XanaduAI/qml-benchmarks, commit
`95e5a07e8e9e75ba7e24e67fb32b030112a1309a` (2025-08-22), Apache License 2.0
(`LICENSE` in this folder). Copyright 2024 Xanadu Quantum Technologies Inc.
Local snapshot: `data/better_than_classical/sources/qml-benchmarks/`.

Only the files needed by the local staged plan were copied (`src/qml_benchmarks/`):
`model_utils.py`, `hyperparameter_settings.py`, and `models/` `circuit_centric`,
`data_reuploading`, `dressed_quantum_circuit`, `iqp_variational`, `tree_tensor`,
`separable`, `iqp_kernel`, `projected_quantum_kernel`, `quantum_kitchen_sinks`,
`quantum_boltzmann_machine`, `weinet`, `convolutional_neural_network`.
Not copied: `QuantumMetricLearner`, `QuanvolutionalNeuralNetwork`,
`VanillaQNN`, energy-based models, data generators (the authors' hosted
datasets are used instead).

## Changes

The circuits, losses, initialisations, scalers, optimiser and convergence
rule are unchanged. Every change is listed here; regenerate the diff with
`diff -r data/better_than_classical/sources/qml-benchmarks/src/qml_benchmarks papers/better_than_classical/lib/qml_benchmarks`.

| File | Change | Why | Effect on results |
|---|---|---|---|
| all copied files | `from qml_benchmarks.x import` -> relative imports; header comment pointing here | package lives inside `lib/` | none |
| `model_utils.py` | `tqdm` progress bar replaced by `logging` (DEBUG line every 200 steps); `print` of the loss history replaced by a DEBUG log; `model.n_steps_` and `model.converged_` recorded (`None` when the convergence check is disabled) before the unchanged `ConvergenceWarning` | run evidence for the logging contract; no `print` | none |
| `compat.py` (new) | `broadcast()` re-implements the `qml.broadcast` wire patterns used here (`double`, `double_odd`, `chain`, `ring`, explicit lists), copied from PennyLane 0.39 `templates/broadcast.py` | `qml.broadcast` was removed in PennyLane 0.40 | none (same gate sequence; tested in `tests/test_compat.py`) |
| `compat.py` (new), `__init__.py` | `restore_jax_core_aliases()` sets `jax.core.is_concrete` / `jax.core.concrete_or_error` to `jax._src.core` functions | PennyLane 0.45.1 still calls these names, removed in JAX 0.11 (`AmplitudeEmbedding`, state preparation fail otherwise). TODO: remove once PennyLane stops using them | none (the removed names were aliases of these functions) |
| `data_reuploading.py`, `dressed_quantum_circuit.py`, `tree_tensor.py`, `quantum_kitchen_sinks.py` | `qml.broadcast(...)` -> `compat.broadcast(...)` | as above | none |
| `quantum_kitchen_sinks.py` | device `shots=1, prng_key=key` -> `seed=key` on the device and `shots=1` on the QNode | PennyLane >= 0.40 device API | sampling distribution unchanged; the JAX random stream may differ from PennyLane 0.34 |
| `separable.py` (SeparableKernelClassifier), `iqp_kernel.py`, `projected_quantum_kernel.py`, `quantum_kitchen_sinks.py` | default `qnode_kwargs` drop `"diff_method": None` (added 2026-10-22, Phase 03 / S2) | PennyLane 0.45 runs `diff_method=None` QNodes through a host callback inside `jax.jit`: 11-27 ms per call instead of ~0.03 ms, which stalled the S2 kitchen-sinks and separable-kernel cells for hours | none: kernel matrices / sampled features identical (exact for SeparableKernel, IQPKernel and QuantumKitchenSinks; within 2.2e-16 for ProjectedQuantumKernel), `tests/test_kernel_qnode_port.py` |
| `quantum_boltzmann_machine.py` | docstring made a raw string | invalid `\p` escape warning | none |

Known issues of the released models are **kept** here, so released behaviour
is reproduced; the corrected or controlled versions live in
`lib/model_variants.py` (see LOG.md, "Reference-code issues"):
WeiNet's inert filter and late-bound circuits, the QBM's shared ZZ parameter,
and `DataReuploadingClassifierSeparable` changing the embedding as well as
removing CZs.

Deprecation warnings that remain (not errors on the installed scikit-learn
1.9): `SVC(probability=...)` (removed in 1.11) in the kernel models and
`LogisticRegression(penalty=None)` (removed in 1.10) in QuantumKitchenSinks.
