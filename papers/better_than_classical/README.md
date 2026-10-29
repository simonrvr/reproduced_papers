# Better than classical? The subtle art of benchmarking quantum machine learning models - Reproduction

> Status: implementation complete and tested (Phase 02); the reduced
> reproduction runs (stages S1-S7) have not been run or analysed yet, so no
> results are reported. See `LOG.md` for the plan and current state.

## Reference and Attribution

- Paper: Better than classical? The subtle art of benchmarking quantum machine
  learning models (arXiv preprint, 2024; v2 dated 14 Mar 2024)
- Authors: Joseph Bowles, Shahnawaz Ahmed, Maria Schuld
- arXiv: https://arxiv.org/abs/2403.07059 (PDF: https://arxiv.org/pdf/2403.07059)
- Original repository: https://github.com/XanaduAI/qml-benchmarks (to be
  confirmed in Phase 01, including its license)
- License and attribution notes: this reproduction is an independent
  reimplementation. Any code or data reused from the original repository will
  keep its license notice and be credited here.

## Original Paper

The paper asks how to benchmark quantum machine learning (QML) classifiers
fairly against classical ones. It runs a large-scale study of popular
near-term quantum classifiers (variational, kernel-based and convolutional
designs) and classical baselines on binary classification tasks. The datasets
are mostly synthetic and controllable (for example, linearly separable data,
hidden-manifold data, two-curves data and bars-and-stripes), plus
MNIST-derived tasks. Each model is tuned by hyperparameter search and judged
on test accuracy. The main reported findings are that out-of-the-box classical
models usually beat the quantum models, and that removing entanglement from
the quantum models often does not lower their scores. The paper presents
these results as a case study of how hard it is to benchmark QML well.

The study tests 12 quantum models from 11 influential papers (6 quantum
neural networks, 3 kernel methods, 2 quantum CNNs, plus separable
no-entanglement controls) against `MLPClassifier`, RBF `SVC` and a small CNN
on 10 benchmarks (160 datasets), tuning every model by 5-fold cross-validated
grid search and reporting test accuracy averaged over 5 retrainings. The
claim and experiment inventories are in `LOG.md`.

## Reproduction Scope (including Updates and Deviations)

The full study cost about 27 core-years on a cluster; this reproduction is a
**reduced, local** one (8 CPU cores), run in stages ordered by evidence per
compute hour (`LOG.md`, "Local staged plan"):

| Stage | What | Paper reference |
|---|---|---|
| S1 | Re-analysis of the authors' released per-dataset results (rankings, accuracy curves, ablations, grid ranges and correlations) and the positivity-bias simulation. Analysis of their numbers only, no training. | Figs. 3, 7, 8, 10-13, 15, 20-24 |
| S2 | Port validation: retrain with the authors' best hyperparameters and compare with their released test accuracies. | - |
| S3 | QNN family on mnist pca and kernel family on mnist pca-, d in {2, 4, 6, 8}, with the paper's full grids. | Figs. 10-12 |
| S4 | Entanglement removal for DressedQC, DataReuploading (d in {2, 4, 6, 8}) and QBM (d in {2, 4}). | Fig. 13 |
| S5 | Both families on linearly separable data, d in {2, 4, 6, 8}. | Fig. 11 |
| S6 | DataReuploading ablation. | Fig. 15 |
| S7 | WeiNet vs CNN on bars & stripes (4x4, 8x8). | Figs. 10, 11 |

Same protocol as the paper: the authors' own data (PennyLane-hosted) and
model code, the full hyperparameter grids of Appendix C, 5-fold stratified CV
on accuracy, the paper's convergence rule, retraining with seeds 0-4.

Updates and deviations:

- **Reduced scope**: not reproduced locally are QuantumMetricLearner,
  QBM beyond d = 4, QuanvolutionalNeuralNetwork, MNIST-CG, the hidden
  manifold / two curves / hyperplanes benchmarks and d > 8 (S1 covers them
  from the released results only).
- **Ported code**: the authors' models run on PennyLane 0.45 / JAX 0.11 /
  scikit-learn 1.9 with minimal compatibility changes
  (`lib/qml_benchmarks/PORTING.md`).
- **Added baselines and controls**: a LogisticRegression floor in each
  family; DataReuploading with only the CZs removed; QBM with independent ZZ
  couplings; two WeiNet diagnostics (`lib/model_variants.py`), because of
  issues found in the released code (`LOG.md`, "Reference-code issues").
- **mnist pca-**: one seeded 250/250 subsample without replacement, shared
  by all d (the released sample is unseeded, with replacement, per d).
- **Seeds**: MLP and SVC are seeded during the search (unseeded in the
  released code).
- No MerLin / photonic adaptation yet (Phase 04).

## Project Layout

```text
papers/better_than_classical/
|-- README.md          # this file
|-- LOG.md             # reproduction log: claims, plan, decisions, handoff
|-- requirements.txt
|-- notebook.ipynb
|-- cli.json           # paper-specific CLI schema
|-- configs/           # defaults.json, smoke_reduced.json, s1_* ... s7_* (one per stage and variant)
|-- lib/
|   |-- runner.py      # train_and_evaluate: grid_search / score_best / reanalysis / positivity_bias
|   |-- grid_search.py # 5-fold CV grid search, selection, retraining with seeds 0-4
|   |-- data.py        # hosted datasets (HDF5 -> .npz cache), mnist pca- samples
|   |-- models.py, hyperparameters.py, model_variants.py
|   |-- reanalysis.py, positivity_bias.py, experiment_logging.py
|   `-- qml_benchmarks/ # authors' models, vendored (Apache-2.0, PORTING.md)
|-- tests/             # 71 tests
|-- utils/             # check_data_loader.py, estimate_grid_cost.py
|-- models/
|-- results/           # curated figures and summaries
`-- outdir/            # raw timestamped runs (not committed)
```

## Install and How to Run

```bash
pip install -r papers/better_than_classical/requirements.txt
```

The repository-level `implementation.py` is the entry point. `cli.json` is the
authoritative list of paper-specific flags (`--model`, `--dataset`,
`--dataset-param`, `--sweep-id`). Global flags (`--config`, `--outdir`,
`--seed`, `--dtype`, `--device`, `--log-level`, `--data-root`) come from the
shared runtime. One run executes one experiment: a single (model, dataset)
grid-search cell, a port-validation cell, the re-analysis, or the
positivity-bias simulation.

Smoke run (data re-uploading on mnist pca d = 2, one-point grid, about 1.5 min):

```bash
# From the repo root
python implementation.py --paper better_than_classical --config configs/smoke_reduced.json

# From inside papers/better_than_classical
python ../../implementation.py --config configs/smoke_reduced.json
```

Stage S1 (minutes, no training):

```bash
python implementation.py --paper better_than_classical --config configs/s1_reanalysis_original.json
python implementation.py --paper better_than_classical --config configs/s1_positivity_bias_original.json
```

One grid-search cell of the reduced reproduction (the paper's full grid,
5-fold CV, retraining with seeds 0-4), for example:

```bash
python implementation.py --paper better_than_classical \
  --config configs/s3_qnn_mnist_pca_original.json \
  --model DataReuploadingClassifier --dataset-param 6
```

Each grid-search config lists its cells in a `sweep` block (models x dataset
parameters). A sweep coordinator that runs all cells in parallel is added in
Phase 03; until then cells are run one by one as above.

Each run writes `outdir/run_YYYYMMDD-HHMMSS/` with `config_snapshot.json`,
`run.log`, `run_status.json`, `metrics.json` and experiment tables
(`cv_results.csv`, `scores.csv`, `best_hyperparameters.json` for grid
searches).

## Configuration

| Config | Stage | What it runs |
|---|---|---|
| `defaults.json` | - | Lightweight default: SVC on linearly separable d = 2 (seconds). |
| `smoke_reduced.json` | smoke | DataReuploading, mnist pca d = 2, one grid point, shortened convergence (not a paper setting). |
| `s1_reanalysis_original.json` | S1 | Re-analysis of the authors' released results (Figs. 7, 8, 10-13, 15, 20-24). |
| `s1_positivity_bias_original.json` | S1 | Fig. 3 simulation over 1000 seeds. |
| `s2_port_validation_mnist_pca.json`, `s2_port_validation_mnist_pca_minus.json` | S2 | Retrain with the authors' best hyperparameters and compare with their released test accuracies. |
| `s3_qnn_mnist_pca_{original,classical}.json`, `s3_kernel_mnist_pca_minus_{original,classical}.json` | S3 | Main comparison, d in {2, 4, 6, 8}. |
| `s4_entanglement_*` | S4 | Entanglement removal (Fig. 13) and the added controls. |
| `s5_{qnn,kernel}_linearly_separable_{original,classical}.json` | S5 | Main comparison on linearly separable data. |
| `s6_drc_ablation_mnist_pca_original.json` | S6 | Data re-uploading ablation (Fig. 15). |
| `s7_weinet_*` | S7 | WeiNet vs CNN on bars & stripes, with two diagnostic WeiNet variants. |

`_original` configs contain models of the original study, `_classical` the
classical baselines (with the LogisticRegression floor added here), and
`controls` the variants added by this reproduction
(`lib/model_variants.py`). The models are the authors' code, vendored in
`lib/qml_benchmarks/` (Apache-2.0); `lib/qml_benchmarks/PORTING.md` lists
every change needed for current PennyLane / JAX / scikit-learn.

## Data

Data lives under the shared data root, `data/better_than_classical/`
(`dataset.root: null` resolves to it). It is not committed (gitignored).
Acquired and verified on 2026-10-19; details and checksums are in `LOG.md`
(Data Acquisition Log).

| Benchmark(s) | Source | Local path |
|---|---|---|
| linearly separable, hidden manifold (+ diff), two curves (+ diff), hyperplanes diff, bars & stripes, mnist pca | Authors' datasets on [PennyLane Datasets](https://pennylane.ai/datasets/) (CC BY-SA 4.0), HDF5 in `qml.data` format | `data/better_than_classical/pennylane_datasets/*.h5` |
| mnist cg, mnist pca- (derived in Phase 02) | Raw MNIST, torchvision mirror | `data/better_than_classical/mnist/MNIST/raw/` |

All files are the original data. The hosted sets match the authors'
committed example CSVs exactly. mnist cg and mnist pca- are not hosted and
are derived from raw MNIST with the reference preprocessing. mnist pca-
used an unseeded subsample in the original code, so only its d = 2 and
d = 10 versions can be matched exactly (from the authors' committed CSVs).

Reading the HDF5 files needs `h5py` (`python3 -m pip install h5py`).
To download again (from the repository root):

```bash
D=data/better_than_classical/pennylane_datasets; mkdir -p "$D"
for n in LinearlySeparable Hyperplanes HiddenManifold TwoCurves BarsAndStripes MNISTPCA; do
  curl -f -C - -o "$D/${n}_${n}.h5" \
    "https://datasets.cloud.pennylane.ai/datasets/h5/other/$n/${n}_${n}.h5"
done
M=data/better_than_classical/mnist/MNIST/raw; mkdir -p "$M"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -f -o "$M/$f.gz" "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz"
done
```

## Results Obtained and Comparison with the Paper

Stages are run one at a time; only S1 is done so far. Details, commands and
run directories are in `LOG.md` ("Phase 03 Stage Results").

**S1: re-analysis of the authors' released results** (scope: analysis of
their numbers, not independent evidence) and the Fig. 3 simulation.

- The rankings of Figs. 10 and 24 are recovered exactly from the released
  per-dataset results: MLP, SVC and CNN come first in their families; with
  separable models, SeparableKernel is 2nd of 5 and SeparableVariational 5th
  of 9 ([`results/s1_rankings.png`](results/s1_rankings.png)).
- Added paired tests over datasets: MLP beats every quantum neural network
  (p < 0.01), but by only +0.021 mean accuracy over DressedQC; SVC beats the
  projected quantum kernel by +0.014. The separable models tie
  DataReuploading, QBM and the projected kernel
  ([`results/s1_paired_differences.png`](results/s1_paired_differences.png)).
- Entanglement removal and the DataReuploading ablation (Figs. 13, 15)
  match the paper
  ([`results/s1_entanglement_and_drc_ablation.png`](results/s1_entanglement_and_drc_ablation.png)).
- Fig. 8 is computed with Spearman correlation in the released script (the
  text says Pearson), and the sign of categorical hyperparameters is
  arbitrary ([`results/s1_hyperparameter_correlations.png`](results/s1_hyperparameter_correlations.png)).
- Positivity bias: published means 0.737 (quantum) vs 0.650 (classical),
  paper 0.74 vs 0.65 ([`results/s1_positivity_bias.png`](results/s1_positivity_bias.png)).

Independent re-runs (S2-S7) are pending.

## Limitations

- Reduced scope (see above): at most d = 8 and two benchmarks are re-run
  locally, so conclusions about high dimensions and the other benchmarks
  rest on the authors' released numbers (S1), not on independent runs.
- One data realisation per dataset, as in the paper; differences smaller
  than the spread over the 5 retrain seeds and the small test sets (60
  points for linearly separable) are not interpretable.
- Final results and their limitations are added after Phase 03.

## Tests

```bash
cd papers/better_than_classical
python3 -m pytest -q
```

The suite (71 tests) checks the qml.broadcast replacement against PennyLane
0.39 patterns, the grid sizes of the paper's Appendix C, that each control
variant changes only what it claims, that the CV loop equals scikit-learn's
`GridSearchCV`, the ranking rule, the positivity-bias simulation, the data
loader and an end-to-end default run. Data-dependent tests are skipped when
the acquired datasets are absent.

## Citation and License

```bibtex
@article{bowles2024better,
  title   = {Better than classical? The subtle art of benchmarking quantum machine learning models},
  author  = {Bowles, Joseph and Ahmed, Shahnawaz and Schuld, Maria},
  journal = {arXiv preprint arXiv:2403.07059},
  year    = {2024}
}
```
