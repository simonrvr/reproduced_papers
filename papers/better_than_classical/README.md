# Better than classical? The subtle art of benchmarking quantum machine learning models - Reproduction

> Status: scaffold only. The benchmark pipeline is not implemented yet, and no
> results have been produced. See `LOG.md` for the current plan and state.

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

The exact model list, dataset family, metrics and claims will be listed in
`LOG.md` during Phase 01 (paper analysis).

## Reproduction Scope (including Updates and Deviations)

TODO (Phase 01): list the reproduced datasets, models, figures, tables, metrics
and seeds, plus every deviation from the paper (for example, a reduced dataset
or model subset, a smaller hyperparameter grid, or a MerLin photonic
adaptation).

## Project Layout

```text
papers/better_than_classical/
|-- README.md          # this file
|-- LOG.md             # reproduction log: claims, plan, decisions, handoff
|-- requirements.txt
|-- notebook.ipynb
|-- cli.json           # paper-specific CLI schema
|-- configs/           # defaults.json and named experiments (smoke.json, ...)
|-- lib/               # runner.py and paper modules
|-- tests/
|-- utils/
|-- models/
|-- results/           # curated figures and summaries
`-- outdir/            # raw timestamped runs (not committed)
```

## Install and How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r papers/better_than_classical/requirements.txt
```

The repository-level `implementation.py` is the entry point. `cli.json` is the
authoritative list of paper-specific flags. Global flags (`--config`,
`--outdir`, `--seed`, `--dtype`, `--device`, `--log-level`) come from the
shared runtime.

Smoke run (checks runtime wiring only, for now):

```bash
# From the repo root
python implementation.py --paper better_than_classical --config configs/smoke.json

# From inside papers/better_than_classical
python ../../implementation.py --config configs/smoke.json
```

Full paper reproduction commands: TODO (Phase 02/03).

Each run writes `outdir/run_YYYYMMDD-HHMMSS/` with `config_snapshot.json`,
`run.log` and the run artifacts.

## Configuration

- `configs/defaults.json`: default, lightweight config.
- `configs/smoke.json`: one-epoch smoke run.
- `cli.json`: `--epochs`, `--batch-size`, `--lr` (scaffold flags; Phase 02 will
  update them to match the benchmark pipeline).

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

No results yet.

## Limitations

TODO: filled once the reproduction scope is fixed.

## Tests

```bash
cd papers/better_than_classical
pytest -q
```

## Citation and License

```bibtex
@article{bowles2024better,
  title   = {Better than classical? The subtle art of benchmarking quantum machine learning models},
  author  = {Bowles, Joseph and Ahmed, Shahnawaz and Schuld, Maria},
  journal = {arXiv preprint arXiv:2403.07059},
  year    = {2024}
}
```
