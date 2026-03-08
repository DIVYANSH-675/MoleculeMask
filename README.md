# MoleculeMask

Equivariant masked-coordinate and scalar-property learning on molecular point clouds.

MoleculeMask is an independently implemented geometric message-passing model. It predicts withheld atom coordinates and a scalar graph property while respecting rotations, reflections, translations, and atom permutations.

## Status and provenance

Version 0.1.0 is an original, AI-assisted implementation developed for **Divyansh Gupta**. This repository was written independently: no upstream source files or Git history were imported. Its scientific inspiration is [EMPP](https://arxiv.org/abs/2502.08209); the research ideas remain credited to that paper's authors. The implementation choices and limitations below describe this repository's scope. It is an experimental prototype, not a claimed reproduction of published results.

## Setup

Python 3.10 or newer:

```bash
python -m venv .venv
# Linux/macOS; on Windows use .venv\Scripts\activate
source .venv/bin/activate
# Install CPU-only PyTorch first (omit this line to choose another build).
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

## Run

```bash
moleculemask train --count 128 --epochs 12 --seed 5 --output runs/train
moleculemask evaluate --checkpoint runs/train/model.pt --count 64 --seed 101 --output runs/evaluation.json
# Or provide a compatible dataset:
moleculemask generate --count 128 --seed 6 --output runs/molecules.npz
moleculemask train --data runs/molecules.npz --epochs 12 --output runs/custom
```

Use fresh output paths for new experiments. Training artifacts and generated data are ignored by Git. For independent evaluation, use a different generation seed or a separate dataset; evaluating on the training data is not an unbiased test.

## Implementation

Masked coordinates are initialized from the centroid of **visible atoms only**. Scalar messages depend on atom features and squared pair distances. Coordinate updates are sums of relative vectors weighted by invariant scalar messages; only masked positions move. A mean-pooled scalar head predicts a graph property. Training combines masked-coordinate squared error and a normalized property loss.

The generator produces six-atom Gaussian point clouds with atomic numbers sampled from H/C/N/O and an explicitly synthetic property. These are geometric fixtures, not chemically valid molecules. Training reserves 20% of examples for validation and fits the property normalizer on training targets only. Validation compares coordinate error with a visible-centroid baseline, and property MAE with the training-mean baseline.

## Data and artifacts

Training writes `config.json`, `split.json`, epoch-level `metrics.json`, and `model.pt`. Evaluation writes masked-atom Euclidean RMSE, scalar-property MAE, and both baselines. Input NPZ keys: `positions: [B,N,3]`, `atomic_numbers: [B,N]` with integer elements 1..118, and `properties: [B]`. B must be at least 5 and N at least 2; each batch has a fixed atom count. The CLI masks one atom per molecule; the model supports multiple masks when at least one atom is visible. All commands run on CPU.

## Validation

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

The test suite covers: Rigid-motion and atom-permutation equivariance; invariant scalar properties; withheld-coordinate leakage prevention; singleton-batch backward pass; full train/evaluate and NPZ round trip.

CI runs the suite on Python 3.10 and 3.12. Small synthetic runs validate software behavior; they do not establish a research result. See [VALIDATION.md](VALIDATION.md) for the release checks actually performed.

## Limitations

This implementation uses point reconstruction rather than EMPP's published position-distribution objective. It is not a reproduction of the paper's architecture or benchmark results. No QM9/MD17 results, chemical validity, force prediction, or energy conservation are claimed. Fixed atom counts, dense quadratic message passing, and multimodal missing coordinates limit this prototype. Synthetic masked coordinates are largely unpredictable; strong symmetry tests do not imply accurate chemistry.

## Preparing numeric molecular data

Positions and properties must be real numeric arrays. The loader converts them to float32 and verifies that the converted values remain finite. A finite float64 value such as `1e100` cannot be used directly: it overflows float32 and now raises a descriptive `ValueError` before training.

Keep units consistent across training and evaluation. If your physical units produce very large values, choose a scientifically meaningful conversion before saving the NPZ file, and record that conversion alongside your dataset. The loader does not silently clip values or change physical units.

Atomic numbers remain integer arrays in the range 1..118. The existing model expects a fixed number of atoms per example; padding is not interpreted as a missing atom.

## License

The original code in this repository is MIT licensed, copyright 2026 Divyansh Gupta. See [LICENSE](LICENSE). The cited paper is a separate work; this code's license does not relicense the paper or external datasets.


## Reproducibility and report utilities

The standard-library artifact CLI can inspect saved experiments without loading a model. The following commands use a small illustrative fixture, not benchmark results:

```bash
python -m moleculemask.artifact_cli manifest --inputs examples/metrics-sample.json --base . --metadata '{"purpose":"example"}' --output runs/manifest.json
python -m moleculemask.artifact_cli verify --input runs/manifest.json --base . --output runs/verification.json
python -m moleculemask.artifact_cli summary --input examples/metrics-sample.json --output runs/summary.json
python -m moleculemask.artifact_cli csv --input examples/metrics-sample.json --output runs/metrics.csv
python -m moleculemask.artifact_cli compare --before examples/metrics-sample.json --after examples/metrics-sample.json --output runs/comparison.json
python -m moleculemask.artifact_cli bootstrap --input examples/metrics-sample.json --key value --samples 1000 --seed 7 --output runs/interval.json
```

The installed `moleculemask-artifacts` command offers the same interface. Use fresh output filenames: existing artifacts are never overwritten. Manifests store SHA-256 hashes and paths relative to `--base`; keep input files fixed while creating or verifying a manifest. Verification exits with status 1 for changed or missing inputs, and status 2 for invalid arguments or manifests.

Metric summaries accept JSON arrays, single objects, reports containing `rows` or `episodes`, and streamed JSONL. They summarize numeric fields with population standard deviation and ignore boolean/nonnumeric fields. CSV export accepts flat scalar records; nested data must be transformed first. Comparisons report signed changes, without assuming whether larger values are better. Relative change is null when the baseline is zero.

Bootstrap intervals assume independent observations. Correlated training steps or snapshots from the same run are not independent experimental repetitions; aggregate by run before interpreting an interval. These intervals do not calibrate model uncertainty.
