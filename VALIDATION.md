# Release validation

Executed locally on 2026-11-17 on Linux CPU.

Environment: Python 3.12.3; pytest 9.2.0; Ruff 0.17.5; PyTorch 2.15.0+cpu; NumPy 2.5.5.

- Editable installation succeeded.
- Ruff lint and format checks passed.
- **5 tests passed**, including a command-line workflow test that creates its own data/artifacts.
- Wheel and source-distribution builds succeeded.
- The experiments below ran successfully; raw evaluation output is preserved in [examples/release-report.json](examples/release-report.json).

## Exploratory run

Trained on 128 synthetic point clouds, 12 epochs, seed 5; evaluated 64 newly generated examples with seed 101.

| Metric | Model | Baseline |
|---|---:|---:|
| Masked coordinate Euclidean RMSE | 2.033470 | 2.033515 (visible centroid) |
| Scalar property MAE | 0.029943 | 0.298275 (training mean) |

The property head learns the synthetic target. Coordinate reconstruction is effectively at the centroid baseline; independently sampled hidden coordinates provide little predictable geometric structure. This is not an evaluation on real molecules.

## Reproduction and scope

Use the README commands, matching the seeds and sizes above. AstroBridge's final evaluation used seed 202. Exact floating-point values can differ across dependency versions and hardware. Generated training checkpoints remain local under ignored `runs/` directories; they are not bundled in the repository or Python distribution.

The CI workflow validates Python 3.10 and 3.12 independently; its current result is visible in [GitHub Actions](https://github.com/DIVYANSH-675/MoleculeMask/actions). No GPU benchmark, large external dataset, or paper-result reproduction was run. There is no claim that these development experiments establish general research superiority.
