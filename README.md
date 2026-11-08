# QCML: source code and test inputs

Repository: [GIT_REPOSITORY_URL]

This distribution contains computational source code, source configuration, and input-only test examples. The test arrays were exported from the existing held-out splits without copying parameter targets, predicted values, reference energies, wavefunctions or validation scores. The scientific code is retained from the previously verified source snapshots.

## Layout

| Path | Contents |
| --- | --- |
| code/original_transformer/ | Original Transformer source; its data path is configurable through QCML_DATA_DIR |
| code/joint17/ | Revised 17-task training, data preparation and evaluation source |
| code/joint_n2/ | Joint original-task + N₂ Transformer source and its original input/schema definitions |
| code/n2_reference/ | N₂ reference generation and dataset export source |
| code/dynamics/nve_lih/ | Corrected-force NVE implementation and original source evidence |
| code/controls/ | Basis-transfer and shared GNN/independent LiH control implementations |
| tests/inputs/ | Test geometries, feature-only NPZ/CSV files, feature scaling and parameter definitions |
| scripts/check_test_inputs.py | Check delivered input hashes, dimensions and sample counts |
| scripts/predict_joint_inputs.py | Forward inference with a separately supplied joint checkpoint |

The model cohorts retain their original distinctions. The N₂ example uses the joint model; no standalone N₂ training data or result comparison is distributed. Historical source may contain unused modes for compatibility, but those modes have no precomputed result files in this package.

## Test inputs

| Input group | Cases | Contents |
| --- | ---: | --- |
| Original 17-task held-out split | 605 | 14 input features per sample, physical input features, geometry/task identifiers and fixed input scaling |
| Joint N₂ held-out split | 9 | 6 input features per sample, test geometries, fixed scaling, molecular settings and generator definitions |
| LiH held-out subset | 44 | Bond geometries and molecular/ansatz settings |
| LiH basis examples | 30 per basis | Existing test geometries for STO-3G, 6-31G* and cc-pVDZ |
| LiH dynamics | 1 initial state | Coordinates, velocities, masses, time step and step count |

HOMO/LUMO quantities are retained only where they are required model input features. They are not target VQE/FCI energies. No trained checkpoint, target theta array, CI wavefunction, density matrix, cached Hamiltonian/integral result, trajectory, training history, score table, figure or manuscript result data is included.

## Check the inputs

The input checker requires NumPy and does not load a model:

```bash
python scripts/check_test_inputs.py
```

The expected status is TEST_INPUTS_OK. To check the full extracted package bytes, run from its root:

```bash
sha256sum --quiet --strict -c FILES_SHA256SUMS
```

## Optional joint-model inference

With PyTorch installed and your separately supplied trained joint checkpoint:

```bash
python scripts/predict_joint_inputs.py \
  --checkpoint /absolute/path/to/joint_best.pt \
  --inputs tests/inputs/n2/model_inputs.npz \
  --out outputs/n2_forward
```

Use tests/inputs/legacy17/model_inputs.npz for the original-task domain of the same joint model. The wrapper reads only input features and model weights. Predictions are generated only when you run it, in a fresh output directory ignored by Git. No prediction outputs are prepackaged. The input checker and byte checks were run locally; the new forward wrapper received static checks only because this packaging environment does not have PyTorch installed.

## Original full-data interfaces

Training and reference-scored evaluation source retains its original full-dataset contracts and file locks. Those interfaces need user-provided training labels, reference data or checkpoints. The new tests/inputs files are input-only exports, not replacements for the old labelled train/validation/test archives. Accordingly, the archived cluster.sh full-data preflight and the historical NVE staging driver are not the input-check command for this distribution. Computational dependencies vary by component; the existing component requirements and source imports identify the full runtime needs.

Precomputed reference tables, raw VQE-labelled training tables, analysis/plotting datasets, paper-reply builders, run logs, model weights and prior server-recovery reports were removed from the upload scope. Missing historical run assets are outside this source-and-test-input release and do not require another collection round for this upload.
