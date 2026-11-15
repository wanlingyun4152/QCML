# QCML source code and test inputs

Repository: https://github.com/wanlingyun4152/QCML

This distribution provides the source code and the input portions of the held-out splits accompanying the revised QCML study. The test files contain geometry identifiers and model inputs only. The LiH dynamics example provides its initial conditions. Model checkpoints, fitted circuit parameters, reference energies, predictions, and run outputs are excluded.

| Path | Contents |
| --- | --- |
| `code/original_transformer/` | Historical Transformer source for the original study and its original split procedure. |
| `code/joint17/` | Geometry-grouped 17-task Transformer and loss-ablation source. |
| `code/controls/joint17_gnn/` | Shared molecular GNN and Transformer comparison on 17 STO-3G tasks. |
| `code/controls/joint17_parallel/` | Separate LiH five-ansatz architecture control. |
| `code/joint_n2/`, `code/n2_reference/` | Joint 18-task model, N₂ reference generation, and LiH label-continuation interface. |
| `code/controls/basis_transfer/` | LiH basis adaptation source and fixed split definition. |
| `code/dynamics/nve_lih/` | LiH NVE implementation and source used to define the dynamics protocol. |
| `tests/inputs/` | Checked geometry, feature arrays, feature scaling and generator definitions. |

## Test inputs

| Case | Count | Input data |
| --- | ---: | --- |
| Original-task held-out set (grouped split) | 605 records | Fourteen descriptors per record; original identifiers and scaling. |
| N₂/UCCSD held-out set (joint 18-task model) | 9 geometries | Six descriptors per record, scaling and generator definitions. |
| LiH/2-UpCCGSD held-out subset | 44 geometries | Bond lengths and task identity. These geometries are a subset of the 605 records. |
| LiH basis adaptation | 30 geometries per basis | Fixed test geometries for STO-3G, 6-31G* and cc-pVDZ. |
| LiH dynamics | 1 initial condition | Positions, velocities, masses, time step and steps. |

The grouped 605-row split is used in the revision experiments. It is not the row-wise split of the original training script. The documented GNN/Transformer comparisons use their validation sets; the public test files do not include scores. For the joint 18-task Transformer, the LiH/2-UpCCGSD overlay changes supervised VQE parameter labels at the same coordinates. The exported inputs are the same for the original 17-task domain before and after applying the overlay. HOMO/LUMO descriptors follow the recorded orbital conventions of the supplied input features.

## Check the unpacked release

From the directory containing this README:

```bash
python -m pip install -r requirements-input-check.txt
python scripts/check_test_inputs.py
sha256sum --quiet --strict -c FILES_SHA256SUMS
```

The input checker verifies file hashes, counts, feature identities and order, and the numerical agreement between each feature array and its CSV/scaling metadata. `SOURCE_FILES.csv` records the provenance of source files and configurations.

## Optional inference

The following command uses PyTorch and a separately supplied checkpoint of the shared 18-task Transformer. Substitute `tests/inputs/legacy17/model_inputs.npz` for the original-task domain of that same checkpoint:

```bash
python scripts/predict_joint_inputs.py \
  --checkpoint /absolute/path/to/joint_best.pt \
  --inputs tests/inputs/n2/model_inputs.npz \
  --out outputs/n2_forward
```

It writes newly predicted parameters from the feature array. A checkpoint from the 17-task comparison, the original historical network, or the independent LiH model uses a different interface.

The full-data training and reference-scored evaluation drivers require external archives. For historical runs, pass the exact prepared version used by the run; preserve the original manifest and file locks. The input-only files in this release and any later FCI reference tables cannot replace those archives. For the 17-task comparison, the external prepared directory contains `manifest.json`, `schema.json`, `samples.tsv`, and the split arrays. See `code/controls/joint17_gnn/README.md` for preflight and training commands.
