# Shared 17-task molecular GNN and Transformer control

Both models use the original STO-3G molecule–ansatz task labels with geometry-grouped partitions. Each model produces a 303-column joint output scored under the per-task masks. The Transformer takes the 14 stored descriptors, including bond length. The GNN takes these descriptors along with up to four atomic numbers and a padded pair-distance matrix derived from the fixed scan geometry (distances in angstrom). Ansatz-associated descriptors provide task context; the same molecule at a given scan coordinate has the same graph across ansätze. N₂ is outside this benchmark.

The frozen source code requires the full external prepared data with manifest SHA-256 `9a86fe733ab1849921bd3d51d84e0aaecbd2758c11123a567251bf816929b12d`. Use that exact original archive when reproducing training; FCI is not part of the objective. Train and validation contain 2,777 and 605 records, respectively. Both configurations use the weighted objective, matched seeds and batch orders, 3,000 epochs, and checkpoint selection by equal-per-sample validation parameter MSE. The protocol does not score the test split.

With Python, NumPy, pandas, and PyTorch installed, run from this directory:

```bash
python qcml.py preflight --prepared /absolute/path/to/exact_prepared --data-only
python qcml.py train --prepared /absolute/path/to/exact_prepared \
  --model gnn --seed 0 --epochs 3000 --out /absolute/path/to/new_run
```

To reproduce the Transformer arm, set `--model transformer`. Allocate a distinct output directory for each run. `scripts/joint17.slurm` shows the corresponding Slurm arguments. The public source includes the fixed geometry rules and the two training configurations.

The historical code snapshot also hashed an internal, report-only geometry diagnostics file. The public distribution omits that results file; the files used for data loading and training are retained unchanged.
