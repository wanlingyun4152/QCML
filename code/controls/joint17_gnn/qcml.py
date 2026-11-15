#!/usr/bin/env python3
"""Matched 17-task molecular GNN and Transformer benchmark."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent
DEFAULT_PREPARED = HERE.parent / "QCML_Legacy_Exact_Recovery" / "prepared"
CONFIGS = {"gnn": "molecular_gnn", "transformer": "transformer"}


def preflight(prepared, data_only):
    import numpy as np
    from qrev.graphs import load_joint_data, DESCRIPTOR_DIM, MAX_ATOMS

    _, schema, meta, data = load_joint_data(prepared, "molecular_gnn")
    for split, a in data.items():
        rows = meta.iloc[a["global_indices"]]
        print(f"{split}: samples={len(rows)} groups={rows.geometry_group.nunique()} "
              f"tasks={rows.task.nunique()} valid_targets={a['mask'].sum()} X={a['X'].shape} Y={a['Y'].shape}")
    if data_only:
        print("DATA_PREFLIGHT_OK (train/val only)")
        return
    import torch
    from qrev.common import setup
    from qrev.metrics import masked_loss
    from qrev.models import make_model
    setup(0, "cpu", 1)
    a = data["train"]
    systems = meta.iloc[a["global_indices"]].system.to_numpy()
    idx = np.array([np.flatnonzero(systems == s)[0] for s in sorted(set(systems))])
    x = torch.as_tensor(a["X"][idx], dtype=torch.float32)
    mask = torch.as_tensor(a["mask"][idx], dtype=torch.bool)
    y = torch.as_tensor(a["Y"][idx], dtype=torch.float32)
    for short, name in CONFIGS.items():
        cfg = json.loads((HERE/"configs"/(name+".json")).read_text())
        d = schema["input_dim"] if short == "gnn" else DESCRIPTOR_DIM
        model = make_model(cfg, d, schema["output_dim"])
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.eval()
        with torch.no_grad():
            batch = model(x[:, :d])
            single = torch.cat([model(row[None, :d]) for row in x])
            torch.testing.assert_close(batch, single, rtol=1e-5, atol=1e-5)
            if short == "gnn":
                order = [2, 0, 3, 1]
                z = x[:, DESCRIPTOR_DIM:DESCRIPTOR_DIM+MAX_ATOMS][:, order]
                m = x[:, DESCRIPTOR_DIM+MAX_ATOMS:].reshape(-1,MAX_ATOMS,MAX_ATOMS)[:, order, :][:, :, order]
                perm = torch.cat([x[:, :DESCRIPTOR_DIM], z, m.flatten(1)], dim=1)
                torch.testing.assert_close(batch, model(perm), rtol=1e-5, atol=1e-5)
        model.train()
        value = masked_loss(model(x[:, :d]), y, mask, "weighted")
        value.backward()
        if not torch.isfinite(value) or any(p.grad is None or not torch.isfinite(p.grad).all()
                                           for p in model.parameters() if p.requires_grad):
            raise ValueError("Nonfinite or missing gradients in " + short)
        print(f"{short}: parameters={params}, forward/backward and invariance checks passed")
    print("PREFLIGHT_OK (6 distinct molecular graphs, no test labels loaded)")


def write_tsv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def collect(root):
    from qrev.graphs import MANIFEST_SHA
    runs, tasks = [], []
    for arch, model in CONFIGS.items():
        for seed in range(3):
            folder = root / arch / ("seed"+str(seed))
            status = json.loads((folder/"status.json").read_text())
            cfg = json.loads((folder/"config.json").read_text())
            if (status["status"] != "COMPLETED" or status["prepared_manifest_sha256"] != MANIFEST_SHA
                or not status["invariance_passed"] or status["epochs_completed"] != 3000
                or cfg["training"]["loss"] != "weighted" or cfg["model"] != model or cfg["seed"] != seed):
                raise ValueError("Not the planned full validation run: "+str(folder))
            metrics = json.loads((folder/"validation_metrics.json").read_text())
            runs.append({"model":arch,"seed":seed,"best_epoch":status["best_epoch"],
                         "parameters":status["trainable_parameters"], **metrics})
            with (folder/"validation_by_task.tsv").open(newline="") as f:
                tasks += [{"model":arch,"seed":seed,**r} for r in csv.DictReader(f,delimiter="\t")]
    summary = []
    for arch in CONFIGS:
        rows = [r for r in runs if r["model"] == arch]
        vals = [r["parameter_RMSE_micro"] for r in rows]
        summary.append({"model":arch,"seeds":len(vals),"pooled_val_RMSE_mean":mean(vals),
                        "pooled_val_RMSE_sample_SD":stdev(vals),
                        "equal_sample_val_MSE_mean":mean(r["sample_mean_MSE"] for r in rows)})
    out = root/"comparison"
    out.mkdir(exist_ok=False)
    write_tsv(out/"summary.tsv", summary)
    write_tsv(out/"by_seed.tsv", runs)
    write_tsv(out/"by_task_and_seed.tsv", tasks)
    print("COMPARISON="+str(out.resolve()), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("preflight")
    pf.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    pf.add_argument("--data-only", action="store_true")
    fit = sub.add_parser("train")
    fit.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    fit.add_argument("--model", choices=CONFIGS, required=True)
    fit.add_argument("--seed", type=int, choices=(0,1,2), default=0)
    fit.add_argument("--epochs", type=int, default=3000)
    fit.add_argument("--threads", type=int, default=1)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--out", type=Path, required=True)
    col = sub.add_parser("collect")
    col.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "preflight":
        preflight(args.prepared, args.data_only)
    elif args.cmd == "train":
        from qrev.train import train
        cfg = HERE/"configs"/(CONFIGS[args.model]+".json")
        train(args.prepared, cfg, args.out, seed=args.seed, epochs=args.epochs,
              device=args.device, threads=args.threads, purpose="joint17_gnn_control")
    elif args.cmd == "collect":
        collect(args.root)


if __name__ == "__main__":
    main()
