"""Paired old-only, N2-only, and joint Transformer experiment on locked splits."""
import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from vqe_only import csv_out, evaluate_n2, n2_data
from legacy_inputs import EXPECTED, sha, verify_prepared


MODES = {"legacy": ("old",), "n2": ("n2",), "joint": ("old", "n2")}


def json_out(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def old_data(root, splits):
    verified = verify_prepared(root, splits)
    root = root.resolve()
    with (root / "samples.tsv").open(newline="",encoding="utf-8") as f:
        meta = {int(r["global_row"]): r for r in csv.DictReader(f, delimiter="\t")}
    if len(meta) != 3987 or len({r["task"] for r in meta.values()}) != 17:
        raise ValueError("Legacy sample/task count mismatch")
    result = {}
    for s in splits:
        with np.load(root / (s+".npz"), allow_pickle=False) as z:
            x, y, mask, i = (z[k] for k in ("X", "Y", "mask", "global_indices"))
        n = EXPECTED[s]
        if (x.shape != (n,14) or y.shape != (n,303) or mask.dtype != bool or i.shape != (n,)
                or len(set(i)) != n or not np.isfinite(x).all() or np.isinf(y).any()
                or not np.array_equal(np.isfinite(y), mask) or not mask.any(1).all()
                or any(meta[int(j)]["split"] != s for j in i)):
            raise ValueError(f"Bad legacy {s} arrays or rows")
        result[s] = dict(index=i, x=x, y=y, mask=mask,
                         task=np.asarray([meta[int(j)]["task"] for j in i]))
    return result, verified


class Adapter(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_in, 64))
        self.b = nn.Parameter(torch.empty(n_in, 64))
        self.head = nn.Linear(64, n_out)
        nn.init.normal_(self.w, std=.02)
        nn.init.normal_(self.b, std=.02)
        nn.init.normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)


class SharedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleDict({"old": Adapter(14,303), "n2": Adapter(6,135)})
        self.readout = nn.Parameter(torch.empty(1,1,64))
        nn.init.normal_(self.readout, std=.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(64, 2, 256, dropout=.1, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(2)])
        self.norm = nn.LayerNorm(64)

    def forward(self, x, domain):
        a = self.adapters[domain]
        tokens = x.unsqueeze(-1)*a.w.unsqueeze(0)+a.b.unsqueeze(0)
        h = torch.cat((self.readout.expand(len(x),-1,-1), tokens), dim=1)
        for layer in self.layers:
            h = layer(h)
        return a.head(self.norm(h[:,0]))


def tensors(row, device):
    mask = row["mask"]
    return (torch.as_tensor(row["x"], dtype=torch.float32, device=device),
            torch.as_tensor(np.where(mask, row["y"], 0), dtype=torch.float32, device=device),
            torch.as_tensor(mask, dtype=torch.bool, device=device))


def squared(model, domain, batch):
    x, y, mask = batch
    return (model(x, domain)-y).square().masked_fill(~mask, 0).sum(1), mask.sum(1)


@torch.no_grad()
def val_mse(model, data, domain, bs):
    x, y, m = data
    losses = []
    for j in range(0, len(x), bs):
        e, n = squared(model, domain, (x[j:j+bs],y[j:j+bs],m[j:j+bs]))
        losses.append(e/n)
    return torch.cat(losses).mean().item()


def fit(data, mode, seed, epochs, device, out, batch_size=64, legacy_identity=None):
    domains = MODES[mode]
    torch.manual_seed(15000+seed)
    model = SharedTransformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50, factor=.5, min_lr=1e-6)
    train = {d: tensors(data[d]["train"], device) for d in domains}
    val = {d: tensors(data[d]["val"], device) for d in domains}
    old_count = train["old"][2].sum(dim=1).float().mean().item() if "old" in domains else None
    out.mkdir(parents=True, exist_ok=False)
    info = dict(mode=mode, seed=seed, epochs=epochs, batch_size=batch_size, device=device,
                torch=str(torch.__version__), train_counts={d: len(train[d][0]) for d in domains},
                val_counts={d: len(val[d][0]) for d in domains}, old_mean_valid_count=old_count,
                val_selection="mean of active domains' equal-per-sample masked parameter MSEs",
                n2_updates_per_epoch=int("n2" in domains), test_used_for_fitting=False)
    if legacy_identity is not None:
        info["legacy_identity"] = legacy_identity
    best, best_epoch, start = math.inf, None, time.perf_counter()
    with (out / "history.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("epoch","old_val_mse","n2_val_mse","selection_mse","learning_rate"))
        for epoch in range(1, epochs+1):
            model.train()
            if "old" in domains:
                rng = np.random.RandomState((seed+104729*epoch+173) % 2**32)
                order = rng.permutation(len(train["old"][0]))
                batches = [order[j:j+batch_size] for j in range(0,len(order),batch_size)]
            else:
                batches = [None]
            n2_batch = (epoch-1) % len(batches)
            for step, batch in enumerate(batches):
                opt.zero_grad(set_to_none=True)
                terms = []
                if batch is not None:
                    ix = torch.as_tensor(batch, device=device)
                    sse, _ = squared(model, "old", tuple(v[ix] for v in train["old"]))
                    terms.append(sse.mean()/old_count)
                if "n2" in domains and step == n2_batch:
                    sse, n = squared(model, "n2", train["n2"])
                    terms.append((sse/n).mean())
                loss = sum(terms)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
                opt.step()
            model.eval()
            losses = {d: val_mse(model, val[d], d, batch_size) for d in domains}
            score = sum(losses.values())/len(losses)
            if not math.isfinite(score):
                raise ValueError("Nonfinite validation score")
            schedule.step(score)
            w.writerow((epoch, losses.get("old",""), losses.get("n2",""), score, opt.param_groups[0]["lr"]))
            if score < best:
                best, best_epoch, best_val = score, epoch, losses
                p = out / "best.pt"
                torch.save(dict(info, epoch=epoch, state=model.state_dict()), p.with_suffix(".tmp"))
                p.with_suffix(".tmp").replace(p)
            if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
                f.flush()
                print(mode, seed, epoch, losses, "best", best_epoch, flush=True)
    json_out(out/"training.json", dict(info, best_epoch=best_epoch, best_validation=best_val,
                                       seconds=time.perf_counter()-start))


@torch.no_grad()
def predict(model, domain, row, device):
    x = torch.as_tensor(row["x"], dtype=torch.float32, device=device)
    y = torch.cat([model(a,domain) for a in x.split(64)]).cpu().numpy().astype(np.float64)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite prediction")
    return y


def old_metrics(pred, row):
    err = np.zeros_like(pred)
    mask = row["mask"]
    err[mask] = pred[mask]-row["y"][mask]
    sse, n = (err**2).sum(1), mask.sum(1)
    out = {}
    for task in ("all", *sorted(set(row["task"]))):
        keep = np.ones(len(pred), bool) if task=="all" else row["task"]==task
        out[task] = dict(samples=int(keep.sum()), valid_labels=int(n[keep].sum()),
                         theta_RMSE_micro=float(np.sqrt(sse[keep].sum()/n[keep].sum())),
                         equal_sample_MSE=float(np.mean(sse[keep]/n[keep])))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("preflight","train","evaluate"))
    p.add_argument("--n2-root", type=Path)
    p.add_argument("--legacy-prepared", type=Path)
    p.add_argument("--mode", choices=tuple(MODES), default="joint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    p.add_argument("--out", type=Path)
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args()
    if args.epochs<1 or args.seed<0 or args.threads<1:
        p.error("epochs/threads must be positive, seed nonnegative")
    if args.action!="preflight" and args.out is None:
        p.error("--out is required for train/evaluate")
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    if args.action=="evaluate":
        if args.checkpoint is None:
            p.error("evaluate requires --checkpoint")
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if args.mode != ck["mode"]:
            p.error("--mode does not match checkpoint")
    domains = MODES[args.mode]
    if "n2" in domains and args.n2_root is None:
        p.error("--n2-root must point to the existing N2 package")
    if args.out:
        for src in (args.n2_root, args.legacy_prepared):
            if src and (src.resolve()==args.out.resolve() or src.resolve() in args.out.resolve().parents):
                p.error("Write the results outside the original input directories")
    splits = ("test",) if args.action=="evaluate" else ("train","val")
    data = {}
    legacy_identity = None
    if "old" in domains:
        try:
            data["old"], legacy_identity = old_data(args.legacy_prepared, splits)
            if args.action=="evaluate":
                fit_identity = ck.get("legacy_identity", {})
                if fit_identity.get("files_sha256") != legacy_identity["files_sha256"]:
                    raise ValueError("Checkpoint and evaluation use different legacy prepared data; "
                                     "evaluate with the same verified prepared directory used for training")
        except (OSError,ValueError,KeyError,TypeError) as e:
            p.exit(2,f"LEGACY_INPUT_REJECTED [{args.legacy_prepared}]: {e}\n")
    if "n2" in domains:
        data["n2"] = n2_data(args.n2_root, splits)
    if args.action=="preflight":
        if legacy_identity:
            print("LEGACY_VERIFIED",legacy_identity["origin"],
                  "source_fingerprint="+legacy_identity["source_fingerprint"],flush=True)
        print("PREFLIGHT_OK", {d:{s:len(a["index"]) for s,a in ds.items()} for d,ds in data.items()})
    elif args.action=="train":
        fit(data,args.mode,args.seed,args.epochs,args.device,args.out,legacy_identity=legacy_identity)
    else:
        model = SharedTransformer().to(args.device)
        model.load_state_dict(ck["state"], strict=True)
        model.eval()
        summary = dict(mode=ck["mode"], seed=ck["seed"], best_epoch=ck["epoch"],
                       checkpoint_sha256=sha(args.checkpoint), split="test")
        if "old" in domains:
            row = data["old"]["test"]
            summary["legacy_identity"] = legacy_identity
            summary["legacy_parameter_test"] = old_metrics(predict(model,"old",row,args.device),row)
        if "n2" in domains:
            row = data["n2"]["test"]
            n2pred = predict(model,"n2",row,args.device)
            summary["N2_vs_VQE"], rows = evaluate_n2(args.n2_root,row,{ck["mode"]:n2pred})
        args.out.mkdir(parents=True, exist_ok=False)
        if "n2" in domains:
            csv_out(args.out/"n2_per_geometry.csv",rows)
            np.savez_compressed(args.out/"n2_predictions.npz", index=row["index"], theta=n2pred)
        json_out(args.out/"metrics.json",summary)
        print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
