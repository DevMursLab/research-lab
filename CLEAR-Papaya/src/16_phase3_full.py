"""Phase 3 (full) — the core experiment of the paper, with error bars.

Two data conditions:
  * as_published : D1 augmented images INCLUDED (this is how prior papaya-leaf
                   papers actually build their corpora)  -- groups.csv
  * raw_only     : augmentation-derived files removed                -- groups_rawonly.csv

Two protocols, identical ResNet-18 recipe:
  * S1 : image-level random stratified split   (3 seeds -> mean +/- sd)
  * S2 : StratifiedGroupKFold, group-disjoint  (3 folds -> mean +/- sd)

The S1 - S2 accuracy gap under `as_published` is the headline "inflation".

Writes outputs/phase3_full.{json,md}.  ~1.5-2 h on a 4-thread CPU.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torchvision.models import ResNet18_Weights, resnet18

from common import OUT_DIR, ROOT, SEED

RES, CROP = 160, 128
EPOCHS, BATCH, LR = 6, 64, 1e-4
N_SPLITS = 3
CLASSES = ["anthracnose", "bacterial_leaf_spot", "healthy",
           "mite_or_deficiency", "powdery_mildew", "prsv"]
C2I = {c: i for i, c in enumerate(CLASSES)}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def make_cache(manifest_csv, groups_csv, tag):
    cache_p = OUT_DIR / f"cache_{tag}_{RES}.npy"
    meta_p = OUT_DIR / f"cache_{tag}_{RES}.csv"
    man = pd.read_csv(OUT_DIR / manifest_csv)
    grp = pd.read_csv(OUT_DIR / groups_csv)[["image_path", "group_id"]]
    df = man.merge(grp, on="image_path")
    if "split_hint" in df:
        df = df[df.split_hint.eq("closed_set")]
    df = df[df.unified_label.isin(CLASSES) & ~df.source_id.isin(["D4", "D5"])].reset_index(drop=True)
    if cache_p.exists() and meta_p.exists() and len(pd.read_csv(meta_p)) == len(df):
        return np.load(cache_p), pd.read_csv(meta_p)
    arr = np.zeros((len(df), RES, RES, 3), dtype=np.uint8)
    t0 = time.time()
    for i, p in enumerate(df.image_path):
        try:
            arr[i] = np.asarray(Image.open(ROOT / p).convert("RGB").resize((RES, RES), Image.BILINEAR))
        except Exception:  # noqa: BLE001
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [{tag}] cache {i+1}/{len(df)}  {time.time()-t0:.0f}s", flush=True)
    np.save(cache_p, arr)
    df[["image_path", "unified_label", "group_id", "source_id"]].to_csv(meta_p, index=False)
    return arr, df


def batches(idx, arr, y, train, rng):
    for s in range(0, len(idx), BATCH):
        b = idx[s:s + BATCH]
        x = torch.from_numpy(arr[b].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        if train:
            i0, j0 = rng.integers(0, RES - CROP + 1), rng.integers(0, RES - CROP + 1)
            x = x[:, :, i0:i0 + CROP, j0:j0 + CROP]
            if rng.random() < 0.5:
                x = torch.flip(x, dims=[3])
        else:
            o = (RES - CROP) // 2
            x = x[:, :, o:o + CROP, o:o + CROP]
        yield (x - MEAN) / STD, torch.from_numpy(y[b])


def train_eval(arr, y, tr, te, wts, tag):
    rng = np.random.default_rng(SEED)
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(weight=wts)
    best = (0.0, 0.0)
    for ep in range(EPOCHS):
        net.train(); rng.shuffle(tr); t0 = time.time()
        for x, yb in batches(tr, arr, y, True, rng):
            opt.zero_grad(); lossf(net(x), yb).backward(); opt.step()
        net.eval(); pr = []
        with torch.no_grad():
            for x, _ in batches(te, arr, y, False, rng):
                pr.append(net(x).argmax(1).numpy())
        p = np.concatenate(pr)
        acc, f1 = accuracy_score(y[te], p), f1_score(y[te], p, average="macro")
        best = max(best, (acc, f1))
        print(f"  [{tag}] ep{ep+1}/{EPOCHS} acc={acc:.3f} mF1={f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return best  # (best acc, best mF1) over epochs


def rho_split(g, tr, te):
    s = set(g[tr].tolist())
    return float(np.mean([x in s for x in g[te]]))


def evaluate_condition(arr, df, cond):
    y = df.unified_label.map(C2I).to_numpy()
    g = df.group_id.to_numpy()
    wts = torch.tensor([len(y) / (len(CLASSES) * max(1, (y == i).sum()))
                        for i in range(len(CLASSES))], dtype=torch.float32)
    rec = {"n_images": int(len(y)), "S1": [], "S2": [], "S1_rho": [], "S2_rho": []}

    for s in range(N_SPLITS):
        tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.30,
                                             random_state=SEED + s).split(arr, y))
        rec["S1_rho"].append(rho_split(g, tr, te))
        rec["S1"].append(train_eval(arr, y, tr, te, wts, f"{cond}/S1.{s}"))

    sgkf = StratifiedGroupKFold(n_splits=max(N_SPLITS, 3), shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(sgkf.split(arr, y, groups=g)):
        if k >= N_SPLITS:
            break
        assert not (set(g[tr]) & set(g[te]))
        rec["S2_rho"].append(rho_split(g, tr, te))
        rec["S2"].append(train_eval(arr, y, tr, te, wts, f"{cond}/S2.{k}"))
    return rec


def agg(pairs):
    a = np.array([p[0] for p in pairs]); f = np.array([p[1] for p in pairs])
    return (a.mean(), a.std(), f.mean(), f.std())


def main():
    torch.manual_seed(SEED)
    conds = {
        "as_published": make_cache("master_manifest.csv", "groups.csv", "aspub"),
        "raw_only": make_cache("master_manifest_rawonly.csv", "groups_rawonly.csv", "raw"),
    }
    out = {}
    for cond, (arr, df) in conds.items():
        print(f"\n==== condition: {cond}  N={len(df)} ====", flush=True)
        rec = evaluate_condition(arr, df, cond)
        s1 = agg(rec["S1"]); s2 = agg(rec["S2"])
        out[cond] = {
            "n_images": rec["n_images"],
            "S1_rho_mean": float(np.mean(rec["S1_rho"])),
            "S2_rho_mean": float(np.mean(rec["S2_rho"])),
            "S1_acc": [s1[0], s1[1]], "S1_mF1": [s1[2], s1[3]],
            "S2_acc": [s2[0], s2[1]], "S2_mF1": [s2[2], s2[3]],
            "acc_inflation": s1[0] - s2[0], "mF1_inflation": s1[2] - s2[2],
            "raw": {k: rec[k] for k in ("S1", "S2", "S1_rho", "S2_rho")},
        }
    (OUT_DIR / "phase3_full.json").write_text(json.dumps(out, indent=2, default=float))

    lines = ["# Phase 3 (full) — leakage inflation with error bars",
             "",
             f"ResNet-18 (ImageNet-pretrained), {CROP}px, {EPOCHS} epochs, class-weighted CE, "
             f"best-epoch test score. S1 = {N_SPLITS} random splits; S2 = {N_SPLITS} "
             "group-disjoint folds. Mean ± sd.",
             "",
             "| condition | N | protocol | split ρ | test acc % | macro-F1 % |",
             "|---|---|---|---|---|---|"]
    for cond, r in out.items():
        lines.append(f"| {cond} | {r['n_images']} | S1 (random) | {r['S1_rho_mean']:.2f} | "
                     f"{r['S1_acc'][0]*100:.1f} ± {r['S1_acc'][1]*100:.1f} | "
                     f"{r['S1_mF1'][0]*100:.1f} ± {r['S1_mF1'][1]*100:.1f} |")
        lines.append(f"| {cond} | {r['n_images']} | S2 (group) | {r['S2_rho_mean']:.2f} | "
                     f"{r['S2_acc'][0]*100:.1f} ± {r['S2_acc'][1]*100:.1f} | "
                     f"{r['S2_mF1'][0]*100:.1f} ± {r['S2_mF1'][1]*100:.1f} |")
        lines.append(f"| **{cond}** | | **inflation** | | "
                     f"**+{r['acc_inflation']*100:.1f}** | **+{r['mF1_inflation']*100:.1f}** |")
    lines += ["",
              "`as_published` keeps D1's shipped augmentation images (the condition prior "
              "work operates in); `raw_only` removes them. The gap between the two inflation "
              "rows is how much of the reported accuracy is an artefact of dataset protocol."]
    (OUT_DIR / "phase3_full.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
