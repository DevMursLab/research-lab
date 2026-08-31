"""Phase 3 (real) — fine-tune a small CNN under leaky S1 vs clean S2.

CPU-feasible recipe: ResNet-18 (ImageNet-pretrained), 128px, images pre-decoded
to an in-RAM uint8 cache so each epoch is pure compute.

S1 = StratifiedShuffleSplit 70/30 over images (near-dups leak across the split)
S2 = one held-out fold of StratifiedGroupKFold on groups_rawonly (group-disjoint)
Both trained with identical hyper-params; the test-accuracy gap = leakage inflation.

Writes outputs/phase3_finetune.{json,md}.  Run:  python src/15_phase3_finetune.py
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
EPOCHS, BATCH, LR = 4, 64, 1e-4
CLASSES = ["anthracnose", "bacterial_leaf_spot", "healthy",
           "mite_or_deficiency", "powdery_mildew", "prsv"]
C2I = {c: i for i, c in enumerate(CLASSES)}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_cache():
    cache_p = OUT_DIR / f"cache_img_{RES}.npy"
    meta_p = OUT_DIR / f"cache_meta_{RES}.csv"
    man = pd.read_csv(OUT_DIR / "master_manifest_rawonly.csv")
    grp = pd.read_csv(OUT_DIR / "groups_rawonly.csv")[["image_path", "group_id"]]
    df = man.merge(grp, on="image_path")
    df = df[df.split_hint.eq("closed_set") & df.unified_label.isin(CLASSES)].reset_index(drop=True)
    if cache_p.exists() and meta_p.exists():
        m = pd.read_csv(meta_p)
        if len(m) == len(df):
            return np.load(cache_p), m
    arr = np.zeros((len(df), RES, RES, 3), dtype=np.uint8)
    t0 = time.time()
    for i, p in enumerate(df.image_path):
        try:
            im = Image.open(ROOT / p).convert("RGB").resize((RES, RES), Image.BILINEAR)
            arr[i] = np.asarray(im)
        except Exception:  # noqa: BLE001
            pass
        if (i + 1) % 500 == 0:
            print(f"  cache {i+1}/{len(df)}  {time.time()-t0:.0f}s", flush=True)
    np.save(cache_p, arr)
    df[["image_path", "unified_label", "group_id", "source_id"]].to_csv(meta_p, index=False)
    return arr, df


def batches(idx, arr, y, train, rng):
    for s in range(0, len(idx), BATCH):
        b = idx[s:s + BATCH]
        x = torch.from_numpy(arr[b].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        if train:
            i0 = rng.integers(0, RES - CROP + 1); j0 = rng.integers(0, RES - CROP + 1)
            x = x[:, :, i0:i0 + CROP, j0:j0 + CROP]
            if rng.random() < 0.5:
                x = torch.flip(x, dims=[3])
        else:
            o = (RES - CROP) // 2
            x = x[:, :, o:o + CROP, o:o + CROP]
        x = (x - MEAN) / STD
        yield x, torch.from_numpy(y[b])


def run(arr, y, tr, te, wts, tag):
    rng = np.random.default_rng(SEED)
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(weight=wts)
    for ep in range(EPOCHS):
        net.train(); rng.shuffle(tr); t0 = time.time(); tot = 0.0
        for x, yb in batches(tr, arr, y, True, rng):
            opt.zero_grad(); out = net(x); ls = lossf(out, yb)
            ls.backward(); opt.step(); tot += ls.item() * len(yb)
        net.eval(); preds = []
        with torch.no_grad():
            for x, _ in batches(te, arr, y, False, rng):
                preds.append(net(x).argmax(1).numpy())
        p = np.concatenate(preds)
        acc = accuracy_score(y[te], p); f1 = f1_score(y[te], p, average="macro")
        print(f"  [{tag}] epoch {ep+1}/{EPOCHS}  loss={tot/len(tr):.3f}  "
              f"test_acc={acc:.3f}  test_mF1={f1:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return float(acc), float(f1), p


def rho_of_split(groups, tr, te):
    trg = set(groups[tr].tolist())
    return float(np.mean([g in trg for g in groups[te]]))


def main():
    torch.manual_seed(SEED)
    arr, df = build_cache()
    y = df.unified_label.map(C2I).to_numpy()
    g = df.group_id.to_numpy()
    cw = torch.tensor([len(y) / (len(CLASSES) * max(1, (y == i).sum()))
                       for i in range(len(CLASSES))], dtype=torch.float32)
    print(f"N={len(y)}  class_weights={[round(float(w),2) for w in cw]}")

    # S1 — leaky
    tr1, te1 = next(StratifiedShuffleSplit(n_splits=1, test_size=0.30,
                                           random_state=SEED).split(arr, y))
    r1 = rho_of_split(g, tr1, te1)
    print(f"\nS1 split leakage rho = {r1:.3f}")
    a1, f1_1, _ = run(arr, y, tr1, te1, cw, "S1")

    # S2 — clean (one group-disjoint fold, matched test size ~20%)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr2, te2 = next(iter(sgkf.split(arr, y, groups=g)))
    assert not (set(g[tr2]) & set(g[te2]))
    r2 = rho_of_split(g, tr2, te2)
    print(f"\nS2 split leakage rho = {r2:.3f}  (must be 0)")
    a2, f1_2, _ = run(arr, y, tr2, te2, cw, "S2")

    res = {"model": "resnet18_ft", "res": CROP, "epochs": EPOCHS,
           "S1": {"rho": r1, "test_acc": a1, "test_macroF1": f1_1, "n_test": int(len(te1))},
           "S2": {"rho": r2, "test_acc": a2, "test_macroF1": f1_2, "n_test": int(len(te2))},
           "acc_inflation": a1 - a2, "macroF1_inflation": f1_1 - f1_2}
    (OUT_DIR / "phase3_finetune.json").write_text(json.dumps(res, indent=2))
    md = (f"""# Phase 3 (real) — ResNet-18 fine-tune, leaky S1 vs clean S2

Raw-only corpus, {len(y)} closed-set images, {CROP}px, {EPOCHS} epochs, class-weighted CE.

| protocol | split rho | test acc | test macro-F1 | n_test |
|---|---|---|---|---|
| S1 (image-level random) | {r1:.3f} | {a1*100:.1f}% | {f1_1*100:.1f}% | {len(te1)} |
| S2 (group-disjoint fold) | {r2:.3f} | {a2*100:.1f}% | {f1_2*100:.1f}% | {len(te2)} |
| **inflation (S1 - S2)** | | **+{(a1-a2)*100:.1f} pts** | **+{(f1_1-f1_2)*100:.1f} pts** | |

Same architecture and hyper-parameters both rows; the only change is how the
train/test line is drawn. ResNet-18 @128px is a deliberately small CPU-feasible
setup — a full ResNet-50/EfficientNet @224 (where the ~99% literature numbers sit)
is expected to widen this gap, not narrow it.
""")
    (OUT_DIR / "phase3_finetune.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
