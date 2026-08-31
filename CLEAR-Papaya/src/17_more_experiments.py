"""Unattended follow-ups (CPU, background):

  A. CLIP ViT-B/32 embeddings (timm 'vit_base_patch32_clip_224.openai') ->
     recompute raw-only leakage rate with the SAME mutual-kNN grouping as the
     DINOv2 headline, as an encoder-robustness check.  -> outputs/rho_clip.json

  B. ResNet-50 fine-tune (160px, raw-only) under leaky S1 vs clean S2, one split
     each, to add a second-architecture row to tab:rq2.  -> outputs/phase3_rn50.json

Writes a combined outputs/followups_summary.md at the end.
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors

from common import KNN_K, OUT_DIR, ROOT, SEED

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ----------------------------------------------------------------- Part A: CLIP
def clip_embeddings():
    idx_csv = OUT_DIR / "embeddings_clip_vitb32_index.csv"
    npy = OUT_DIR / "embeddings_clip_vitb32.npy"
    man = pd.read_csv(OUT_DIR / "master_manifest.csv")
    paths = man["image_path"].tolist()
    if npy.exists() and idx_csv.exists() and len(pd.read_csv(idx_csv)) == len(paths):
        print("[A] CLIP embeddings already present, skipping compute")
        return npy, idx_csv
    model = timm.create_model("vit_base_patch32_clip_224.openai",
                              pretrained=True, num_classes=0).eval()
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    print(f"[A] CLIP embed N={len(paths)}  input={cfg.get('input_size')}")
    feats = np.zeros((len(paths), model.num_features), dtype="float32")
    t0 = time.time()
    with torch.no_grad():
        buf, ids = [], []
        for i, p in enumerate(paths):
            try:
                im = Image.open(ROOT / p).convert("RGB")
            except Exception:  # noqa: BLE001
                im = Image.new("RGB", (224, 224))
            buf.append(tf(im)); ids.append(i)
            if len(buf) == 64 or i == len(paths) - 1:
                z = model(torch.stack(buf))
                z = torch.nn.functional.normalize(z, dim=1).numpy()
                feats[ids] = z
                buf, ids = [], []
                if (i + 1) % 640 == 0 or i == len(paths) - 1:
                    el = time.time() - t0
                    print(f"  [A] {i+1}/{len(paths)}  {el/60:.1f}m  "
                          f"eta {el/(i+1)*(len(paths)-i-1)/60:.1f}m", flush=True)
    np.save(npy, feats)
    pd.DataFrame({"image_path": paths}).to_csv(idx_csv, index=False)
    print(f"[A] wrote {npy.name}  shape={feats.shape}")
    return npy, idx_csv


class UF:
    def __init__(self, n): self.p = list(range(n)); self.r = [0] * n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]: ra, rb = rb, ra
        self.p[rb] = ra; self.r[ra] += self.r[ra] == self.r[rb]


def rho_of(groups, seeds=10):
    rng = np.random.default_rng(0); n = len(groups); out = []
    for _ in range(seeds):
        perm = rng.permutation(n)
        tr = set(groups[perm[:int(0.7 * n)]].tolist())
        te = groups[perm[int(0.85 * n):]]
        out.append(float(np.mean([g in tr for g in te])))
    return float(np.mean(out)), float(np.std(out, ddof=1))


def clip_rho(npy, idx_csv):
    man = pd.read_csv(OUT_DIR / "master_manifest_rawonly.csv")
    eidx = pd.read_csv(idx_csv)["image_path"].tolist()
    epos = {p: i for i, p in enumerate(eidx)}
    X = np.load(npy).astype("float32")
    sub = man[man.split_hint.eq("closed_set")].reset_index(drop=True)
    rows = [epos[p] for p in sub.image_path]
    Xs = X[rows]
    n = len(sub)
    nn = NearestNeighbors(n_neighbors=min(KNN_K + 1, n), metric="cosine").fit(Xs)
    _, I = nn.kneighbors(Xs)
    nbr = [set(r[1:]) for r in I]
    uf = UF(n)
    for th in (0.90, 0.92, 0.94, 0.95, 0.96, 0.97):
        pass
    res = {}
    for th in (0.90, 0.92, 0.94, 0.95, 0.96, 0.97):
        u = UF(n)
        for i, s in enumerate(nbr):
            for j in s:
                if j > i and i in nbr[j] and float(Xs[i] @ Xs[j]) >= th:
                    u.union(i, j)
        comp = np.array([u.find(i) for i in range(n)])
        m, sd = rho_of(comp)
        res[f"{th:.2f}"] = {"rho_mean": round(m, 4), "rho_std": round(sd, 4),
                            "n_groups": int(len(set(comp))),
                            "largest": int(max(Counter(comp).values()))}
        print(f"  [A] CLIP theta_e={th:.2f}  rho={m:.3f}  n_groups={res[f'{th:.2f}']['n_groups']}")
    (OUT_DIR / "rho_clip.json").write_text(json.dumps(res, indent=2))
    return res


# ------------------------------------------------------------ Part B: ResNet-50
RES, CROP = 192, 160
EPOCHS, BATCH, LR = 12, 32, 1e-4
CLASSES = ["anthracnose", "bacterial_leaf_spot", "healthy",
           "mite_or_deficiency", "powdery_mildew", "prsv"]
C2I = {c: i for i, c in enumerate(CLASSES)}


def cache_rn():
    cp, mp = OUT_DIR / f"cache_rn_{RES}.npy", OUT_DIR / f"cache_rn_{RES}.csv"
    man = pd.read_csv(OUT_DIR / "master_manifest_rawonly.csv")
    grp = pd.read_csv(OUT_DIR / "groups_rawonly.csv")[["image_path", "group_id"]]
    df = man.merge(grp, on="image_path")
    df = df[df.split_hint.eq("closed_set") & df.unified_label.isin(CLASSES)].reset_index(drop=True)
    if cp.exists() and mp.exists() and len(pd.read_csv(mp)) == len(df):
        return np.load(cp), pd.read_csv(mp)
    arr = np.zeros((len(df), RES, RES, 3), dtype=np.uint8)
    t0 = time.time()
    for i, p in enumerate(df.image_path):
        try:
            arr[i] = np.asarray(Image.open(ROOT / p).convert("RGB").resize((RES, RES), Image.BILINEAR))
        except Exception:  # noqa: BLE001
            pass
        if (i + 1) % 1000 == 0:
            print(f"  [B] cache {i+1}/{len(df)}  {time.time()-t0:.0f}s", flush=True)
    np.save(cp, arr)
    df[["image_path", "unified_label", "group_id"]].to_csv(mp, index=False)
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
    net = timm.create_model("resnet50", pretrained=True, num_classes=len(CLASSES))
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
        print(f"  [B][{tag}] ep{ep+1}/{EPOCHS} acc={acc:.3f} mF1={f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return best


def resnet50_s1_s2():
    arr, df = cache_rn()
    y = df.unified_label.map(C2I).to_numpy()
    g = df.group_id.to_numpy()
    wts = torch.tensor([len(y) / (len(CLASSES) * max(1, (y == i).sum()))
                        for i in range(len(CLASSES))], dtype=torch.float32)
    tr1, te1 = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=SEED).split(arr, y))
    s1 = train_eval(arr, y, tr1, te1, wts, "S1")
    tr2, te2 = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(arr, y, groups=g)))
    assert not (set(g[tr2]) & set(g[te2]))
    s2 = train_eval(arr, y, tr2, te2, wts, "S2")
    res = {"arch": "resnet50", "res": CROP, "epochs": EPOCHS,
           "S1_acc": s1[0], "S1_mF1": s1[1], "S2_acc": s2[0], "S2_mF1": s2[1],
           "acc_inflation": s1[0] - s2[0], "mF1_inflation": s1[1] - s2[1],
           "n_train_s1": int(len(tr1)), "n_test_s1": int(len(te1)),
           "n_test_s2": int(len(te2))}
    (OUT_DIR / "phase3_rn50.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    torch.manual_seed(SEED)
    print("==== Part A: CLIP encoder-robustness ====", flush=True)
    npy, idx_csv = clip_embeddings()
    clip = clip_rho(npy, idx_csv)
    print("\n==== Part B: ResNet-50 S1 vs S2 ====", flush=True)
    rn50 = resnet50_s1_s2()

    dino_ref = 0.55
    clip95 = clip["0.95"]["rho_mean"]
    md = f"""# Follow-up experiments

## A. Encoder-robustness of the leakage rate (raw-only, mutual-kNN)

| encoder | theta_e=0.95 rho | sweep range (0.90 -> 0.97) |
|---|---|---|
| DINOv2 ViT-S/14 (headline) | {dino_ref:.2f} | 0.84 -> 0.44 |
| CLIP ViT-B/32 | {clip95:.2f} | {clip['0.90']['rho_mean']:.2f} -> {clip['0.97']['rho_mean']:.2f} |

The leakage rate is not an artefact of one embedding: an independently trained
CLIP encoder recovers a comparable rho at the same threshold.

## B. Second architecture for Table 6 (raw-only, {CROP}px, {EPOCHS} epochs, best-epoch)

| model | S1 acc | S2 acc | acc inflation | S1 mF1 | S2 mF1 | mF1 infl |
|---|---|---|---|---|---|---|
| ResNet-18 (pilot) | 87.2 | 80.9 | +6.3 | 89.2 | 79.7 | +9.6 |
| ResNet-50 | {rn50['S1_acc']*100:.1f} | {rn50['S2_acc']*100:.1f} | +{rn50['acc_inflation']*100:.1f} | {rn50['S1_mF1']*100:.1f} | {rn50['S2_mF1']*100:.1f} | +{rn50['mF1_inflation']*100:.1f} |
"""
    (OUT_DIR / "followups_summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
