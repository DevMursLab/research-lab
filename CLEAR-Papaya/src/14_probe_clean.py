"""Phase 3 (clean) — frozen-feature probe on the RAW-ONLY corpus with
augmentation removed and mutual-kNN groups.

S1 = image-level random stratified split (leaky)
S2 = StratifiedGroupKFold on groups_rawonly.csv (group-disjoint, clean)

Reports S1-S2 inflation for 1-NN cosine and logistic-regression probes.
Writes outputs/phase3_probe_clean.{json,md}.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from common import OUT_DIR, SEED

N_SEEDS = 5
N_FOLDS = 5
TEST_FRAC = 0.30
DROP = {"UNMAPPED"}


def knn1(Xtr, ytr, Xte):
    return ytr[np.argmax(Xte @ Xtr.T, axis=1)]


def ev(Xtr, ytr, Xte, yte):
    p = knn1(Xtr, ytr, Xte)
    r = {"knn1": (accuracy_score(yte, p), f1_score(yte, p, average="macro"))}
    c = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    p = c.predict(Xte)
    r["logreg"] = (accuracy_score(yte, p), f1_score(yte, p, average="macro"))
    return r


def main():
    idx = pd.read_csv(OUT_DIR / "embeddings_dinov2_vits14_index.csv")
    X = np.load(OUT_DIR / "embeddings_dinov2_vits14.npy").astype("float32")
    idx["_row"] = np.arange(len(idx))
    man = pd.read_csv(OUT_DIR / "master_manifest_rawonly.csv")
    grp = pd.read_csv(OUT_DIR / "groups_rawonly.csv")[["image_path", "group_id"]]
    df = man.merge(grp, on="image_path").merge(idx, on="image_path")
    df = df[~df["unified_label"].isin(DROP) & df["split_hint"].eq("closed_set")]
    df = df.reset_index(drop=True)
    X = X[df["_row"].to_numpy()]
    y = df["unified_label"].to_numpy()
    g = df["group_id"].to_numpy()
    print(f"N={len(y)}  groups={len(set(g))}  classes={sorted(set(y))}")

    acc = {(s, c): [] for s in ("S1", "S2") for c in ("knn1", "logreg")}
    f1 = {k: [] for k in acc}

    for s in range(SEED, SEED + N_SEEDS):
        tr, te = next(StratifiedShuffleSplit(
            n_splits=1, test_size=TEST_FRAC, random_state=s).split(X, y))
        r = ev(X[tr], y[tr], X[te], y[te])
        for c in ("knn1", "logreg"):
            acc[("S1", c)].append(r[c][0]); f1[("S1", c)].append(r[c][1])
        print(f"S1 seed {s}: knn1={r['knn1'][0]:.3f} logreg={r['logreg'][0]:.3f}")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(sgkf.split(X, y, groups=g)):
        assert not (set(g[tr]) & set(g[te])), "group leak in S2!"
        r = ev(X[tr], y[tr], X[te], y[te])
        for c in ("knn1", "logreg"):
            acc[("S2", c)].append(r[c][0]); f1[("S2", c)].append(r[c][1])
        print(f"S2 fold {k}: knn1={r['knn1'][0]:.3f} logreg={r['logreg'][0]:.3f}")

    res, rows = {}, []
    for c in ("knn1", "logreg"):
        a1, a2 = np.array(acc[("S1", c)]), np.array(acc[("S2", c)])
        g1, g2 = np.array(f1[("S1", c)]), np.array(f1[("S2", c)])
        res[c] = {"S1_acc": [a1.mean(), a1.std()], "S2_acc": [a2.mean(), a2.std()],
                  "acc_inflation": a1.mean() - a2.mean(),
                  "S1_mF1": [g1.mean(), g1.std()], "S2_mF1": [g2.mean(), g2.std()],
                  "f1_inflation": g1.mean() - g2.mean()}
        rows.append(f"| {c} | {a1.mean()*100:.1f} ± {a1.std()*100:.1f} | "
                    f"{a2.mean()*100:.1f} ± {a2.std()*100:.1f} | "
                    f"**+{(a1.mean()-a2.mean())*100:.1f}** | {g1.mean()*100:.1f} | "
                    f"{g2.mean()*100:.1f} | +{(g1.mean()-g2.mean())*100:.1f} |")

    (OUT_DIR / "phase3_probe_clean.json").write_text(
        json.dumps({"n_seeds": N_SEEDS, "n_folds": N_FOLDS, "n_images": len(y),
                    "results": {k: {kk: (list(vv) if isinstance(vv, list) else vv)
                                    for kk, vv in v.items()} for k, v in res.items()}},
                   indent=2, default=float))
    md = ("# Phase 3 (clean) — leaky S1 vs group-disjoint S2, raw-only corpus\n\n"
          f"DINOv2 ViT-S/14 frozen features, {len(y)} raw closed-set images, "
          f"{len(set(g))} leaf groups. S1 = {N_SEEDS} random splits; "
          f"S2 = {N_FOLDS}-fold StratifiedGroupKFold.\n\n"
          "| probe | S1 acc % | S2 acc % | acc inflation | S1 mF1 | S2 mF1 | mF1 infl |\n"
          "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    (OUT_DIR / "phase3_probe_clean.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
