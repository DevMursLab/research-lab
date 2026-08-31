"""Phase 3 (CPU-feasible) — the leakage effect, measured.

Trains cheap classifiers on FROZEN DINOv2 features under two protocols:
  S1  = leaky   : image-level random stratified 70/30 split (near-dups cross the line)
  S2  = clean   : group-disjoint split from outputs/splits_S2.csv (no leaf in both sides)

Classifiers (both are standard leakage-sensitive probes):
  * 1-NN cosine     -- maximally exposes duplicate leakage
  * Logistic reg    -- linear probe on L2-normalised features

Repeats over seeds; reports mean +/- std accuracy & macro-F1, and the
S1 - S2 inflation gap.  Writes outputs/phase3_probe_results.json (+ .md table).

    python src/12_probe_s1_vs_s2.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from common import OUT_DIR, SEED

N_SEEDS = 5
TEST_FRAC = 0.30
DROP_LABELS = {"UNMAPPED"}


def load():
    idx = pd.read_csv(OUT_DIR / "embeddings_dinov2_vits14_index.csv")
    X = np.load(OUT_DIR / "embeddings_dinov2_vits14.npy").astype("float32")
    man = pd.read_csv(OUT_DIR / "master_manifest.csv")
    grp = pd.read_csv(OUT_DIR / "groups.csv")[["image_path", "group_id"]]
    s2 = pd.read_csv(OUT_DIR / "splits_S2.csv").rename(columns={"fold": "s2_fold"})

    df = idx.merge(man, on="image_path").merge(grp, on="image_path").merge(s2, on="image_path")
    assert len(df) == len(idx), "join dropped rows"
    keep = ~df["unified_label"].isin(DROP_LABELS)
    df, X = df[keep].reset_index(drop=True), X[keep.values]
    y = df["unified_label"].to_numpy()
    return X, y, df


def _knn1_cos(Xtr, ytr, Xte):
    # features already L2-normalised -> dot product == cosine
    sims = Xte @ Xtr.T
    return ytr[np.argmax(sims, axis=1)]


def _eval(Xtr, ytr, Xte, yte):
    out = {}
    pred = _knn1_cos(Xtr, ytr, Xte)
    out["knn1"] = (accuracy_score(yte, pred), f1_score(yte, pred, average="macro"))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    out["logreg"] = (accuracy_score(yte, pred), f1_score(yte, pred, average="macro"))
    return out


def run_s1(X, y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    tr, te = next(sss.split(X, y))
    return _eval(X[tr], y[tr], X[te], y[te])


def run_s2_fold(X, y, df, te_fold):
    # group-disjoint by construction: no physical leaf spans folds
    te = (df["s2_fold"] == te_fold).to_numpy()
    return _eval(X[~te], y[~te], X[te], y[te])


def main():
    X, y, df = load()
    print(f"N={len(y)}  classes={sorted(set(y))}")
    acc = {("S1", "knn1"): [], ("S1", "logreg"): [],
           ("S2", "knn1"): [], ("S2", "logreg"): []}
    f1 = {k: [] for k in acc}

    # S1: N_SEEDS independent image-level random splits (leaky)
    for s in range(SEED, SEED + N_SEEDS):
        r1 = run_s1(X, y, s)
        for clf in ("knn1", "logreg"):
            acc[("S1", clf)].append(r1[clf][0]); f1[("S1", clf)].append(r1[clf][1])
        print(f"S1 seed {s}: knn1={r1['knn1'][0]:.3f} logreg={r1['logreg'][0]:.3f}")

    # S2: full group-disjoint 5-fold CV (each fold held out once)
    for te_fold in sorted(df["s2_fold"].unique()):
        r2 = run_s2_fold(X, y, df, te_fold)
        for clf in ("knn1", "logreg"):
            acc[("S2", clf)].append(r2[clf][0]); f1[("S2", clf)].append(r2[clf][1])
        print(f"S2 fold {te_fold}: knn1={r2['knn1'][0]:.3f} logreg={r2['logreg'][0]:.3f}")

    res = {}
    rows = []
    for clf in ("knn1", "logreg"):
        a1, a2 = np.array(acc[("S1", clf)]), np.array(acc[("S2", clf)])
        g1, g2 = np.array(f1[("S1", clf)]), np.array(f1[("S2", clf)])
        res[clf] = {
            "S1_acc": [float(a1.mean()), float(a1.std())],
            "S2_acc": [float(a2.mean()), float(a2.std())],
            "acc_inflation": float(a1.mean() - a2.mean()),
            "S1_macroF1": [float(g1.mean()), float(g1.std())],
            "S2_macroF1": [float(g2.mean()), float(g2.std())],
            "f1_inflation": float(g1.mean() - g2.mean()),
        }
        rows.append(f"| {clf} | {a1.mean()*100:.1f} ± {a1.std()*100:.1f} | "
                    f"{a2.mean()*100:.1f} ± {a2.std()*100:.1f} | "
                    f"**+{(a1.mean()-a2.mean())*100:.1f}** | "
                    f"{g1.mean()*100:.1f} | {g2.mean()*100:.1f} | "
                    f"+{(g1.mean()-g2.mean())*100:.1f} |")

    (OUT_DIR / "phase3_probe_results.json").write_text(json.dumps(
        {"n_seeds": N_SEEDS, "test_frac": TEST_FRAC, "n_images": len(y),
         "classes": sorted(set(y)), "results": res}, indent=2))

    md = ("# Phase 3 — frozen-feature probe: leaky (S1) vs clean (S2)\n\n"
          f"DINOv2 ViT-S/14 features, {N_SEEDS} seeds, {len(y)} closed-set images.\n\n"
          "| probe | S1 acc % | S2 acc % | acc inflation | S1 mF1 | S2 mF1 | mF1 inflation |\n"
          "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
          "S1 = image-level random split (near-duplicates leak across train/test).\n"
          "S2 = group-disjoint split (same physical leaf never on both sides).\n"
          "The inflation column is the accuracy a naive protocol reports but a clean one does not.\n")
    (OUT_DIR / "phase3_probe_results.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
