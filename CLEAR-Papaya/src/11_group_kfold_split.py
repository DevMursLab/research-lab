"""Task 2.6 — create the group-stratified 5-fold split (S2, the primary protocol).

StratifiedGroupKFold with group = group_id (leaf instance), preserving class
balance. Verifies rho = 0 for every fold (no group spans folds -> no leakage).

Writes outputs/splits_S2.csv: image_path, fold  (0..4)
plus outputs/splits_S2_summary.json with per-fold class counts and verified rho.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import N_FOLDS, OUT_DIR, SEED, require, save_json


def main():
    g = pd.read_csv(require(OUT_DIR / "groups.csv", "07_similarity_graph.py"))
    man = pd.read_csv(require(OUT_DIR / "master_manifest.csv", "03_build_manifest.py"))
    df = g.merge(man[["image_path", "unified_label", "split_hint", "source_id"]],
                 on="image_path", how="left")
    df = df[(df["split_hint"] == "closed_set") & (~df["source_id"].isin(["D4", "D5"]))].reset_index(drop=True)
    if df.empty:
        print("no closed-set rows — nothing to split.")
        return

    y = df["unified_label"].astype("category").cat.codes.to_numpy()
    grp = df["group_id"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, test_idx) in enumerate(sgkf.split(df, y, grp)):
        fold[test_idx] = k
    df["fold"] = fold

    # verify: no group appears in two folds
    spanning = (df.groupby("group_id")["fold"].nunique() > 1).sum()

    # verify rho = 0 for each held-out fold vs the rest
    per_fold_rho = {}
    for k in range(N_FOLDS):
        te = set(df.loc[df.fold == k, "group_id"])
        tr = set(df.loc[df.fold != k, "group_id"])
        inter = te & tr
        n_te = int((df.fold == k).sum())
        leaked = int(df.loc[df.fold == k, "group_id"].isin(inter).sum())
        per_fold_rho[k] = round(leaked / n_te, 6) if n_te else 0.0

    df[["image_path", "fold"]].to_csv(OUT_DIR / "splits_S2.csv", index=False)

    counts = (df.groupby(["fold", "unified_label"]).size()
              .unstack(fill_value=0).astype(int))
    save_json({
        "n_folds": N_FOLDS,
        "n_images": int(len(df)),
        "groups_spanning_multiple_folds": int(spanning),
        "per_fold_rho": per_fold_rho,
        "all_folds_rho_zero": all(v == 0 for v in per_fold_rho.values()),
        "per_fold_class_counts": counts.to_dict(orient="index"),
    }, OUT_DIR / "splits_S2_summary.json")

    print("wrote outputs/splits_S2.csv")
    print(f"groups spanning >1 fold: {spanning}  (must be 0)")
    print(f"per-fold rho: {per_fold_rho}  (all must be 0.0)")
    print("\nper-fold class counts:")
    print(counts)


if __name__ == "__main__":
    main()
