"""Task 2.4 — THE GATE.  Compute the leakage rate rho for a standard random split.

For N_RANDOM_SPLITS (=10) independent uniform 70/15/15 splits over IMAGES
(ignoring groups, exactly as prior work does), compute

    rho = (# test images whose group also appears in train) / |test|      (paper Eq. 2)

Report mean +/- sd and every per-split value.

Writes outputs/leakage_report.json.

GATE:
    mean_rho > 0.30   -> PROCEED (strong headline)
    0.10..0.30        -> moderate; adjust framing
    mean_rho < 0.05   -> STOP, rethink the angle
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (N_RANDOM_SPLITS, OUT_DIR, SPLIT_FRACS, require, save_json)


def one_split_rho(groups: np.ndarray, rng: np.random.Generator) -> float:
    n = len(groups)
    perm = rng.permutation(n)
    n_tr = int(SPLIT_FRACS[0] * n)
    n_va = int(SPLIT_FRACS[1] * n)
    tr = perm[:n_tr]
    te = perm[n_tr + n_va:]
    train_groups = set(groups[tr].tolist())
    leaked = np.fromiter((g in train_groups for g in groups[te]), dtype=bool, count=len(te))
    return float(leaked.mean())


def main():
    g = pd.read_csv(require(OUT_DIR / "groups.csv", "07_similarity_graph.py"))
    man = pd.read_csv(require(OUT_DIR / "master_manifest.csv", "03_build_manifest.py"))
    df = g.merge(man[["image_path", "unified_label", "split_hint", "source_id"]],
                 on="image_path", how="left")
    # closed-set only, and exclude OOD-control sources from the headline number
    df = df[df["split_hint"] == "closed_set"]
    df = df[~df["source_id"].isin(["D4", "D5"])]
    groups = df["group_id"].to_numpy()

    rng = np.random.default_rng(0)
    rhos = [one_split_rho(groups, rng) for _ in range(N_RANDOM_SPLITS)]
    mean_rho, std_rho = float(np.mean(rhos)), float(np.std(rhos, ddof=1))

    # context numbers
    vc = pd.Series(groups).value_counts()
    frac_multi = float((vc[vc > 1].sum()) / len(groups))

    report = {
        "mean_rho": round(mean_rho, 4),
        "std_rho": round(std_rho, 4),
        "per_split_rho": [round(x, 4) for x in rhos],
        "n_splits": N_RANDOM_SPLITS,
        "split_fracs": SPLIT_FRACS,
        "n_images_scored": int(len(groups)),
        "n_groups": int(vc.size),
        "frac_images_in_multi_image_group": round(frac_multi, 4),
        "sources_included": sorted(df["source_id"].dropna().unique().tolist()),
    }
    save_json(report, OUT_DIR / "leakage_report.json")

    print("\n" + "=" * 56)
    print(f"  LEAKAGE RATE  rho = {mean_rho:.3f} +/- {std_rho:.3f}   "
          f"(n={N_RANDOM_SPLITS} random 70/15/15 splits)")
    print(f"  per-split: {[round(x,3) for x in rhos]}")
    print("=" * 56)
    if mean_rho > 0.30:
        print("  GATE: PROCEED. Strong headline result. Continue to 2.5 / 2.6 and Phase 3.")
    elif mean_rho >= 0.10:
        print("  GATE: MODERATE. Real but smaller correction — reframe as "
              "'the protocol matters even when the gap is modest'.")
    else:
        print("  GATE: STOP. rho < 0.05 region — rethink the angle before Phase 2+.")
    print("=" * 56)
    print("\n  -> paste mean/std into manuscript Table tab:rq2 caption and S1 row of tab:splits")


if __name__ == "__main__":
    main()
