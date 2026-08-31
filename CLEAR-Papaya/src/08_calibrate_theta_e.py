"""Task 2.3 (calibration step) — sample candidate image pairs for manual labelling.

To pick theta_e defensibly you must hand-label ~100-200 pairs as
"same physical leaf" / "different leaf". This script draws a stratified sample of
pairs across a range of cosine similarities so your calibration set is not all
trivial matches.

Output: outputs/pairs_to_label.csv  with columns
    pair_id, path_a, path_b, cosine, phash_hamming, same_leaf   <-- you fill same_leaf (1/0)

Then re-run this script with --score to compute precision/recall of the
"same physical leaf" relation at each candidate theta_e and pick the smallest
theta_e with precision >= 0.95 (paper requirement).
"""
from __future__ import annotations

import argparse
import itertools
import random

import numpy as np
import pandas as pd

from common import OUT_DIR, SEED, require, save_json, THETA_P

BANDS = [(0.995, 1.0), (0.98, 0.995), (0.965, 0.98), (0.95, 0.965),
         (0.92, 0.95), (0.88, 0.92), (0.80, 0.88)]
PER_BAND = 30


def phash_hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def sample_pairs(emb_model: str):
    idx = pd.read_csv(require(OUT_DIR / f"embeddings_{emb_model}_index.csv", "06_embeddings.py"))
    X = np.load(OUT_DIR / f"embeddings_{emb_model}.npy")
    hashes = pd.read_csv(require(OUT_DIR / "hashes.csv", "05_hashes.py")).set_index("image_path")

    rng = random.Random(SEED)
    n = len(idx)
    # candidate neighbours via a random subsample of anchors (full N^2 is wasteful here)
    anchors = rng.sample(range(n), min(n, 4000))
    sims_by_band = {b: [] for b in BANDS}
    Xn = X  # already normalised
    for a in anchors:
        d = Xn @ Xn[a]
        d[a] = -1
        order = np.argpartition(-d, min(25, n - 1))[:25]
        for j in order:
            c = float(d[j])
            for lo, hi in BANDS:
                if lo <= c < hi:
                    sims_by_band[(lo, hi)].append((a, int(j), c))
                    break

    rows = []
    pid = 0
    for band, cand in sims_by_band.items():
        rng.shuffle(cand)
        for a, j, c in cand[:PER_BAND]:
            pa, pb = idx.iloc[a]["image_path"], idx.iloc[j]["image_path"]
            try:
                hd = phash_hamming(hashes.loc[pa, "phash_hex"], hashes.loc[pb, "phash_hex"])
            except Exception:  # noqa: BLE001
                hd = -1
            rows.append({"pair_id": pid, "path_a": pa, "path_b": pb,
                         "cosine": round(c, 4), "phash_hamming": hd, "same_leaf": ""})
            pid += 1
    out = OUT_DIR / "pairs_to_label.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}  ({len(rows)} pairs across {len(BANDS)} similarity bands)")
    print("\nNEXT: open it, view each path_a/path_b side by side, set same_leaf to 1 or 0")
    print("(same physical leaf photographed again = 1). Then: python src/08_calibrate_theta_e.py --score")


def score():
    lab = pd.read_csv(OUT_DIR / "pairs_to_label.csv")
    lab = lab[lab["same_leaf"].astype(str).str.strip().isin(["0", "1"])].copy()
    if len(lab) < 40:
        print(f"only {len(lab)} labelled pairs — label at least ~100 for a credible calibration.")
    lab["same_leaf"] = lab["same_leaf"].astype(int)
    y = lab["same_leaf"].values
    res = []
    for te in [0.90, 0.925, 0.95, 0.975, 0.99]:
        pred = (lab["cosine"].values >= te).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        res.append({"theta_e": te, "precision": round(prec, 4), "recall": round(rec, 4),
                    "tp": tp, "fp": fp, "fn": fn})
        print(f"theta_e={te:.3f}  precision={prec:.3f}  recall={rec:.3f}  (tp={tp} fp={fp} fn={fn})")
    ok = [r for r in res if r["precision"] >= 0.95]
    chosen = min(ok, key=lambda r: r["theta_e"])["theta_e"] if ok else 0.975
    save_json({"n_labelled": len(lab), "grid": res, "chosen_theta_e": chosen,
               "criterion": "smallest theta_e with precision>=0.95", "theta_p": THETA_P},
              OUT_DIR / "theta_e_calibration.json")
    print(f"\n=> chosen theta_e = {chosen}   (pass to 07_similarity_graph.py --theta_e {chosen})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--score", action="store_true", help="score labelled pairs instead of sampling")
    a = ap.parse_args()
    score() if a.score else sample_pairs(a.model)
