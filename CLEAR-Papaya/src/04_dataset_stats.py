"""Task 1.5 — basic dataset statistics (fills paper Table 5 / Table 'dist').

Reads outputs/master_manifest.csv. Reports, per source and per class:
image count, resolution (min/max/median W,H), aspect-ratio distribution,
file-size distribution. Writes outputs/dataset_stats.json + prints a LaTeX-ready
class-distribution table.
"""
from __future__ import annotations

import statistics as st
from collections import defaultdict

import pandas as pd
from PIL import Image

from common import OUT_DIR, ROOT, CANONICAL_CLASSES, save_json, require


def q(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    i = min(len(vals) - 1, int(p * (len(vals) - 1)))
    return vals[i]


def main():
    man = require(OUT_DIR / "master_manifest.csv", "03_build_manifest.py")
    df = pd.read_csv(man)

    per = defaultdict(lambda: {"w": [], "h": [], "ar": [], "kb": []})
    for r in df.itertuples(index=False):
        fp = ROOT / r.image_path
        try:
            with Image.open(fp) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            continue
        kb = fp.stat().st_size / 1024
        for key in ((r.source_id, "ALL"), (r.source_id, r.unified_label), ("ALL", r.unified_label), ("ALL", "ALL")):
            d = per[key]
            d["w"].append(w); d["h"].append(h)
            d["ar"].append(round(w / h, 3) if h else 0)
            d["kb"].append(round(kb, 1))

    stats = {}
    for (src, cls), d in sorted(per.items()):
        n = len(d["w"])
        stats[f"{src}/{cls}"] = {
            "n": n,
            "width":  {"min": min(d["w"]), "median": int(st.median(d["w"])), "max": max(d["w"])} if n else None,
            "height": {"min": min(d["h"]), "median": int(st.median(d["h"])), "max": max(d["h"])} if n else None,
            "aspect_ratio": {"min": min(d["ar"]), "median": round(st.median(d["ar"]), 3), "max": max(d["ar"])} if n else None,
            "file_kb": {"min": min(d["kb"]), "median": round(st.median(d["kb"]), 1),
                        "p95": q(d["kb"], 0.95), "max": max(d["kb"])} if n else None,
        }
    save_json(stats, OUT_DIR / "dataset_stats.json")

    # ---- LaTeX class-distribution table (paper Table 'tab:dist') ----
    sources = sorted(df["source_id"].unique())
    print("\n% ---- paste into manuscript Table tab:dist ----")
    piv = df[df["split_hint"] == "closed_set"].pivot_table(
        index="unified_label", columns="source_id", values="image_path",
        aggfunc="count", fill_value=0)
    total_all = int(piv.values.sum())
    for cls in CANONICAL_CLASSES:
        if cls not in piv.index:
            cells = " & ".join(["0"] * len(sources))
            print(f"{cls} & {cells} & 0 & 0.0 \\\\")
            continue
        row = piv.loc[cls]
        tot = int(row.sum())
        cells = " & ".join(str(int(row.get(s, 0))) for s in sources)
        pct = 100 * tot / total_all if total_all else 0
        print(f"{cls} & {cells} & {tot} & {pct:.1f} \\\\")
    col_tot = " & ".join(str(int(piv[s].sum())) for s in sources)
    print(f"\\textbf{{Total}} & {col_tot} & {total_all} & 100 \\\\")
    if total_all:
        counts = piv.sum(axis=1)
        ir = counts.max() / max(1, counts.min())
        print(f"% Imbalance ratio IR = {ir:.2f}")


if __name__ == "__main__":
    main()
