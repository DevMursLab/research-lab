"""Task 2.5 — visualise the largest duplicate clusters (paper Figure 2).

Plots the images of the 10 largest groups, one row per group, as a grid.
This is the reviewer-facing "same leaf, different photo" evidence.

Writes outputs/duplicate_clusters_top10.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from common import OUT_DIR, ROOT, require

MAX_GROUPS = 10
MAX_PER_ROW = 8


def main():
    g = pd.read_csv(require(OUT_DIR / "groups.csv", "07_similarity_graph.py"))
    man = pd.read_csv(require(OUT_DIR / "master_manifest.csv", "03_build_manifest.py"))
    df = g.merge(man[["image_path", "unified_label", "source_id"]], on="image_path", how="left")

    top = (df.groupby("group_id").size().sort_values(ascending=False)
           .head(MAX_GROUPS).index.tolist())
    if not top or df[df.group_id == top[0]].shape[0] < 2:
        print("No multi-image groups found — nothing to visualise. "
              "(If embeddings were skipped this is expected; re-run 07 with embeddings.)")
        return

    fig, axes = plt.subplots(len(top), MAX_PER_ROW,
                             figsize=(MAX_PER_ROW * 1.6, len(top) * 1.7))
    if len(top) == 1:
        axes = axes.reshape(1, -1)

    for r, gid in enumerate(top):
        rows = df[df.group_id == gid].head(MAX_PER_ROW)
        lbls = rows["unified_label"].unique()
        srcs = rows["source_id"].unique()
        for c in range(MAX_PER_ROW):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(rows):
                fp = ROOT / rows.iloc[c]["image_path"]
                try:
                    ax.imshow(Image.open(fp).convert("RGB").resize((160, 160)))
                except Exception:  # noqa: BLE001
                    pass
            if c == 0:
                ax.set_title(f"g{gid} · n={ (df.group_id==gid).sum() }\n"
                             f"{'/'.join(map(str,lbls))} · {'/'.join(map(str,srcs))}",
                             fontsize=7, loc="left")
    fig.suptitle("Ten largest inferred leaf-instance groups "
                 "(same physical leaf, multiple photographs)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "duplicate_clusters_top10.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    # cross-source / cross-label leakage flags — strong reviewer material
    bad_lbl = [gid for gid in top if df[df.group_id == gid]["unified_label"].nunique() > 1]
    if bad_lbl:
        print(f"!! groups spanning >1 label (annotation inconsistency): {bad_lbl}")
    x_src = [gid for gid in top if df[df.group_id == gid]["source_id"].nunique() > 1]
    if x_src:
        print(f"note: groups spanning >1 source (same image redistributed): {x_src}")


if __name__ == "__main__":
    main()
