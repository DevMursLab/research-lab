"""Task 1.4 — unified data loader + label mapping.

Walks every data/<source>/ folder. Infers the raw label from the immediate parent
directory name (the near-universal ImageFolder convention). Maps it to a canonical
class via common.LABEL_MAP. Emits outputs/master_manifest.csv:

    image_path, source_id, original_label, unified_label, split_hint

Unmapped labels are written to outputs/unmapped_labels.csv so you can extend
LABEL_MAP in common.py and re-run.
"""
from __future__ import annotations

import csv
from collections import Counter

from common import (DATA_DIR, OUT_DIR, ROOT, canon_label, iter_images,
                    OPEN_SET_LABELS)


def raw_label_for(path, source_root):
    """Immediate parent folder name, unless that's the source root itself."""
    parent = path.parent
    if parent == source_root:
        return "UNLABELLED"
    return parent.name


def main():
    sources = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    if not sources:
        print(f"No source folders under {DATA_DIR}.")
        return

    rows = []
    unmapped = Counter()
    per_source_class = Counter()

    for src in sources:
        for p in iter_images(src):
            raw = raw_label_for(p, src)
            key = " ".join(raw.strip().lower().replace("_", " ").replace("-", " ").split())
            unified = canon_label(raw)
            if unified is None:
                if key in OPEN_SET_LABELS:
                    unified = OPEN_SET_LABELS[key]
                else:
                    unmapped[(src.name, raw)] += 1
                    unified = "UNMAPPED"
            rows.append({
                "image_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "source_id": src.name,
                "original_label": raw,
                "unified_label": unified,
                "split_hint": "open_set" if unified in OPEN_SET_LABELS.values() else "closed_set",
            })
            per_source_class[(src.name, unified)] += 1

    manifest = OUT_DIR / "master_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "source_id", "original_label",
                                          "unified_label", "split_hint"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {manifest}  ({len(rows)} images)")

    if unmapped:
        up = OUT_DIR / "unmapped_labels.csv"
        with up.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_id", "raw_label", "count"])
            for (s, r), c in unmapped.most_common():
                w.writerow([s, r, c])
        print(f"\n!! {sum(unmapped.values())} images have UNMAPPED labels — see {up}")
        print("   Add the raw strings to LABEL_MAP in src/common.py and re-run this script.")

    print("\nper-source / per-class counts:")
    for (s, cls), c in sorted(per_source_class.items()):
        print(f"  {s:8s} {cls:22s} {c:6d}")


if __name__ == "__main__":
    main()
