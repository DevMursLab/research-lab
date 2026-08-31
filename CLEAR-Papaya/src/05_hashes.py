"""Task 2.1 — MD5 + perceptual hash (pHash) for every image.

MD5   -> exact duplicates.
pHash -> near-duplicates (resize / re-compress / slight crop).  64-bit DCT hash.
Hamming-distance threshold theta_p = 5 (common.THETA_P) is applied later in
07_similarity_graph.py.

Writes outputs/hashes.csv: image_path, md5, phash_hex, width, height
"""
from __future__ import annotations

import csv
import hashlib

import imagehash
from PIL import Image
from tqdm import tqdm

from common import OUT_DIR, ROOT, require
import pandas as pd


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    man = require(OUT_DIR / "master_manifest.csv", "03_build_manifest.py")
    df = pd.read_csv(man)
    out = OUT_DIR / "hashes.csv"

    done = set()
    if out.exists():
        done = set(pd.read_csv(out)["image_path"])
        print(f"resuming — {len(done)} already hashed")

    mode = "a" if done else "w"
    with out.open(mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["image_path", "md5", "phash_hex", "width", "height"])
        for rel in tqdm(df["image_path"], desc="hashing"):
            if rel in done:
                continue
            fp = ROOT / rel
            try:
                md5 = md5_of(fp)
                with Image.open(fp) as im:
                    im = im.convert("RGB")
                    ph = imagehash.phash(im, hash_size=8)  # 64-bit
                    wd, ht = im.size
                w.writerow([rel, md5, str(ph), wd, ht])
            except Exception as e:  # noqa: BLE001
                w.writerow([rel, "ERROR", f"ERROR:{type(e).__name__}", "", ""])

    d = pd.read_csv(out)
    n_exact_dupe = int(d["md5"].duplicated(keep=False).sum())
    print(f"\nwrote {out}  ({len(d)} rows)")
    print(f"exact-duplicate images (same MD5): {n_exact_dupe}")


if __name__ == "__main__":
    main()
