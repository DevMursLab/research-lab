"""Task 2.3 — build the similarity graph and recover leaf-instance groups.

Edges:
  (a) exact-duplicate  : same MD5
  (b) near-duplicate    : pHash Hamming distance <= theta_p   (common.THETA_P)
  (c) semantic near-dup : embedding cosine >= theta_e         (kNN, common.KNN_K)

Connected components (union-find) = leaf-instance groups.

Writes outputs/groups.csv: image_path, group_id, group_size
Prints group-size histogram + how many images fall in a multi-image group.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from common import (KNN_K, OUT_DIR, THETA_E, THETA_P, require, save_json)


class UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def phash_int(h):
    try:
        return int(str(h), 16)
    except Exception:  # noqa: BLE001
        return None


def knn_edges(X, k):
    """Cosine kNN. Uses faiss if available, else sklearn brute force."""
    try:
        import faiss
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        _, I = index.search(X, k + 1)
        for i, row in enumerate(I):
            for j in row:
                if j != -1 and j != i:
                    yield i, int(j)
        return
    except Exception:  # noqa: BLE001
        pass
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="cosine")
    nn.fit(X)
    _, I = nn.kneighbors(X)
    for i, row in enumerate(I):
        for j in row:
            if j != i:
                yield i, int(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta_p", type=int, default=THETA_P)
    ap.add_argument("--theta_e", type=float, default=THETA_E)
    ap.add_argument("--knn", type=int, default=KNN_K)
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip embedding edges (pHash/MD5 only)")
    args = ap.parse_args()

    man = pd.read_csv(require(OUT_DIR / "master_manifest.csv", "03_build_manifest.py"))
    hashes = pd.read_csv(require(OUT_DIR / "hashes.csv", "05_hashes.py"))
    df = man.merge(hashes[["image_path", "md5", "phash_hex"]], on="image_path", how="left")

    paths = df["image_path"].tolist()
    pos = {p: i for i, p in enumerate(paths)}
    n = len(paths)
    uf = UF(n)
    n_md5 = n_ph = n_emb = 0

    # (a) exact duplicates
    for _, grp in df[df["md5"].notna() & (df["md5"] != "ERROR")].groupby("md5"):
        ids = [pos[p] for p in grp["image_path"]]
        for j in ids[1:]:
            uf.union(ids[0], j)
            n_md5 += 1

    # (b) pHash near-duplicates — bucket by high bits to avoid full N^2
    ph = [(i, phash_int(h)) for i, h in enumerate(df["phash_hex"]) if phash_int(h) is not None]
    buckets = {}
    for i, v in ph:
        key = v >> 48  # top 16 bits
        buckets.setdefault(key, []).append((i, v))
    # also compare across adjacent buckets by re-bucketing on a second key
    buckets2 = {}
    for i, v in ph:
        key = (v >> 32) & 0xFFFF
        buckets2.setdefault(key, []).append((i, v))
    for bset in (buckets, buckets2):
        for arr in bset.values():
            if len(arr) < 2:
                continue
            for a in range(len(arr)):
                ia, va = arr[a]
                for b in range(a + 1, len(arr)):
                    ib, vb = arr[b]
                    if bin(va ^ vb).count("1") <= args.theta_p:
                        uf.union(ia, ib)
                        n_ph += 1

    # (c) embedding near-duplicates
    if not args.no_embed:
        idx_csv = OUT_DIR / f"embeddings_{args.model}_index.csv"
        npy = OUT_DIR / f"embeddings_{args.model}.npy"
        if idx_csv.exists() and npy.exists():
            eidx = pd.read_csv(idx_csv)["image_path"].tolist()
            X = np.load(npy).astype("float32")
            emap = [pos[p] for p in eidx]
            for i, j in knn_edges(X, args.knn):
                if float(X[i] @ X[j]) >= args.theta_e:
                    uf.union(emap[i], emap[j])
                    n_emb += 1
        else:
            print(f"!! no embeddings for '{args.model}' — run 06_embeddings.py. "
                  f"Proceeding with pHash/MD5 only (ρ will be an UNDER-estimate).")

    # native leaf_id override for D6
    if "original_label" in df.columns:
        d6 = df[df["source_id"] == "D6"]
        # convention: D6 filenames encode leaf id as  <leafid>_<n>.jpg  -> group by prefix
        pref = d6["image_path"].str.rsplit("/", n=1).str[-1].str.split("_").str[0]
        for _, g in pd.DataFrame({"p": d6["image_path"], "k": pref}).groupby("k"):
            ids = [pos[p] for p in g["p"]]
            for j in ids[1:]:
                uf.union(ids[0], j)

    comp = [uf.find(i) for i in range(n)]
    remap = {c: k for k, c in enumerate(sorted(set(comp)))}
    gid = [remap[c] for c in comp]
    size = Counter(gid)
    out = pd.DataFrame({"image_path": paths, "group_id": gid,
                        "group_size": [size[g] for g in gid]})
    out.to_csv(OUT_DIR / "groups.csv", index=False)

    n_groups = len(size)
    multi = sum(v for v in size.values() if v > 1)
    hist = Counter(size.values())
    save_json({
        "n_images": n, "n_groups": n_groups,
        "n_images_in_multi_image_group": multi,
        "frac_images_in_multi_image_group": round(multi / n, 4) if n else 0,
        "edges": {"md5": n_md5, "phash<=%d" % args.theta_p: n_ph,
                  "emb>=%.3f" % args.theta_e: n_emb},
        "group_size_histogram": dict(sorted(hist.items())),
        "largest_groups": [s for _, s in size.most_common(10)],
        "params": {"theta_p": args.theta_p, "theta_e": args.theta_e, "knn": args.knn},
    }, OUT_DIR / "groups_summary.json")

    print(f"\nwrote outputs/groups.csv")
    print(f"images={n}  groups={n_groups}  images in a multi-image group={multi} "
          f"({100*multi/n:.1f}%)")
    print(f"edges: md5={n_md5}  phash<= {args.theta_p}={n_ph}  emb>= {args.theta_e}={n_emb}")
    print(f"largest 10 group sizes: {[s for _, s in size.most_common(10)]}")


if __name__ == "__main__":
    main()
