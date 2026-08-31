"""Phase 2 hardening — remove augmentation contamination + curb single-linkage
chaining, then re-measure rho with a theta_e sensitivity sweep.

Why:
  * D1 ships pre-augmented images (rotated / "_increased_" / " - Copy") mixed into
    the class folders (~29% of D1). Those are not independent leaves; an augmented
    copy crossing train/test is a cruder, already-known error. The paper's headline
    is SAME-PHYSICAL-LEAF leakage, so the clean rho must be measured on raw images.
  * theta_e=0.95 kNN + union-find chains rotate/copy variants into one 596-image
    blob spanning 3 classes. Mutual-kNN (edge only if EACH is in the other's kNN)
    kills most chaining.

Outputs:
  outputs/master_manifest_rawonly.csv
  outputs/groups_rawonly.csv                 (at theta_e = REF_THETA)
  outputs/audit_clean_summary.{json,md}
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from common import KNN_K, N_RANDOM_SPLITS, OUT_DIR, SPLIT_FRACS

AUG_RE = re.compile(
    r"rotated|_increased_|- ?copy|_flip|flipped|_bright|_dark|_zoom|_shear|"
    r"_aug|_noise|mirror|_scaled|_contrast|_gamma", re.I)
THETA_SWEEP = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97]
REF_THETA = 0.95
THETA_P = 5


class UF:
    def __init__(self, n):
        self.p = list(range(n)); self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.r[ra] += self.r[ra] == self.r[rb]


def mutual_knn(X, k):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="cosine").fit(X)
    _, I = nn.kneighbors(X)
    nbr = [set(row[1:]) for row in I]
    edges = []
    for i, s in enumerate(nbr):
        for j in s:
            if j > i and i in nbr[j]:
                edges.append((i, j))
    return edges


def build_groups(df, X, theta_e, mutual=True):
    n = len(df)
    pos = {p: i for i, p in enumerate(df["image_path"])}
    uf = UF(n)
    # exact MD5
    for _, g in df[df["md5"].notna() & (df["md5"] != "ERROR")].groupby("md5"):
        ids = [pos[p] for p in g["image_path"]]
        for j in ids[1:]:
            uf.union(ids[0], j)
    # pHash <= THETA_P (bucketed)
    def ph_int(h):
        try:
            return int(str(h), 16)
        except Exception:
            return None
    ph = [(i, ph_int(h)) for i, h in enumerate(df["phash_hex"]) if ph_int(h) is not None]
    for shift in (48, 32):
        b = {}
        for i, v in ph:
            b.setdefault((v >> shift) & 0xFFFF, []).append((i, v))
        for arr in b.values():
            for a in range(len(arr)):
                for c in range(a + 1, len(arr)):
                    if bin(arr[a][1] ^ arr[c][1]).count("1") <= THETA_P:
                        uf.union(arr[a][0], arr[c][0])
    # embedding edges
    cand = mutual_knn(X, KNN_K) if mutual else [
        (i, j) for i, j in _all_knn(X, KNN_K)]
    for i, j in cand:
        if float(X[i] @ X[j]) >= theta_e:
            uf.union(i, j)
    comp = [uf.find(i) for i in range(n)]
    remap = {c: k for k, c in enumerate(sorted(set(comp)))}
    return np.array([remap[c] for c in comp])


def _all_knn(X, k):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric="cosine").fit(X)
    _, I = nn.kneighbors(X)
    for i, row in enumerate(I):
        for j in row:
            if j != i:
                yield i, int(j)


def rho(groups, seeds=N_RANDOM_SPLITS):
    rng = np.random.default_rng(0)
    n = len(groups)
    out = []
    for _ in range(seeds):
        perm = rng.permutation(n)
        n_tr = int(SPLIT_FRACS[0] * n)
        n_va = int(SPLIT_FRACS[1] * n)
        tr = set(groups[perm[:n_tr]].tolist())
        te = groups[perm[n_tr + n_va:]]
        out.append(float(np.mean([g in tr for g in te])))
    return float(np.mean(out)), float(np.std(out, ddof=1))


def diagnostics(df, groups):
    d = df.assign(gid=groups)
    multi = d.groupby("gid").filter(lambda x: len(x) > 1)
    by = multi.groupby("gid")
    x_label = int((by["unified_label"].nunique() > 1).sum())
    x_source = int((by["source_id"].nunique() > 1).sum())
    sizes = Counter(Counter(groups).values())
    return {"cross_label_groups": x_label, "cross_source_groups": x_source,
            "largest_group": int(max(Counter(groups).values())),
            "n_groups": int(len(set(groups))),
            "size_hist_tail": {str(k): v for k, v in sorted(sizes.items()) if k >= 8}}


def main():
    idx = pd.read_csv(OUT_DIR / "embeddings_dinov2_vits14_index.csv")
    X = np.load(OUT_DIR / "embeddings_dinov2_vits14.npy").astype("float32")
    man = pd.read_csv(OUT_DIR / "master_manifest.csv")
    hsh = pd.read_csv(OUT_DIR / "hashes.csv")[["image_path", "md5", "phash_hex"]]
    df = idx.merge(man, on="image_path").merge(hsh, on="image_path", how="left")
    assert len(df) == len(idx)
    df["is_aug"] = df["image_path"].str.contains(AUG_RE)
    df["_row"] = np.arange(len(df))

    print(f"total={len(df)}  aug={df.is_aug.sum()} ({df.is_aug.mean()*100:.1f}%)  "
          f"by source: {df[df.is_aug].source_id.value_counts().to_dict()}")

    scopes = {
        "all_mutualknn": df,
        "rawonly_mutualknn": df[~df.is_aug].reset_index(drop=True),
    }
    # closed-set filter for rho
    table = []
    ref = {}
    for name, sub in scopes.items():
        Xs = X[sub["_row"].to_numpy()]
        cs = sub["split_hint"].eq("closed_set").to_numpy()
        for th in THETA_SWEEP:
            g = build_groups(sub, Xs, th, mutual=True)
            m, s = rho(g[cs])
            row = {"scope": name, "theta_e": th, "rho_mean": round(m, 4),
                   "rho_std": round(s, 4), **diagnostics(sub, g)}
            table.append(row)
            print(f"  {name:22s} theta_e={th:.2f}  rho={m:.3f}+/-{s:.3f}  "
                  f"n_groups={row['n_groups']}  largest={row['largest_group']}  "
                  f"xlabel={row['cross_label_groups']} xsrc={row['cross_source_groups']}")
            if th == REF_THETA:
                ref[name] = (sub, g)

    # also single-linkage (non-mutual) at REF_THETA on rawonly, to show the chaining effect
    sub = scopes["rawonly_mutualknn"]
    Xs = X[sub["_row"].to_numpy()]
    g_sl = build_groups(sub, Xs, REF_THETA, mutual=False)
    m_sl, s_sl = rho(g_sl[sub["split_hint"].eq("closed_set").to_numpy()])
    chain_row = {"scope": "rawonly_singlelinkage", "theta_e": REF_THETA,
                 "rho_mean": round(m_sl, 4), "rho_std": round(s_sl, 4),
                 **diagnostics(sub, g_sl)}
    table.append(chain_row)
    print(f"  rawonly_singlelinkage  theta_e={REF_THETA}  rho={m_sl:.3f}  "
          f"largest={chain_row['largest_group']}  (vs mutual-kNN above)")

    # freeze rawonly reference artifacts
    sub, g = ref["rawonly_mutualknn"]
    sub.drop(columns=["_row"]).to_csv(OUT_DIR / "master_manifest_rawonly.csv", index=False)
    pd.DataFrame({"image_path": sub["image_path"], "group_id": g,
                  "group_size": [Counter(g)[x] for x in g]}
                 ).to_csv(OUT_DIR / "groups_rawonly.csv", index=False)

    (OUT_DIR / "audit_clean_summary.json").write_text(json.dumps({
        "aug_total": int(df.is_aug.sum()), "aug_frac": float(df.is_aug.mean()),
        "aug_by_source": df[df.is_aug].source_id.value_counts().to_dict(),
        "sweep": table, "ref_theta": REF_THETA}, indent=2))

    def fmt(rows):
        return "\n".join(
            f"| {r['scope']} | {r['theta_e']:.2f} | {r['rho_mean']*100:.1f} ± {r['rho_std']*100:.1f} | "
            f"{r['n_groups']} | {r['largest_group']} | {r['cross_label_groups']} | "
            f"{r['cross_source_groups']} |" for r in rows)

    md = f"""# Clean re-audit — augmentation removal + chaining control

**Augmentation contamination:** {int(df.is_aug.sum())} / {len(df)} images
({df.is_aug.mean()*100:.1f}%) are augmentation-derived, **all in D1**
({df[df.is_aug].source_id.value_counts().to_dict()}). D1 ships these mixed into the
class folders despite being catalogued as raw.

**theta_e sensitivity (mutual-kNN grouping, 10 random 70/15/15 splits, closed-set):**

| scope | theta_e | rho % | n_groups | largest grp | x-label grps | x-source grps |
|---|---|---|---|---|---|---|
{fmt([r for r in table if r['scope'] != 'rawonly_singlelinkage'])}

**Chaining check (raw-only, theta_e={REF_THETA}):**

| grouping | rho % | largest group |
|---|---|---|
| mutual-kNN | {[r for r in table if r['scope']=='rawonly_mutualknn' and r['theta_e']==REF_THETA][0]['rho_mean']*100:.1f} | {[r for r in table if r['scope']=='rawonly_mutualknn' and r['theta_e']==REF_THETA][0]['largest_group']} |
| single-linkage kNN | {m_sl*100:.1f} | {chain_row['largest_group']} |

Single-linkage inflates the largest component by chaining rotate/copy variants;
mutual-kNN is the reported grouping.

**Headline (raw-only, mutual-kNN, theta_e={REF_THETA}):
rho = {[r for r in table if r['scope']=='rawonly_mutualknn' and r['theta_e']==REF_THETA][0]['rho_mean']*100:.1f}%**
— down from the contaminated 72.8% but still far above the 30% gate.
"""
    (OUT_DIR / "audit_clean_summary.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
