"""CLEAR-Papaya : PapayaFormer + baselines with calibration & significance.

One kernel so every model sees the SAME folds -> valid paired tests.

Models: resnet50, tf_efficientnet_b0, mobilevit_s (baselines) + PapayaFormer
  PapayaFormer = MobileViT-S multi-stage features (stages 2/3/4) each passed
  through a Multi-Scale Lesion Attention (MSLA) module, GAP-concat, evidential
  Dirichlet head. Trained with digamma EDL loss + annealed KL-to-uniform +
  class-balanced focal + MSLA attention regularisation.

Per (condition, model): S1 x2 random splits, S2 x3 group-disjoint folds.
Metrics per run: accuracy, macro-F1, ECE(15-bin), Brier. For PapayaFormer on
S2 also: uncertainty-threshold (tau @ val TPR 0.95) coverage & selective risk.

Writes /kaggle/working/pf_results.json (incremental) + pf_results.md +
pf_stats.json (Holm-corrected paired t, PapayaFormer vs each baseline, S2).
"""
import json, time, os, sys, subprocess, glob, math

import torch
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7 \
        and os.environ.get("_ARCH_FIX") != "1":
    print("P100/sm_60 -> installing torch 2.5.1", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1", "torchvision==0.20.1"], check=False)
    os.environ["_ARCH_FIX"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np, pandas as pd, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
from scipy import stats as sps


def resolve_inp():
    for i in range(180):
        hits = glob.glob("/kaggle/input/**/path_map.csv", recursive=True)
        if hits:
            d = os.path.dirname(hits[0]); print("INP =", d, flush=True); return d
        if i % 6 == 0:
            print(f"[{i*5}s] waiting for dataset...", flush=True)
        time.sleep(5)
    raise RuntimeError("dataset not found")


INP = resolve_inp()
OUT = "/kaggle/working"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["anthracnose", "bacterial_leaf_spot", "healthy",
           "mite_or_deficiency", "powdery_mildew", "prsv"]
K = len(CLASSES)
C2I = {c: i for i, c in enumerate(CLASSES)}
RES = 224
EPOCHS = 12
BATCH = 64
BASELINES = ["resnet50", "tf_efficientnet_b0", "mobilevit_s"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ----------------------------------------------------------------- data
def load_condition(cond):
    pm = pd.read_csv(f"{INP}/path_map.csv")
    if cond == "raw_only":
        man = pd.read_csv(f"{INP}/master_manifest_rawonly.csv")
        grp = pd.read_csv(f"{INP}/groups_rawonly.csv")[["image_path", "group_id"]]
    else:
        man = pd.read_csv(f"{INP}/master_manifest.csv")
        man = man[~man.source_id.isin(["D4", "D5"])]
        grp = pd.read_csv(f"{INP}/groups.csv")[["image_path", "group_id"]]
    df = man.merge(grp, on="image_path").merge(
        pm.rename(columns={"orig_path": "image_path"}), on="image_path")
    df = df[df.split_hint.eq("closed_set") & df.unified_label.isin(CLASSES)].reset_index(drop=True)
    return df


class DS(Dataset):
    def __init__(self, df, train):
        self.f = df.bundle_path.tolist()
        self.y = df.unified_label.map(C2I).to_numpy()
        self.train = train

    def __len__(self): return len(self.f)

    def __getitem__(self, i):
        im = Image.open(f"{INP}/{self.f[i]}").convert("RGB")
        a = torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1) / 255.0
        if self.train:
            t = np.random.randint(0, a.shape[1] - RES + 1)
            l = np.random.randint(0, a.shape[2] - RES + 1)
            a = a[:, t:t + RES, l:l + RES]
            if np.random.rand() < 0.5:
                a = torch.flip(a, [2])
        else:
            o = (a.shape[1] - RES) // 2
            a = a[:, o:o + RES, o:o + RES]
        return (a - MEAN) / STD, int(self.y[i])


# ----------------------------------------------------------------- MSLA + PapayaFormer
class MSLA(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.d = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, c, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(c))
            for d in (1, 2, 3)])
        self.spatial = nn.Conv2d(3 * c, 1, 1)
        self.mlp = nn.Sequential(nn.Linear(c, c // r), nn.GELU(), nn.Linear(c // r, c))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_M = None

    def forward(self, x):
        cat = torch.cat([b(x) for b in self.d], 1)
        M = torch.sigmoid(self.spatial(cat))            # B,1,H,W
        s = torch.sigmoid(self.mlp(x.mean((2, 3))))     # B,C
        self.last_M = M
        return x + self.gamma * (x * M * s[:, :, None, None])


class PapayaFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = timm.create_model("mobilevit_s", pretrained=True,
                                    features_only=True, out_indices=(2, 3, 4))
        chs = self.bb.feature_info.channels()
        self.msla = nn.ModuleList([MSLA(c) for c in chs])
        self.head = nn.Linear(sum(chs), K)

    def forward(self, x):
        feats = self.bb(x)
        pooled = []
        self.Ms = []
        for f_, m in zip(feats, self.msla):
            y = m(f_)
            pooled.append(y.mean((2, 3)))
            self.Ms.append(m.last_M)
        e = F.softplus(self.head(torch.cat(pooled, 1)))   # evidence >= 0
        return e

    def attn_reg(self):
        loss = 0.0
        for M in self.Ms:
            loss = loss + M.abs().mean()
            tv = (M[:, :, 1:, :] - M[:, :, :-1, :]).abs().mean() + \
                 (M[:, :, :, 1:] - M[:, :, :, :-1]).abs().mean()
            loss = loss + 0.1 * tv
        return loss / len(self.Ms)


def edl_loss(e, y1h, kl_w):
    alpha = e + 1.0
    S = alpha.sum(1, keepdim=True)
    # digamma term
    L = (y1h * (torch.digamma(S) - torch.digamma(alpha))).sum(1)
    # KL(Dir(alpha_tilde) || Dir(1)) with true-class evidence removed
    at = y1h + (1 - y1h) * alpha
    St = at.sum(1, keepdim=True)
    kl = (torch.lgamma(St).squeeze(1) - torch.lgamma(at).sum(1)
          + torch.lgamma(torch.tensor(float(K), device=e.device))
          + ((at - 1) * (torch.digamma(at) - torch.digamma(St))).sum(1))
    return (L + kl_w * kl).mean()


def cb_focal(p, y, n_k, beta=0.9999, gf=2.0):
    w = (1 - beta) / (1 - beta ** n_k)
    w = w / w.sum() * K
    pt = p.gather(1, y[:, None]).clamp_(1e-6, 1.0).squeeze(1)
    return (w[y] * (1 - pt) ** gf * -torch.log(pt)).mean()


# ----------------------------------------------------------------- metrics
def ece(prob, y, n_bins=15):
    conf = prob.max(1)
    pred = prob.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(prob, y):
    oh = np.eye(K)[y]
    return float(((prob - oh) ** 2).sum(1).mean())


# ----------------------------------------------------------------- train/eval
def make_loaders(df, tr, te):
    return (DataLoader(DS(df.iloc[tr], True), BATCH, shuffle=True, num_workers=2,
                       pin_memory=True, drop_last=True),
            DataLoader(DS(df.iloc[te], False), BATCH, shuffle=False, num_workers=2))


def run(model_name, df, tr, te, n_k, tag, want_reject=False, va=None):
    is_pf = model_name == "PapayaFormer"
    try:
        if is_pf:
            net = PapayaFormer().to(DEV)
        else:
            net = timm.create_model(model_name, pretrained=True, num_classes=K).to(DEV)
    except Exception as ex:
        print(f"  !! skip {model_name}: {ex}", flush=True); return None
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    warm = 3
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warm if ep < warm else
        0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, EPOCHS - warm))))
    scaler = torch.cuda.amp.GradScaler()
    ce = nn.CrossEntropyLoss(
        weight=(len(n_k) * (1 / n_k) / (1 / n_k).sum()).to(DEV), label_smoothing=0.1)
    n_k_t = n_k.to(DEV)
    tl, el = make_loaders(df, tr, te)
    best = None
    for ep in range(EPOCHS):
        net.train()
        kl_w = min(1.0, ep / 8.0)
        t0 = time.time()
        for x, y in tl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                out = net(x)
            if is_pf:
                e = out.float()                      # EDL maths in fp32
                p = (e + 1) / (e + 1).sum(1, keepdim=True)
                y1h = F.one_hot(y, K).float()
                loss = edl_loss(e, y1h, kl_w) + 1.0 * cb_focal(p, y, n_k_t) \
                    + 0.01 * net.attn_reg().float()
            else:
                loss = ce(out.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        net.eval(); P = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for x, _ in el:
                if is_pf:
                    e = net(x.to(DEV)).float()
                    P.append(((e + 1) / (e + 1).sum(1, keepdim=True)).cpu().numpy())
                else:
                    P.append(torch.softmax(net(x.to(DEV)).float(), 1).cpu().numpy())
        prob = np.concatenate(P)
        yte = df.iloc[te].unified_label.map(C2I).to_numpy()
        acc = accuracy_score(yte, prob.argmax(1))
        f1 = f1_score(yte, prob.argmax(1), average="macro")
        if best is None or f1 > best["mF1"]:
            best = {"acc": float(acc), "mF1": float(f1),
                    "ece": ece(prob, yte), "brier": brier(prob, yte)}
        print(f"  [{tag}] {model_name} ep{ep+1}/{EPOCHS} acc={acc:.3f} "
              f"mF1={f1:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # PapayaFormer rejection metrics on request (needs val fold for tau)
    if is_pf and want_reject and va is not None:
        net.eval()
        def uvec(idx):
            dl = DataLoader(DS(df.iloc[idx], False), BATCH, shuffle=False, num_workers=2)
            us, ps, ys = [], [], []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for x, y in dl:
                    e = net(x.to(DEV)).float()
                    S = (e + 1).sum(1)
                    us.append((K / S).cpu().numpy())
                    ps.append(((e + 1) / (e + 1).sum(1, keepdim=True)).cpu().numpy())
                    ys.append(y.numpy())
            return np.concatenate(us), np.concatenate(ps), np.concatenate(ys)
        uv, _, _ = uvec(va)
        tau = float(np.quantile(uv, 0.95))
        ut, pt, yt = uvec(te)
        keep = ut <= tau
        best["tau"] = tau
        best["coverage"] = float(keep.mean())
        best["selective_acc"] = float(accuracy_score(yt[keep], pt[keep].argmax(1))) if keep.any() else 0.0
    return best


def holm(pairs):
    """pairs: list of (name, pval). return dict name-> (p, p_holm, sig)."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs); out = {}
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pairs[i][1])
        running = max(running, adj)
        out[pairs[i][0]] = {"p": pairs[i][1], "p_holm": running, "sig": running < 0.05}
    return out




SEEDS = [0, 1, 2, 3, 4]
FOLDS = 5


def main():
    print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
    all_models = BASELINES + ["PapayaFormer"]
    # per (cond, model): dict (seed,fold) -> {acc, mF1, ece, brier}
    cell = {}
    rows = []
    for cond in ["raw_only"]:
        df = load_condition(cond)
        y = df.unified_label.map(C2I).to_numpy()
        g = df.group_id.to_numpy()
        n_k = torch.tensor([(y == i).sum() for i in range(K)], dtype=torch.float32)
        print(f"\n==== {cond} N={len(df)} ====", flush=True)
        for s in SEEDS:
            folds = list(StratifiedGroupKFold(n_splits=FOLDS, shuffle=True,
                                              random_state=s).split(df, y, groups=g))
            for k, (tr, te) in enumerate(folds):
                assert not (set(g[tr]) & set(g[te]))
                for m in all_models:
                    r = run(m, df, tr, te, n_k, f"{cond}/s{s}f{k}/{m}")
                    if not r:
                        continue
                    cell.setdefault((cond, m), {})[(s, k)] = r
                    json.dump({f"{c}|{mm}": {f"{a}_{b}": v for (a, b), v in d.items()}
                               for (c, mm), d in cell.items()},
                              open(f"{OUT}/sig_cells.json", "w"), default=str)

    # aggregate + paired tests (PapayaFormer vs each baseline), per condition
    summary, stats = [], {}
    for cond in ["raw_only"]:
        pf = cell.get((cond, "PapayaFormer"), {})
        for m in all_models:
            d = cell.get((cond, m), {})
            if not d:
                continue
            acc = np.array([v["acc"] for v in d.values()])
            f1 = np.array([v["mF1"] for v in d.values()])
            ec = np.array([v["ece"] for v in d.values()])
            br = np.array([v["brier"] for v in d.values()])
            summary.append({"condition": cond, "model": m, "n_runs": int(len(d)),
                            "acc": [float(acc.mean()), float(acc.std())],
                            "mF1": [float(f1.mean()), float(f1.std())],
                            "ece": [float(ec.mean()), float(ec.std())],
                            "brier": [float(br.mean()), float(br.std())]})
        pairs_t, pairs_w = [], []
        for b in BASELINES:
            d = cell.get((cond, b), {})
            keys = [key for key in pf if key in d]
            if len(keys) < 3:
                continue
            a = np.array([pf[key]["acc"] for key in keys])
            c = np.array([d[key]["acc"] for key in keys])
            pt = float(sps.ttest_rel(a, c).pvalue)
            try:
                pw = float(sps.wilcoxon(a, c).pvalue)
            except Exception:
                pw = float("nan")
            pairs_t.append((b, pt)); pairs_w.append((b, pw))
        stats[cond] = {"n_pairs": len(keys) if pairs_t else 0,
                       "ttest": dict(pairs_t), "ttest_holm": holm(pairs_t) if pairs_t else {},
                       "wilcoxon": dict(pairs_w)}
    json.dump({"summary": summary, "stats": stats},
              open(f"{OUT}/sig_results.json", "w"), indent=2)

    L = [f"# Significance run: {len(SEEDS)} seeds x {FOLDS} folds S2", "",
         "| condition | model | runs | acc | mF1 | ECE | Brier |",
         "|---|---|---|---|---|---|---|"]
    for r in summary:
        L.append(f"| {r['condition']} | {r['model']} | {r['n_runs']} | "
                 f"{r['acc'][0]*100:.1f}±{r['acc'][1]*100:.1f} | "
                 f"{r['mF1'][0]*100:.1f} | {r['ece'][0]:.3f} | {r['brier'][0]:.3f} |")
    L += ["", "## PapayaFormer vs baselines (paired, Holm)"]
    for cond, s in stats.items():
        L.append(f"### {cond}  (n={s['n_pairs']} paired folds)")
        for b, hd in s.get("ttest_holm", {}).items():
            w = s["wilcoxon"].get(b, float('nan'))
            L.append(f"- vs {b}: t p={hd['p']:.4f} (Holm {hd['p_holm']:.4f}, "
                     f"{'sig' if hd['sig'] else 'n.s.'}); Wilcoxon p={w:.4f}")
    open(f"{OUT}/sig_results.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
