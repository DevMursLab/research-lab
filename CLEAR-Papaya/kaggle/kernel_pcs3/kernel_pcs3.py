"""#9 per-class results table + #10 cross-source S3 per-class breakdown.

A. Train PapayaFormer + MobileViT-S on raw S2 (fold 0, seed 0), 14 epochs.
   Report per-class precision / recall / F1 / support / one-vs-rest AUROC on the
   held-out fold, plus the normalised confusion matrix (-> figs/fig_cm.png).

B. Cross-source S3: train each on D1 u D2 (raw closed-set), test on D3.
   Report per-class F1 in-dist vs cross-source, and the D3 confusion matrix, to
   show whether the collapse is uniform (domain shift) or class-specific (label
   drift between corpora).

Writes /kaggle/working/pcs3.json + pcs3.md + fig_cm.png + fig_s3_cm.png .
"""
import json, os, sys, subprocess, glob, time, math

import torch
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7 \
        and os.environ.get("_ARCH_FIX") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1", "torchvision==0.20.1"], check=False)
    os.environ["_ARCH_FIX"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np, pandas as pd, torch.nn as nn, torch.nn.functional as F, timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             confusion_matrix, f1_score, accuracy_score)
from sklearn.model_selection import StratifiedGroupKFold


def resolve_inp():
    for _ in range(180):
        h = glob.glob("/kaggle/input/**/path_map.csv", recursive=True)
        if h:
            return os.path.dirname(h[0])
        time.sleep(5)
    raise RuntimeError("no dataset")


INP = resolve_inp()
OUT = "/kaggle/working"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["anthracnose", "bacterial_leaf_spot", "healthy",
           "mite_or_deficiency", "powdery_mildew", "prsv"]
SHORT = ["anthr", "bact", "healthy", "mite/def", "powd", "prsv"]
K = len(CLASSES)
C2I = {c: i for i, c in enumerate(CLASSES)}
RES, EPOCHS, BATCH = 224, 14, 64
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEV)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEV)


def load(sources=None, rawonly=True):
    pm = pd.read_csv(f"{INP}/path_map.csv")
    man = pd.read_csv(f"{INP}/master_manifest_rawonly.csv" if rawonly else f"{INP}/master_manifest.csv")
    gf = "groups_rawonly.csv" if rawonly else "groups.csv"
    grp = pd.read_csv(f"{INP}/{gf}")[["image_path", "group_id"]]
    df = man.merge(grp, on="image_path").merge(
        pm.rename(columns={"orig_path": "image_path"}), on="image_path")
    df = df[df.split_hint.eq("closed_set") & df.unified_label.isin(CLASSES)]
    if sources:
        df = df[df.source_id.isin(sources)]
    return df.reset_index(drop=True)


class DS(Dataset):
    def __init__(self, df, train):
        self.bp = df.bundle_path.tolist(); self.y = df.unified_label.map(C2I).to_numpy(); self.train = train

    def __len__(self): return len(self.bp)

    def __getitem__(self, i):
        a = torch.from_numpy(np.asarray(Image.open(f"{INP}/{self.bp[i]}").convert("RGB")
                             .resize((RES, RES), Image.BILINEAR)).astype(np.float32) / 255.0).permute(2, 0, 1)
        if self.train and np.random.rand() < 0.5:
            a = torch.flip(a, [2])
        return a, int(self.y[i])


class MSLA(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.d = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, c, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(c)) for d in (1, 2, 3)])
        self.spatial = nn.Conv2d(3 * c, 1, 1)
        self.mlp = nn.Sequential(nn.Linear(c, c // r), nn.GELU(), nn.Linear(c // r, c))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        cat = torch.cat([b(x) for b in self.d], 1)
        M = torch.sigmoid(self.spatial(cat)); s = torch.sigmoid(self.mlp(x.mean((2, 3))))
        return x + self.gamma * (x * M * s[:, :, None, None])


class PapayaFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = timm.create_model("mobilevit_s", pretrained=True, features_only=True, out_indices=(2, 3, 4))
        chs = self.bb.feature_info.channels()
        self.msla = nn.ModuleList([MSLA(c) for c in chs]); self.head = nn.Linear(sum(chs), K)

    def forward(self, x):
        x = (x - MEAN) / STD
        feats = self.bb(x)
        return F.softplus(self.head(torch.cat([m(z).mean((2, 3)) for z, m in zip(feats, self.msla)], 1)))


class MViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = timm.create_model("mobilevit_s", pretrained=True, num_classes=K)

    def forward(self, x):
        return self.net((x - MEAN) / STD)


def train(net, df, tr, n_k):
    torch.manual_seed(0); np.random.seed(0)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / 3 if ep < 3 else 0.5 * (1 + math.cos(math.pi * (ep - 3) / (EPOCHS - 3))))
    scaler = torch.cuda.amp.GradScaler()
    w = (K * (1 / n_k) / (1 / n_k).sum()).to(DEV)
    ce = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    dl = DataLoader(DS(df.iloc[tr], True), BATCH, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    is_pf = isinstance(net, PapayaFormer)
    for ep in range(EPOCHS):
        net.train(); t0 = time.time()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                o = net(x)
            if is_pf:
                e = o.float(); p = (e + 1) / (e + 1).sum(1, keepdim=True)
                y1 = F.one_hot(y, K).float(); S = (e + 1).sum(1, keepdim=True)
                loss = (y1 * (torch.digamma(S) - torch.digamma(e + 1))).sum(1).mean() + ce(torch.log(p.clamp_min(1e-6)), y)
            else:
                loss = ce(o.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        print(f"    ep{ep+1}/{EPOCHS} {time.time()-t0:.0f}s", flush=True)
    return net.eval()


def predict(net, df, idx):
    dl = DataLoader(DS(df.iloc[idx], False), 64, shuffle=False, num_workers=2)
    is_pf = isinstance(net, PapayaFormer)
    P = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, _ in dl:
            o = net(x.to(DEV)).float()
            P.append(((o + 1) / (o + 1).sum(1, keepdim=True)).cpu().numpy() if is_pf
                     else torch.softmax(o, 1).cpu().numpy())
    return np.concatenate(P)


def per_class(y, prob):
    pred = prob.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=range(K), zero_division=0)
    try:
        au = roc_auc_score(np.eye(K)[y], prob, average=None)
    except Exception:
        au = [float("nan")] * K
    return [{"class": CLASSES[i], "precision": float(pr[i]), "recall": float(rc[i]),
             "f1": float(f1[i]), "support": int(sup[i]), "auroc": float(au[i])} for i in range(K)]


def cm_png(y, pred, path, title):
    cm = confusion_matrix(y, pred, labels=range(K), normalize="true")
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels(SHORT, fontsize=7)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    print("device", DEV, flush=True)
    out = {}

    # A. per-class on raw S2 fold 0
    df = load(rawonly=True)
    y = df.unified_label.map(C2I).to_numpy(); g = df.group_id.to_numpy()
    n_k = torch.tensor([(y == i).sum() for i in range(K)], dtype=torch.float32)
    tr, te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups=g)))
    out["per_class"] = {}
    for name, mk in (("PapayaFormer", PapayaFormer), ("MobileViT-S", MViT)):
        print(f"\n== per-class {name} ==", flush=True)
        net = train(mk().to(DEV), df, tr, n_k)
        prob = predict(net, df, te)
        yte = y[te]
        out["per_class"][name] = per_class(yte, prob)
        if name == "PapayaFormer":
            cm_png(yte, prob.argmax(1), f"{OUT}/fig_cm.png", "PapayaFormer confusion matrix (S2)")
        del net
        if DEV == "cuda":
            torch.cuda.empty_cache()

    # B. cross-source S3
    dfa = load(["D1", "D2"], rawonly=True)
    dfb = load(["D3"], rawonly=True)
    ya = dfa.unified_label.map(C2I).to_numpy(); yb = dfb.unified_label.map(C2I).to_numpy()
    nka = torch.tensor([(ya == i).sum() for i in range(K)], dtype=torch.float32)
    print(f"\nS3: train D1+D2 N={len(dfa)}  test D3 N={len(dfb)}  "
          f"D3 class dist={np.bincount(yb, minlength=K).tolist()}", flush=True)
    out["s3"] = {"d3_class_counts": np.bincount(yb, minlength=K).tolist()}
    for name, mk in (("PapayaFormer", PapayaFormer), ("MobileViT-S", MViT)):
        print(f"\n== S3 {name} ==", flush=True)
        net = train(mk().to(DEV), dfa, np.arange(len(dfa)), nka)
        prob = predict(net, dfb, np.arange(len(dfb)))
        pred = prob.argmax(1)
        out["s3"][name] = {"acc": float(accuracy_score(yb, pred)),
                           "macro_f1": float(f1_score(yb, pred, average="macro")),
                           "per_class_f1": {CLASSES[i]: float(
                               f1_score((yb == i).astype(int), (pred == i).astype(int), zero_division=0))
                               for i in range(K)}}
        if name == "PapayaFormer":
            cm_png(yb, pred, f"{OUT}/fig_s3_cm.png", "S3 cross-source: train D1+D2 -> test D3")
        del net
        if DEV == "cuda":
            torch.cuda.empty_cache()
    json.dump(out, open(f"{OUT}/pcs3.json", "w"), indent=2)

    L = ["# Per-class results + cross-source S3 breakdown", "", "## A. Per-class on S2 (raw, fold 0)"]
    for m, rows in out["per_class"].items():
        L += [f"### {m}", "| class | precision | recall | F1 | support | AUROC |", "|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                     f"{r['support']} | {r['auroc']:.3f} |")
        L.append("")
    L += ["## B. Cross-source S3 (train D1+D2 -> test D3)",
          f"D3 test class counts: {out['s3']['d3_class_counts']}", ""]
    for m in ("PapayaFormer", "MobileViT-S"):
        d = out["s3"][m]
        L.append(f"**{m}**: acc {d['acc']*100:.1f}%, macro-F1 {d['macro_f1']*100:.1f}%; "
                 "per-class F1 = " + ", ".join(f"{c} {v:.2f}" for c, v in d["per_class_f1"].items()))
    open(f"{OUT}/pcs3.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
