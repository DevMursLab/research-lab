"""Ablation: does MSLA help? does the evidential head help? is gamma-init-0 needed?

Variants (raw corpus, S2 group-disjoint, seed 0, 3 folds, 12 ep):
  full        : MSLA @ stages 2/3/4 + evidential head
  no_msla     : backbone + evidential head
  no_evid     : MSLA @ 2/3/4 + softmax head
  plain       : backbone + softmax head        (== MobileViT-S baseline, same folds)
  gamma_rand  : full, but MSLA gamma initialised ~N(0,0.1) instead of 0
  msla_s4only : MSLA at stage 4 only + evidential head

Per variant: acc, macro-F1, ECE, Brier (mean +- sd over 3 folds), params.
Writes /kaggle/working/ablation.json (incremental) + ablation.md.
"""
import json, time, os, sys, subprocess, glob, math

import torch
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7 \
        and os.environ.get("_ARCH_FIX") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1", "torchvision==0.20.1"], check=False)
    os.environ["_ARCH_FIX"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np, pandas as pd, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
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
K = len(CLASSES)
C2I = {c: i for i, c in enumerate(CLASSES)}
RES, EPOCHS, BATCH = 224, 12, 64
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_raw():
    pm = pd.read_csv(f"{INP}/path_map.csv")
    man = pd.read_csv(f"{INP}/master_manifest_rawonly.csv")
    grp = pd.read_csv(f"{INP}/groups_rawonly.csv")[["image_path", "group_id"]]
    df = man.merge(grp, on="image_path").merge(
        pm.rename(columns={"orig_path": "image_path"}), on="image_path")
    return df[df.split_hint.eq("closed_set") & df.unified_label.isin(CLASSES)].reset_index(drop=True)


class DS(Dataset):
    def __init__(self, df, train):
        self.bp = df.bundle_path.tolist()
        self.y = df.unified_label.map(C2I).to_numpy()
        self.train = train

    def __len__(self): return len(self.bp)

    def __getitem__(self, i):
        a = np.asarray(Image.open(f"{INP}/{self.bp[i]}").convert("RGB")
                       .resize((RES, RES), Image.BILINEAR)).astype(np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1)
        if self.train and np.random.rand() < 0.5:
            t = torch.flip(t, [2])
        return (t - MEAN) / STD, int(self.y[i])


class MSLA(nn.Module):
    def __init__(self, c, r=8, gamma_std=0.0):
        super().__init__()
        self.d = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, c, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(c))
            for d in (1, 2, 3)])
        self.spatial = nn.Conv2d(3 * c, 1, 1)
        self.mlp = nn.Sequential(nn.Linear(c, c // r), nn.GELU(), nn.Linear(c // r, c))
        g = torch.zeros(1) if gamma_std == 0 else torch.randn(1) * gamma_std
        self.gamma = nn.Parameter(g)

    def forward(self, x):
        cat = torch.cat([b(x) for b in self.d], 1)
        M = torch.sigmoid(self.spatial(cat))
        s = torch.sigmoid(self.mlp(x.mean((2, 3))))
        return x + self.gamma * (x * M * s[:, :, None, None])


class Net(nn.Module):
    def __init__(self, use_msla="all", evidential=True, gamma_std=0.0):
        super().__init__()
        self.bb = timm.create_model("mobilevit_s", pretrained=True,
                                    features_only=True, out_indices=(2, 3, 4))
        chs = self.bb.feature_info.channels()
        self.evidential = evidential
        if use_msla == "all":
            self.msla = nn.ModuleList([MSLA(c, gamma_std=gamma_std) for c in chs])
        elif use_msla == "s4":
            self.msla = nn.ModuleList([nn.Identity(), nn.Identity(),
                                       MSLA(chs[-1], gamma_std=gamma_std)])
        else:
            self.msla = nn.ModuleList([nn.Identity() for _ in chs])
        self.head = nn.Linear(sum(chs), K)

    def forward(self, x):
        feats = self.bb(x)
        z = torch.cat([m(f).mean((2, 3)) for f, m in zip(feats, self.msla)], 1)
        o = self.head(z)
        return F.softplus(o) if self.evidential else o


VARIANTS = {
    "full":        dict(use_msla="all", evidential=True),
    "no_msla":     dict(use_msla="none", evidential=True),
    "no_evid":     dict(use_msla="all", evidential=False),
    "plain":       dict(use_msla="none", evidential=False),
    "gamma_rand":  dict(use_msla="all", evidential=True, gamma_std=0.1),
    "msla_s4only": dict(use_msla="s4", evidential=True),
}


def ece(prob, y, nb=15):
    conf, pred = prob.max(1), prob.argmax(1)
    acc = (pred == y).astype(float)
    e, b = 0.0, np.linspace(0, 1, nb + 1)
    for i in range(nb):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(prob, y):
    return float(((prob - np.eye(K)[y]) ** 2).sum(1).mean())


def run(cfg, df, tr, te):
    torch.manual_seed(0); np.random.seed(0)
    net = Net(**cfg).to(DEV)
    npar = sum(p.numel() for p in net.parameters()) / 1e6
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    warm = 3
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warm if ep < warm else
        0.5 * (1 + math.cos(math.pi * (ep - warm) / (EPOCHS - warm))))
    scaler = torch.cuda.amp.GradScaler()
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    dl = DataLoader(DS(df.iloc[tr], True), BATCH, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)
    el = DataLoader(DS(df.iloc[te], False), BATCH, shuffle=False, num_workers=2)
    yte = df.iloc[te].unified_label.map(C2I).to_numpy()
    best = None
    for ep in range(EPOCHS):
        net.train(); t0 = time.time()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                o = net(x)
            if cfg["evidential"]:
                e = o.float(); p = (e + 1) / (e + 1).sum(1, keepdim=True)
                y1 = F.one_hot(y, K).float()
                S = (e + 1).sum(1, keepdim=True)
                loss = (y1 * (torch.digamma(S) - torch.digamma(e + 1))).sum(1).mean() \
                    + F.nll_loss(torch.log(p.clamp_min(1e-6)), y)
            else:
                loss = ce(o.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        net.eval(); P = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for x, _ in el:
                o = net(x.to(DEV)).float()
                p = (o + 1) / (o + 1).sum(1, keepdim=True) if cfg["evidential"] \
                    else torch.softmax(o, 1)
                P.append(p.cpu().numpy())
        prob = np.concatenate(P)
        acc = accuracy_score(yte, prob.argmax(1))
        f1 = f1_score(yte, prob.argmax(1), average="macro")
        if best is None or f1 > best[1]:
            best = (acc, f1, ece(prob, yte), brier(prob, yte))
        print(f"    ep{ep+1}/{EPOCHS} acc={acc:.3f} f1={f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return best, npar


def main():
    print("device", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", flush=True)
    df = load_raw()
    y = df.unified_label.map(C2I).to_numpy(); g = df.group_id.to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups=g))[:3]
    out = {}
    for vname, cfg in VARIANTS.items():
        rows, npar = [], None
        for k, (tr, te) in enumerate(folds):
            print(f"\n== {vname} fold {k} ==", flush=True)
            (acc, f1, ec, br), npar = run(cfg, df, tr, te)
            rows.append((acc, f1, ec, br))
        a = np.array(rows)
        out[vname] = {"acc": [float(a[:, 0].mean()), float(a[:, 0].std())],
                      "mF1": [float(a[:, 1].mean()), float(a[:, 1].std())],
                      "ece": [float(a[:, 2].mean()), float(a[:, 2].std())],
                      "brier": [float(a[:, 3].mean()), float(a[:, 3].std())],
                      "params_M": float(npar)}
        json.dump(out, open(f"{OUT}/ablation.json", "w"), indent=2)
        print("  =>", vname, json.dumps(out[vname]), flush=True)

    L = ["# PapayaFormer ablation (raw, S2, 3 folds, seed 0)", "",
         "| variant | acc | macro-F1 | ECE | Brier | params (M) |",
         "|---|---|---|---|---|---|"]
    for v, d in out.items():
        L.append(f"| {v} | {d['acc'][0]*100:.1f}±{d['acc'][1]*100:.1f} | "
                 f"{d['mF1'][0]*100:.1f} | {d['ece'][0]:.3f} | {d['brier'][0]:.3f} | "
                 f"{d['params_M']:.1f} |")
    open(f"{OUT}/ablation.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
