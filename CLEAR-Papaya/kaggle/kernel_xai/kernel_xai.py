"""RQ8 --- explanation faithfulness (mask-free part).

Deletion AUC (lower better), Insertion AUC (higher better) and attribution
stability (Spearman rho under small Gaussian input noise), for:
  * Grad-CAM++ on ResNet-50
  * Grad-CAM++ on PapayaFormer (last backbone stage)
  * the intrinsic MSLA spatial mask on PapayaFormer

Pointing-game / attribution-mass-in-leaf need expert masks and are deferred.

Also renders a qualitative grid (input / Grad-CAM++ / MSLA) with 2 successes and
2 failures -> figs/fig_xai.png .

Writes /kaggle/working/xai.json + xai.md .
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
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import spearmanr


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
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEV)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEV)


def load_raw():
    pm = pd.read_csv(f"{INP}/path_map.csv")
    man = pd.read_csv(f"{INP}/master_manifest_rawonly.csv")
    grp = pd.read_csv(f"{INP}/groups_rawonly.csv")[["image_path", "group_id"]]
    df = man.merge(grp, on="image_path").merge(
        pm.rename(columns={"orig_path": "image_path"}), on="image_path")
    return df[df.split_hint.eq("closed_set") & df.unified_label.isin(CLASSES)].reset_index(drop=True)


def rgb(bp):
    return np.asarray(Image.open(f"{INP}/{bp}").convert("RGB").resize((RES, RES), Image.BILINEAR))


class DS(Dataset):
    def __init__(self, df, train):
        self.bp = df.bundle_path.tolist(); self.y = df.unified_label.map(C2I).to_numpy(); self.train = train

    def __len__(self): return len(self.bp)

    def __getitem__(self, i):
        a = torch.from_numpy(rgb(self.bp[i]).astype(np.float32) / 255.0).permute(2, 0, 1)
        if self.train and np.random.rand() < 0.5:
            a = torch.flip(a, [2])
        return a, int(self.y[i])   # NOTE: un-normalised; normalise in the model wrapper


class MSLA(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.d = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, c, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(c)) for d in (1, 2, 3)])
        self.spatial = nn.Conv2d(3 * c, 1, 1)
        self.mlp = nn.Sequential(nn.Linear(c, c // r), nn.GELU(), nn.Linear(c // r, c))
        self.gamma = nn.Parameter(torch.zeros(1)); self.last_M = None

    def forward(self, x):
        cat = torch.cat([b(x) for b in self.d], 1)
        M = torch.sigmoid(self.spatial(cat)); s = torch.sigmoid(self.mlp(x.mean((2, 3))))
        self.last_M = M
        return x + self.gamma * (x * M * s[:, :, None, None])


class PapayaFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = timm.create_model("mobilevit_s", pretrained=True, features_only=True, out_indices=(2, 3, 4))
        chs = self.bb.feature_info.channels()
        self.msla = nn.ModuleList([MSLA(c) for c in chs]); self.head = nn.Linear(sum(chs), K)
        self.last_feat = None

    def forward(self, x):
        x = (x - MEAN) / STD
        feats = self.bb(x)
        self.last_feat = feats[-1]
        pooled = []
        for f_, m in zip(feats, self.msla):
            pooled.append(m(f_).mean((2, 3)))
        return F.softplus(self.head(torch.cat(pooled, 1)))


class RN50(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = timm.create_model("resnet50", pretrained=True, num_classes=K)
        self.last_feat = None
        self.net.layer4.register_forward_hook(lambda m, i, o: setattr(self, "last_feat", o))

    def forward(self, x):
        return self.net((x - MEAN) / STD)


def train(net, df, tr):
    torch.manual_seed(0); np.random.seed(0)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / 3 if ep < 3 else 0.5 * (1 + math.cos(math.pi * (ep - 3) / (EPOCHS - 3))))
    scaler = torch.cuda.amp.GradScaler()
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    dl = DataLoader(DS(df.iloc[tr], True), BATCH, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    for ep in range(EPOCHS):
        net.train(); t0 = time.time()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                o = net(x)
                loss = ce(o.float() if not isinstance(net, PapayaFormer) else torch.log(
                    (o.float() + 1) / (o.float() + 1).sum(1, keepdim=True)).clamp_min(-20), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        print(f"    ep{ep+1}/{EPOCHS} {time.time()-t0:.0f}s", flush=True)
    return net.eval()


def gradcampp(net, x, cls):
    """Grad-CAM++ on net.last_feat. x: (1,3,H,W) un-normalised, requires_grad not needed."""
    net.zero_grad()
    x = x.clone().to(DEV)
    feat_store = {}
    h1 = None
    out = net(x)
    A = net.last_feat                      # (1,C,h,w)
    A.retain_grad()
    score = out[0, cls]
    score.backward(retain_graph=False)
    grads = A.grad                         # (1,C,h,w)
    g2 = grads ** 2
    g3 = g2 * grads
    denom = 2 * g2 + (A * g3).sum((2, 3), keepdim=True)
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    alpha = g2 / denom
    w = (alpha * F.relu(grads)).sum((2, 3), keepdim=True)   # (1,C,1,1)
    cam = F.relu((w * A).sum(1, keepdim=True))              # (1,1,h,w)
    cam = F.interpolate(cam, size=(RES, RES), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def msla_map(net, x):
    with torch.no_grad():
        net(x.to(DEV))
    M = net.msla[-1].last_M                # (1,1,h,w)
    M = F.interpolate(M, size=(RES, RES), mode="bilinear", align_corners=False)[0, 0]
    M = (M - M.min()) / (M.max() - M.min() + 1e-8)
    return M.detach().cpu().numpy()


def del_ins_auc(net, x, cls, sal, steps=25, mode="del"):
    order = np.argsort(sal.ravel())[::-1]        # most salient first
    n = len(order); base = x.clone()
    xs = base.clone().to(DEV)
    if mode == "ins":
        blur = F.avg_pool2d(base.to(DEV), 15, 1, 7)
        xs = blur.clone()
    ys = []
    with torch.no_grad():
        for s in range(steps + 1):
            k = int(n * s / steps)
            idx = order[:k]
            r, c = np.unravel_index(idx, (RES, RES))
            cur = xs.clone()
            if mode == "del":
                cur[0, :, r, c] = 0.0
            else:
                cur[0, :, r, c] = base.to(DEV)[0, :, r, c]
            o = net(cur)
            p = (o.float() + 1) / (o.float() + 1).sum(1, keepdim=True) if isinstance(net, PapayaFormer) \
                else torch.softmax(o.float(), 1)
            ys.append(float(p[0, cls]))
    return float(np.trapz(ys, dx=1.0 / steps))


def main():
    print("device", DEV, flush=True)
    df = load_raw()
    y = df.unified_label.map(C2I).to_numpy(); g = df.group_id.to_numpy()
    tr, te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups=g)))
    pf = train(PapayaFormer().to(DEV), df, tr)
    rn = train(RN50().to(DEV), df, tr)

    rng = np.random.default_rng(0)
    samp = rng.choice(te, min(120, len(te)), replace=False)
    rec = {"gradcampp_rn50": [], "gradcampp_pf": [], "msla_pf": [], "stability": {}}
    stab_gc, stab_ms = [], []
    for i in samp:
        x = DS(df.iloc[[i]], False)[0][0].unsqueeze(0)
        cls = int(y[i])
        for tag, net, sal in (("gradcampp_rn50", rn, None), ("gradcampp_pf", pf, None), ("msla_pf", pf, None)):
            if tag == "msla_pf":
                s = msla_map(pf, x)
            else:
                s = gradcampp(net, x, cls)
            d = del_ins_auc(net, x, cls, s, mode="del")
            ins = del_ins_auc(net, x, cls, s, mode="ins")
            rec[tag].append((d, ins))
        # stability: gradcam++ (pf) and msla under noise
        xn = (x + 0.03 * torch.randn_like(x)).clamp(0, 1)
        rho_gc = spearmanr(gradcampp(pf, x, cls).ravel(), gradcampp(pf, xn, cls).ravel()).correlation
        rho_ms = spearmanr(msla_map(pf, x).ravel(), msla_map(pf, xn).ravel()).correlation
        stab_gc.append(rho_gc); stab_ms.append(rho_ms)
    out = {}
    for k_, v in (("gradcampp_rn50", rec["gradcampp_rn50"]), ("gradcampp_pf", rec["gradcampp_pf"]),
                  ("msla_pf", rec["msla_pf"])):
        a = np.array(v)
        out[k_] = {"del_auc": [float(a[:, 0].mean()), float(a[:, 0].std())],
                   "ins_auc": [float(a[:, 1].mean()), float(a[:, 1].std())]}
    out["stability_spearman"] = {"gradcampp_pf": float(np.nanmean(stab_gc)),
                                 "msla_pf": float(np.nanmean(stab_ms))}
    json.dump(out, open(f"{OUT}/xai.json", "w"), indent=2)

    # qualitative grid: 2 correct + 2 wrong
    with torch.no_grad():
        preds = []
        for i in te[:400]:
            x = DS(df.iloc[[i]], False)[0][0].unsqueeze(0).to(DEV)
            o = pf(x); preds.append((i, int(((o + 1)).argmax(1)), int(y[i])))
    ok = [i for i, p, t in preds if p == t][:2]
    bad = [i for i, p, t in preds if p != t][:2]
    rows = ok + bad
    fig, ax = plt.subplots(len(rows), 3, figsize=(6, 2.0 * len(rows)))
    for r, i in enumerate(rows):
        x = DS(df.iloc[[i]], False)[0][0].unsqueeze(0)
        cls = int(y[i])
        im = rgb(df.iloc[i].bundle_path)
        gc = gradcampp(pf, x, cls); ms = msla_map(pf, x)
        for c, (img, ttl) in enumerate([(im, "input"), (gc, "Grad-CAM++"), (ms, "MSLA mask")]):
            ax[r, c].imshow(im if c == 0 else im, alpha=1.0)
            if c > 0:
                ax[r, c].imshow(img, cmap="jet", alpha=0.5)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(ttl, fontsize=9)
        ax[r, 0].set_ylabel(("correct: " if i in ok else "wrong: ") + CLASSES[cls], fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_xai.png", dpi=180); plt.close(fig)

    md = ["# RQ8 faithfulness (mask-free; pointing-game deferred)", "",
          "| method | Deletion AUC (lower better) | Insertion AUC (higher better) |",
          "|---|---|---|",
          f"| Grad-CAM++ (ResNet-50) | {out['gradcampp_rn50']['del_auc'][0]:.3f} | {out['gradcampp_rn50']['ins_auc'][0]:.3f} |",
          f"| Grad-CAM++ (PapayaFormer) | {out['gradcampp_pf']['del_auc'][0]:.3f} | {out['gradcampp_pf']['ins_auc'][0]:.3f} |",
          f"| MSLA mask (intrinsic) | {out['msla_pf']['del_auc'][0]:.3f} | {out['msla_pf']['ins_auc'][0]:.3f} |",
          "",
          f"Attribution stability (Spearman rho under 3% input noise): "
          f"Grad-CAM++ {out['stability_spearman']['gradcampp_pf']:.2f}, "
          f"MSLA {out['stability_spearman']['msla_pf']:.2f}."]
    open(f"{OUT}/xai.md", "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
