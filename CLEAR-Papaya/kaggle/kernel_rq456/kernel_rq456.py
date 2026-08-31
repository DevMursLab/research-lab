"""RQ4 (corruption) + RQ5 (open-set) + RQ6 (low-data) + latency + tab:rq1 extra
metrics (MCC, kappa, AUROC), in one kernel. Raw corpus, S2 group-disjoint.

Models: resnet50, mobilevit_s, PapayaFormer. 2 seeds x fold 0 for RQ4/5/metrics;
PapayaFormer seed0 for RQ6 low-data curve.

Writes /kaggle/working/rq456.json (incremental) + rq456.md.
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
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             cohen_kappa_score, roc_auc_score, average_precision_score)
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
RESULTS = {}


def save():
    json.dump(RESULTS, open(f"{OUT}/rq456.json", "w"), indent=2, default=float)


# ---------------- data
def load_raw():
    pm = pd.read_csv(f"{INP}/path_map.csv")
    man = pd.read_csv(f"{INP}/master_manifest_rawonly.csv")
    grp = pd.read_csv(f"{INP}/groups_rawonly.csv")[["image_path", "group_id"]]
    df = man.merge(grp, on="image_path").merge(
        pm.rename(columns={"orig_path": "image_path"}), on="image_path")
    return df[df.split_hint.eq("closed_set") & df.unified_label.isin(CLASSES)].reset_index(drop=True)


def load_ood():
    pm = pd.read_csv(f"{INP}/path_map.csv")
    man = pd.read_csv(f"{INP}/master_manifest.csv")
    ood = man[man.unified_label == "UNMAPPED"].merge(
        pm.rename(columns={"orig_path": "image_path"}), on="image_path")
    return ood.bundle_path.tolist()


def rgb(bp):
    return np.asarray(Image.open(f"{INP}/{bp}").convert("RGB").resize((RES, RES), Image.BILINEAR))


class ArrDS(Dataset):
    """serves pre-corrupted uint8 arrays or bundle paths."""
    def __init__(self, items, labels=None, train=False, corrupt=None):
        self.items, self.labels, self.train, self.corrupt = items, labels, train, corrupt

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        a = self.items[i]
        if isinstance(a, str):
            a = rgb(a)
        a = a.astype(np.float32) / 255.0
        if self.corrupt is not None:
            a = self.corrupt(a)
        a = np.ascontiguousarray(a, dtype=np.float32)
        t = torch.from_numpy(a).permute(2, 0, 1)
        if self.train and np.random.rand() < 0.5:
            t = torch.flip(t, [2])
        t = (t - MEAN) / STD
        return (t, int(self.labels[i])) if self.labels is not None else t


# ---------------- MSLA / PapayaFormer (same as main kernel)
class MSLA(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.d = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, c, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(c))
            for d in (1, 2, 3)])
        self.spatial = nn.Conv2d(3 * c, 1, 1)
        self.mlp = nn.Sequential(nn.Linear(c, c // r), nn.GELU(), nn.Linear(c // r, c))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        cat = torch.cat([b(x) for b in self.d], 1)
        M = torch.sigmoid(self.spatial(cat))
        s = torch.sigmoid(self.mlp(x.mean((2, 3))))
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
        return F.softplus(self.head(torch.cat(
            [m(f).mean((2, 3)) for f, m in zip(feats, self.msla)], 1)))


def build(name):
    return PapayaFormer().to(DEV) if name == "PapayaFormer" else \
        timm.create_model(name, pretrained=True, num_classes=K).to(DEV)


def probs(net, name, loader):
    net.eval(); P = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for b in loader:
            x = b[0] if isinstance(b, (list, tuple)) else b
            o = net(x.to(DEV)).float()
            p = (o + 1) / (o + 1).sum(1, keepdim=True) if name == "PapayaFormer" \
                else torch.softmax(o, 1)
            P.append(p.cpu().numpy())
    return np.concatenate(P)


def edl_u(net, loader):
    net.eval(); U = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for b in loader:
            x = b[0] if isinstance(b, (list, tuple)) else b
            e = net(x.to(DEV)).float()
            U.append((K / (e + 1).sum(1)).cpu().numpy())
    return np.concatenate(U)


def train(name, df, tr, seed, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    net = build(name)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    warm = 3
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warm if ep < warm else
        0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, epochs - warm))))
    scaler = torch.cuda.amp.GradScaler()
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    items = df.iloc[tr].bundle_path.tolist()
    lab = df.iloc[tr].unified_label.map(C2I).to_numpy()
    dl = DataLoader(ArrDS(items, lab, train=True), BATCH, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)
    for ep in range(epochs):
        net.train(); t0 = time.time()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                o = net(x)
            if name == "PapayaFormer":
                e = o.float(); p = (e + 1) / (e + 1).sum(1, keepdim=True)
                y1 = F.one_hot(y, K).float()
                S = (e + 1).sum(1, keepdim=True)
                loss = (y1 * (torch.digamma(S) - torch.digamma(e + 1))).sum(1).mean() \
                    + F.nll_loss(torch.log(p.clamp_min(1e-6)), y)
            else:
                loss = ce(o.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        print(f"    [{name} s{seed}] ep{ep+1}/{epochs} {time.time()-t0:.0f}s", flush=True)
    return net


# ---------------- corruptions (numpy, img in [0,1] HxWx3)
def c_gauss_noise(a, s): return np.clip(a + np.random.randn(*a.shape) * s, 0, 1)
def c_shot(a, s): return np.clip(np.random.poisson(a * (1 / s)) * s, 0, 1)
def c_blur(a, s):
    from scipy.ndimage import gaussian_filter
    return np.clip(gaussian_filter(a, sigma=(s, s, 0)), 0, 1)
def c_bright(a, s): return np.clip(a + s, 0, 1)
def c_contrast(a, s):
    m = a.mean(); return np.clip((a - m) * s + m, 0, 1)
def c_jpeg(a, q):
    import io
    im = Image.fromarray((a * 255).astype(np.uint8)); buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(q)); return np.asarray(Image.open(buf)).astype(np.float32) / 255
def c_pixelate(a, s):
    h = max(4, int(RES * s)); im = Image.fromarray((a * 255).astype(np.uint8))
    im = im.resize((h, h), Image.BILINEAR).resize((RES, RES), Image.NEAREST)
    return np.asarray(im).astype(np.float32) / 255

CORRUPT = {
    "gauss_noise": (c_gauss_noise, [0.05, 0.10, 0.18]),
    "shot_noise": (c_shot, [0.02, 0.05, 0.10]),
    "blur": (c_blur, [1.0, 2.0, 3.0]),
    "brightness": (c_bright, [0.15, 0.30, 0.45]),
    "contrast": (c_contrast, [0.7, 0.5, 0.35]),
    "jpeg": (c_jpeg, [40, 20, 10]),
    "pixelate": (c_pixelate, [0.5, 0.35, 0.22]),
}


def main():
    print("device", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", flush=True)
    df = load_raw()
    y = df.unified_label.map(C2I).to_numpy(); g = df.group_id.to_numpy()
    folds0 = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups=g))
    tr0, te0 = folds0[0]
    te_items = df.iloc[te0].bundle_path.tolist()
    te_lab = y[te0]
    ood_items = load_ood()
    print(f"raw N={len(df)}  test0={len(te0)}  OOD={len(ood_items)}", flush=True)

    MODELS = ["resnet50", "mobilevit_s", "PapayaFormer"]

    # ---- RQ1 extra metrics + RQ4 + RQ5, 2 seeds ----
    RESULTS["rq1_metrics"] = {}
    RESULTS["rq4"] = {}
    RESULTS["rq5"] = {}
    clean_te = DataLoader(ArrDS(te_items), 64, num_workers=2)
    ood_dl = DataLoader(ArrDS(ood_items), 64, num_workers=2)
    for name in MODELS:
        accs, mccs, kaps, aurs = [], [], [], []
        for seed in (0, 1):
            net = train(name, df, tr0, seed)
            p = probs(net, name, clean_te)
            pred = p.argmax(1)
            accs.append(accuracy_score(te_lab, pred))
            mccs.append(matthews_corrcoef(te_lab, pred))
            kaps.append(cohen_kappa_score(te_lab, pred))
            try:
                aurs.append(roc_auc_score(np.eye(K)[te_lab], p, multi_class="ovr", average="macro"))
            except Exception:
                aurs.append(float("nan"))
            if seed == 0:
                # RQ4 corruption on this seed
                base = accuracy_score(te_lab, pred)
                cres = {}
                for cn, (fn, sevs) in CORRUPT.items():
                    ca = []
                    for sv in sevs:
                        dl = DataLoader(ArrDS(te_items, corrupt=lambda a, f=fn, s=sv: f(a, s)),
                                        64, num_workers=2)
                        ca.append(float(accuracy_score(te_lab, probs(net, name, dl).argmax(1))))
                    cres[cn] = ca
                mca = float(np.mean([np.mean(v) for v in cres.values()]))
                RESULTS["rq4"][name] = {"clean": float(base), "per_corruption": cres,
                                        "mean_corruption_acc": mca,
                                        "relative_robustness": mca / base}
                save()
                # RQ5 open-set: score = evidential u (PapayaFormer) else 1-maxprob
                if name == "PapayaFormer":
                    s_in = edl_u(net, clean_te); s_out = edl_u(net, ood_dl)
                else:
                    s_in = 1 - probs(net, name, clean_te).max(1)
                    s_out = 1 - probs(net, name, ood_dl).max(1)
                yb = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]
                sc = np.r_[s_in, s_out]
                auroc = float(roc_auc_score(yb, sc))
                aupr = float(average_precision_score(yb, sc))
                thr = np.quantile(s_in, 0.95)
                fpr95 = float((s_out <= thr).mean())  # OOD wrongly accepted
                RESULTS["rq5"][name] = {"auroc": auroc, "aupr": aupr, "fpr_at_95tpr": fpr95}
                save()
        RESULTS["rq1_metrics"][name] = {
            "acc": [float(np.mean(accs)), float(np.std(accs))],
            "mcc": [float(np.mean(mccs)), float(np.std(mccs))],
            "kappa": [float(np.mean(kaps)), float(np.std(kaps))],
            "auroc_ovr": [float(np.nanmean(aurs)), float(np.nanstd(aurs))]}
        save()

    # ---- latency: PapayaFormer FP32 ----
    net = PapayaFormer().to(DEV).eval()
    x1 = torch.randn(1, 3, RES, RES, device=DEV)
    with torch.no_grad():
        for _ in range(10):
            net(x1)
        if DEV == "cuda":
            torch.cuda.synchronize()
        t = []
        for _ in range(100):
            s = time.time(); net(x1)
            if DEV == "cuda":
                torch.cuda.synchronize()
            t.append((time.time() - s) * 1000)
    net_cpu = net.to("cpu"); xc = torch.randn(1, 3, RES, RES)
    with torch.no_grad():
        for _ in range(3):
            net_cpu(xc)
        tc = []
        for _ in range(20):
            s = time.time(); net_cpu(xc); tc.append((time.time() - s) * 1000)
    RESULTS["latency_ms"] = {"gpu_b1_median": float(np.median(t)),
                             "cpu_b1_median": float(np.median(tc)),
                             "params_M": sum(p.numel() for p in net.parameters()) / 1e6}
    save()
    del net, net_cpu
    if DEV == "cuda":
        torch.cuda.empty_cache()

    # ---- RQ6 low-data curve: PapayaFormer, seed 0 ----
    RESULTS["rq6_lowdata"] = {}
    rng = np.random.default_rng(0)
    tr_groups = np.array(sorted(set(g[tr0])))
    for frac in [0.05, 0.10, 0.25, 0.50, 1.0]:
        keepg = set(rng.choice(tr_groups, max(1, int(len(tr_groups) * frac)), replace=False))
        sub = np.array([i for i in tr0 if g[i] in keepg])
        net = train("PapayaFormer", df, sub, 0, epochs=10)
        acc = accuracy_score(te_lab, probs(net, "PapayaFormer", clean_te).argmax(1))
        RESULTS["rq6_lowdata"][f"{frac:.2f}"] = {"n_train": int(len(sub)), "acc": float(acc)}
        print(f"  RQ6 frac={frac} n={len(sub)} acc={acc:.3f}", flush=True)
        save()

    # markdown
    L = ["# RQ4 / RQ5 / RQ6 / latency  (raw corpus, S2 fold 0)", ""]
    L.append("## tab:rq1 extra metrics (2 seeds)")
    L.append("| model | acc | MCC | kappa | AUROC(ovr) |")
    L.append("|---|---|---|---|---|")
    for m, d in RESULTS["rq1_metrics"].items():
        L.append(f"| {m} | {d['acc'][0]*100:.1f} | {d['mcc'][0]:.3f} | "
                 f"{d['kappa'][0]:.3f} | {d['auroc_ovr'][0]:.3f} |")
    L += ["", "## RQ4 corruption robustness"]
    L.append("| model | clean | mean corruption acc | relative robustness |")
    L.append("|---|---|---|---|")
    for m, d in RESULTS["rq4"].items():
        L.append(f"| {m} | {d['clean']*100:.1f} | {d['mean_corruption_acc']*100:.1f} | "
                 f"{d['relative_robustness']:.3f} |")
    L += ["", "## RQ5 open-set (Phytophthora as unseen class)"]
    L.append("| model | AUROC | AUPR | FPR@95TPR |")
    L.append("|---|---|---|---|")
    for m, d in RESULTS["rq5"].items():
        L.append(f"| {m} | {d['auroc']:.3f} | {d['aupr']:.3f} | {d['fpr_at_95tpr']:.3f} |")
    L += ["", "## Latency (PapayaFormer FP32, batch 1)",
          f"- GPU median: {RESULTS['latency_ms']['gpu_b1_median']:.1f} ms",
          f"- CPU median: {RESULTS['latency_ms']['cpu_b1_median']:.1f} ms",
          f"- Params: {RESULTS['latency_ms']['params_M']:.1f} M", "",
          "## RQ6 low-data (PapayaFormer)"]
    L.append("| train frac | n_train | acc |")
    L.append("|---|---|---|")
    for k, d in RESULTS["rq6_lowdata"].items():
        L.append(f"| {k} | {d['n_train']} | {d['acc']*100:.1f} |")
    open(f"{OUT}/rq456.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
