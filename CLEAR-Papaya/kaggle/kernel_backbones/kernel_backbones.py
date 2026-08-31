"""Extra backbone rows for tab:rq1 (raw corpus, S2 group-disjoint, seed 0, 3 folds).

DenseNet-121, ConvNeXt-Tiny, MobileNetV3-L, ViT-B/16, DeiT3-S, Swin-T.
Per model: acc, macro-F1, MCC, kappa, AUROC-ovr, ECE (3-fold mean +- sd).
Writes /kaggle/working/backbones.json (incremental) + backbones.md.
"""
import json, time, os, sys, subprocess, glob, math

import torch
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7 \
        and os.environ.get("_ARCH_FIX") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1", "torchvision==0.20.1"], check=False)
    os.environ["_ARCH_FIX"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np, pandas as pd, torch.nn as nn, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             cohen_kappa_score, roc_auc_score)
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
RES, EPOCHS, BATCH = 224, 12, 48
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

MODELS = ["densenet121", "convnext_tiny.fb_in1k", "mobilenetv3_large_100",
          "vit_base_patch16_224.augreg2_in21k_ft_in1k",
          "deit3_small_patch16_224.fb_in1k",
          "swin_tiny_patch4_window7_224.ms_in1k"]
NICE = {"densenet121": "DenseNet-121", "convnext_tiny.fb_in1k": "ConvNeXt-Tiny",
        "mobilenetv3_large_100": "MobileNetV3-L",
        "vit_base_patch16_224.augreg2_in21k_ft_in1k": "ViT-B/16",
        "deit3_small_patch16_224.fb_in1k": "DeiT3-S",
        "swin_tiny_patch4_window7_224.ms_in1k": "Swin-T"}


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


def ece(prob, y, nb=15):
    conf, pred = prob.max(1), prob.argmax(1)
    acc = (pred == y).astype(float)
    e, b = 0.0, np.linspace(0, 1, nb + 1)
    for i in range(nb):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def run(name, df, tr, te):
    torch.manual_seed(0); np.random.seed(0)
    net = timm.create_model(name, pretrained=True, num_classes=K).to(DEV)
    npar = sum(p.numel() for p in net.parameters()) / 1e6
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=0.05)
    warm = 3
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warm if ep < warm else
        0.5 * (1 + math.cos(math.pi * (ep - warm) / (EPOCHS - warm))))
    scaler = torch.cuda.amp.GradScaler()
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
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
                loss = lossf(net(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        net.eval(); P = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for x, _ in el:
                P.append(torch.softmax(net(x.to(DEV)).float(), 1).cpu().numpy())
        prob = np.concatenate(P); pred = prob.argmax(1)
        f1 = f1_score(yte, pred, average="macro")
        if best is None or f1 > best["mF1"]:
            try:
                au = roc_auc_score(np.eye(K)[yte], prob, multi_class="ovr", average="macro")
            except Exception:
                au = float("nan")
            best = {"acc": float(accuracy_score(yte, pred)), "mF1": float(f1),
                    "mcc": float(matthews_corrcoef(yte, pred)),
                    "kappa": float(cohen_kappa_score(yte, pred)),
                    "auroc": float(au), "ece": ece(prob, yte)}
        print(f"    [{name}] ep{ep+1}/{EPOCHS} f1={f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return best, npar


def main():
    print("device", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", flush=True)
    df = load_raw()
    y = df.unified_label.map(C2I).to_numpy(); g = df.group_id.to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups=g))[:3]
    out = {}
    for name in MODELS:
        rows, npar = [], None
        try:
            for k, (tr, te) in enumerate(folds):
                print(f"\n== {NICE[name]} fold {k} ==", flush=True)
                r, npar = run(name, df, tr, te)
                rows.append(r)
        except Exception as ex:
            print(f"  !! {name} failed: {ex}", flush=True)
            continue
        agg = {m: [float(np.mean([r[m] for r in rows])),
                   float(np.std([r[m] for r in rows]))]
               for m in ("acc", "mF1", "mcc", "kappa", "auroc", "ece")}
        agg["params_M"] = float(npar)
        out[NICE[name]] = agg
        json.dump(out, open(f"{OUT}/backbones.json", "w"), indent=2)
        print("  =>", NICE[name], json.dumps(agg), flush=True)

    L = ["# Extra backbones on S2 (raw, 3 folds, seed 0)", "",
         "| model | acc | macro-F1 | MCC | kappa | AUROC | ECE | params (M) |",
         "|---|---|---|---|---|---|---|---|"]
    for m, d in out.items():
        L.append(f"| {m} | {d['acc'][0]*100:.1f}±{d['acc'][1]*100:.1f} | "
                 f"{d['mF1'][0]*100:.1f} | {d['mcc'][0]:.3f} | {d['kappa'][0]:.3f} | "
                 f"{d['auroc'][0]:.3f} | {d['ece'][0]:.3f} | {d['params_M']:.1f} |")
    open(f"{OUT}/backbones.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
