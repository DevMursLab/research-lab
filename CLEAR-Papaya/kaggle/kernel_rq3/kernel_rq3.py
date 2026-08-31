"""RQ3 --- Clever-Hans background probe.

Train a model on S2 (group-disjoint) then evaluate the SAME weights on three
versions of every test image:
  V_full : unchanged
  V_leaf : background replaced by neutral grey (leaf kept)
  V_bg   : leaf replaced by neutral grey (background kept)

Report  Acc(V_full), Acc(V_leaf), Acc(V_bg),
        Delta_bg   = Acc(V_bg)  - 1/K            (background alone carries class info)
        Delta_leaf = Acc(V_full) - Acc(V_leaf)   (reliance on context)

Segmentation: rembg / U^2-Net foreground mask (fallback: OpenCV GrabCut with a
central-box prior). Mean foreground-area fraction is logged as a sanity check.

Models: mobilevit_s, resnet50.  Condition: raw_only.  One S2 fold (seed 0, k 0).
Writes /kaggle/working/rq3_results.json + rq3_results.md.
"""
import json, time, os, sys, subprocess, glob

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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

HAVE_REMBG = False   # avoid pip clobbering numpy/pandas on Kaggle
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


def resolve_inp():
    for i in range(180):
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
RES = 224
EPOCHS = 12
BATCH = 64
GREY = 0.5
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_raw():
    pm = pd.read_csv(f"{INP}/path_map.csv")
    man = pd.read_csv(f"{INP}/master_manifest_rawonly.csv")
    grp = pd.read_csv(f"{INP}/groups_rawonly.csv")[["image_path", "group_id"]]
    df = man.merge(grp, on="image_path").merge(
        pm.rename(columns={"orig_path": "image_path"}), on="image_path")
    return df[df.split_hint.eq("closed_set") & df.unified_label.isin(CLASSES)].reset_index(drop=True)


def load_rgb(bp):
    return np.asarray(Image.open(f"{INP}/{bp}").convert("RGB").resize((RES, RES), Image.BILINEAR))


def leaf_mask(rgb):
    """return HxW bool mask, True = leaf/foreground."""
    if HAVE_REMBG:
        try:
            out = rembg_remove(Image.fromarray(rgb), session=_RS)
            a = np.asarray(out)
            if a.shape[-1] == 4:
                return a[..., 3] > 128
        except Exception:
            pass
    if HAVE_CV2:
        m = np.zeros(rgb.shape[:2], np.uint8)
        rect = (12, 12, RES - 24, RES - 24)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(rgb, m, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            return (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
        except Exception:
            pass
    # last resort: centre ellipse
    yy, xx = np.ogrid[:RES, :RES]
    return ((yy - RES / 2) ** 2 / (RES * 0.42) ** 2 + (xx - RES / 2) ** 2 / (RES * 0.42) ** 2) <= 1


class TrainDS(Dataset):
    def __init__(self, df, train):
        self.bp = df.bundle_path.tolist()
        self.y = df.unified_label.map(C2I).to_numpy()
        self.train = train

    def __len__(self): return len(self.bp)

    def __getitem__(self, i):
        a = torch.from_numpy(load_rgb(self.bp[i]).copy()).float().permute(2, 0, 1) / 255.0
        if self.train and np.random.rand() < 0.5:
            a = torch.flip(a, [2])
        return (a - MEAN) / STD, int(self.y[i])


class VariantDS(Dataset):
    """emits (full, leaf, bg) tensors + label."""
    def __init__(self, df):
        self.bp = df.bundle_path.tolist()
        self.y = df.unified_label.map(C2I).to_numpy()
        self.area = []

    def __len__(self): return len(self.bp)

    def __getitem__(self, i):
        rgb = load_rgb(self.bp[i])
        m = leaf_mask(rgb)
        self.area.append(float(m.mean()))
        f = rgb.astype(np.float32) / 255.0
        m3 = np.repeat(m[..., None], 3, 2)
        leaf = np.where(m3, f, GREY)
        bg = np.where(m3, GREY, f)
        def norm(x):
            t = torch.from_numpy(x).permute(2, 0, 1)
            return (t - MEAN) / STD
        return norm(f), norm(leaf), norm(bg), int(self.y[i])


def train_model(name, df, tr):
    net = timm.create_model(name, pretrained=True, num_classes=K).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    dl = DataLoader(TrainDS(df.iloc[tr], True), BATCH, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(EPOCHS):
        net.train(); t0 = time.time()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = ce(net(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        print(f"  [{name}] ep{ep+1}/{EPOCHS} ({time.time()-t0:.0f}s)", flush=True)
    return net


def eval_variants(net, df, te):
    ds = VariantDS(df.iloc[te])
    dl = DataLoader(ds, 32, shuffle=False, num_workers=2)
    net.eval()
    P = {"full": [], "leaf": [], "bg": []}
    Y = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for f, lf, bg, y in dl:
            for key, x in (("full", f), ("leaf", lf), ("bg", bg)):
                P[key].append(net(x.to(DEV)).float().argmax(1).cpu().numpy())
            Y.append(y.numpy())
    Y = np.concatenate(Y)
    acc = {k: float(accuracy_score(Y, np.concatenate(v))) for k, v in P.items()}
    f1 = {k: float(f1_score(Y, np.concatenate(v), average="macro")) for k, v in P.items()}
    return acc, f1, float(np.mean(ds.area))


def main():
    print("device", DEV, "| rembg", HAVE_REMBG, "| cv2", HAVE_CV2, flush=True)
    df = load_raw()
    y = df.unified_label.map(C2I).to_numpy(); g = df.group_id.to_numpy()
    tr, te = next(iter(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                       .split(df, y, groups=g)))
    out = []
    for name in ["mobilevit_s", "resnet50"]:
        print(f"\n== {name} ==", flush=True)
        net = train_model(name, df, tr)
        acc, f1, area = eval_variants(net, df, te)
        rec = {"model": name, "seg_area_frac": area,
               "acc_full": acc["full"], "acc_leaf": acc["leaf"], "acc_bg": acc["bg"],
               "mF1_full": f1["full"], "mF1_leaf": f1["leaf"], "mF1_bg": f1["bg"],
               "delta_bg": acc["bg"] - 1.0 / K,
               "delta_leaf": acc["full"] - acc["leaf"]}
        out.append(rec)
        json.dump(out, open(f"{OUT}/rq3_results.json", "w"), indent=2)
        print("  =>", json.dumps(rec), flush=True)

    L = ["# RQ3 --- Clever-Hans background probe (raw corpus, S2 fold 0)", "",
         f"Segmentation: {'rembg/U^2-Net' if HAVE_REMBG else 'GrabCut'}; "
         f"mean foreground area = {np.mean([r['seg_area_frac'] for r in out]):.2f}. "
         f"Chance = {100.0/K:.1f}%.", "",
         "| model | Acc full | Acc leaf-only | Acc bg-only | Δ_bg (pp) | Δ_leaf (pp) |",
         "|---|---|---|---|---|---|"]
    for r in out:
        L.append(f"| {r['model']} | {r['acc_full']*100:.1f} | {r['acc_leaf']*100:.1f} | "
                 f"{r['acc_bg']*100:.1f} | {r['delta_bg']*100:+.1f} | {r['delta_leaf']*100:+.1f} |")
    L += ["", "Δ_bg >> 0 means the background alone predicts the class (dataset defect).",
          "Large Δ_leaf means the model leans on context that will not transfer."]
    open(f"{OUT}/rq3_results.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
