"""CLEAR-Papaya full-scale baselines on Kaggle GPU.

Full-scale S1 (leaky random) vs S2 (group-disjoint) comparison for
tab:rq2 / tab:main, at 224px with proper convergence.

Reads the companion dataset:
  /kaggle/input/clear-papaya-bundle/img/XXXXX.jpg
  /kaggle/input/clear-papaya-bundle/path_map.csv
  /kaggle/input/clear-papaya-bundle/master_manifest.csv
  /kaggle/input/clear-papaya-bundle/master_manifest_rawonly.csv
  /kaggle/input/clear-papaya-bundle/groups.csv
  /kaggle/input/clear-papaya-bundle/groups_rawonly.csv

Writes /kaggle/working/baselines_results.json  (appended incrementally, so a
timeout still leaves partial results) and baselines_results.md.
"""
import json, time, os, sys, subprocess

# --- GPU arch guard: Kaggle sometimes assigns a P100 (sm_60) that the preinstalled
# torch no longer supports. Fall back to a torch build that still ships sm_60. ---
import torch
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7 \
        and os.environ.get("_ARCH_FIX") != "1":
    print("P100/sm_60 detected -> installing torch 2.5.1 (has sm_60 kernels)", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.5.1", "torchvision==0.20.1"], check=False)
    os.environ["_ARCH_FIX"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import glob
import numpy as np, pandas as pd, torch.nn as nn, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold

# --- locate the attached dataset regardless of its mount folder name ---
def _resolve_inp():
    for i in range(180):
        top = os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else []
        hits = glob.glob("/kaggle/input/**/path_map.csv", recursive=True)
        if hits:
            d = os.path.dirname(hits[0])
            print("resolved INP =", d, "| files:", os.listdir(d)[:8], flush=True)
            return d
        if i % 6 == 0:
            print(f"[{i*5}s] /kaggle/input = {top}", flush=True)
        time.sleep(5)
    raise RuntimeError("dataset never appeared under /kaggle/input")

INP = _resolve_inp()
OUT = "/kaggle/working"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["anthracnose", "bacterial_leaf_spot", "healthy",
           "mite_or_deficiency", "powdery_mildew", "prsv"]
C2I = {c: i for i, c in enumerate(CLASSES)}
RES = 224
EPOCHS = 12
BATCH = 64
S1_REPEATS = 2
S2_FOLDS = 3
MODELS = ["resnet50", "tf_efficientnet_b0", "mobilevit_s"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_condition(cond):
    pm = pd.read_csv(f"{INP}/path_map.csv")
    if cond == "raw_only":
        man = pd.read_csv(f"{INP}/master_manifest_rawonly.csv")
        grp = pd.read_csv(f"{INP}/groups_rawonly.csv")[["image_path", "group_id"]]
    else:
        man = pd.read_csv(f"{INP}/master_manifest.csv")
        grp = pd.read_csv(f"{INP}/groups.csv")[["image_path", "group_id"]]
        man = man[~man.source_id.isin(["D4", "D5"])]
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
            if a.shape[1] > RES:
                t = np.random.randint(0, a.shape[1] - RES + 1)
                l = np.random.randint(0, a.shape[2] - RES + 1)
                a = a[:, t:t + RES, l:l + RES]
            if np.random.rand() < 0.5:
                a = torch.flip(a, [2])
        else:
            o = (a.shape[1] - RES) // 2
            a = a[:, o:o + RES, o:o + RES]
        return (a - MEAN) / STD, int(self.y[i])


def run(model_name, df, tr, te, wts, tag):
    try:
        net = timm.create_model(model_name, pretrained=True, num_classes=len(CLASSES)).to(DEV)
    except Exception as e:  # unknown timm name / download issue -> skip this model
        print(f"  !! skip {model_name}: {e}", flush=True)
        return None
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    lossf = nn.CrossEntropyLoss(weight=wts.to(DEV), label_smoothing=0.1)
    dltr = DataLoader(DS(df.iloc[tr], True), batch_size=BATCH, shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True)
    dlte = DataLoader(DS(df.iloc[te], False), batch_size=BATCH, shuffle=False, num_workers=2)
    best = (0.0, 0.0)
    for ep in range(EPOCHS):
        net.train(); t0 = time.time()
        for x, y in dltr:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = lossf(net(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        net.eval(); P, Y = [], []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for x, y in dlte:
                P.append(net(x.to(DEV)).argmax(1).cpu().numpy()); Y.append(y.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        acc, f1 = accuracy_score(Y, P), f1_score(Y, P, average="macro")
        best = max(best, (acc, f1))
        print(f"  [{tag}] {model_name} ep{ep+1}/{EPOCHS} acc={acc:.3f} mF1={f1:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return best


def append_result(rec):
    p = f"{OUT}/baselines_results.json"
    data = json.load(open(p)) if os.path.exists(p) else []
    data.append(rec); json.dump(data, open(p, "w"), indent=2)


def main():
    print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
    for cond in ["raw_only", "as_published"]:
        df = load_condition(cond)
        y = df.unified_label.map(C2I).to_numpy(); g = df.group_id.to_numpy()
        wts = torch.tensor([len(y) / (len(CLASSES) * max(1, (y == i).sum()))
                            for i in range(len(CLASSES))], dtype=torch.float32)
        print(f"\n==== {cond}  N={len(df)} ====", flush=True)
        for m in MODELS:
            s1 = []
            for s in range(S1_REPEATS):
                tr, te = next(StratifiedShuffleSplit(1, test_size=0.3, random_state=s).split(df, y))
                s1.append(run(m, df, tr, te, wts, f"{cond}/S1.{s}"))
            s2 = []
            sgkf = StratifiedGroupKFold(n_splits=max(S2_FOLDS, 3), shuffle=True, random_state=0)
            for k, (tr, te) in enumerate(sgkf.split(df, y, groups=g)):
                if k >= S2_FOLDS: break
                assert not (set(g[tr]) & set(g[te]))
                s2.append(run(m, df, tr, te, wts, f"{cond}/S2.{k}"))
            s1 = [x for x in s1 if x]; s2 = [x for x in s2 if x]
            if not s1 or not s2:
                print(f"  (no results for {m} under {cond})", flush=True); continue
            a1 = np.array([x[0] for x in s1]); a2 = np.array([x[0] for x in s2])
            f1a = np.array([x[1] for x in s1]); f2a = np.array([x[1] for x in s2])
            rec = {"condition": cond, "model": m, "n": int(len(df)),
                   "S1_acc": [float(a1.mean()), float(a1.std())],
                   "S2_acc": [float(a2.mean()), float(a2.std())],
                   "S1_mF1": [float(f1a.mean()), float(f1a.std())],
                   "S2_mF1": [float(f2a.mean()), float(f2a.std())],
                   "acc_inflation": float(a1.mean() - a2.mean()),
                   "mF1_inflation": float(f1a.mean() - f2a.mean())}
            append_result(rec)
            print("  => ", json.dumps(rec), flush=True)

    data = json.load(open(f"{OUT}/baselines_results.json"))
    lines = ["# Full-scale baselines (224px, GPU)", "",
             "| condition | model | S1 acc | S2 acc | acc infl | S1 mF1 | S2 mF1 | mF1 infl |",
             "|---|---|---|---|---|---|---|---|"]
    for r in data:
        lines.append(f"| {r['condition']} | {r['model']} | "
                     f"{r['S1_acc'][0]*100:.1f}±{r['S1_acc'][1]*100:.1f} | "
                     f"{r['S2_acc'][0]*100:.1f}±{r['S2_acc'][1]*100:.1f} | "
                     f"+{r['acc_inflation']*100:.1f} | "
                     f"{r['S1_mF1'][0]*100:.1f} | {r['S2_mF1'][0]*100:.1f} | "
                     f"+{r['mF1_inflation']*100:.1f} |")
    open(f"{OUT}/baselines_results.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
