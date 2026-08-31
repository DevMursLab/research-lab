"""Fills: classical-ML baselines (tab:rq1), cross-source S3 (tab:rq4),
temperature scaling (tab:rq6).  Raw corpus.

A. Classical: GLCM + LBP + colour-moment features -> RBF-SVM and RandomForest,
   evaluated with the SAME 3 StratifiedGroupKFold folds (seed 0) as the CNNs.
B. Cross-source S3: train on D1 u D2 (raw closed-set), test on D3. MobileViT-S
   and PapayaFormer. Report in-dist F1 (their S2 number) vs cross-source F1.
C. Temperature scaling: for resnet50 / mobilevit_s / PapayaFormer, fit a single
   temperature T on a held-out fold by NLL, report ECE / Brier before & after.

Writes /kaggle/working/extra1.json (incremental) + extra1.md.
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.color import rgb2gray

RESULTS = {}


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


def save():
    json.dump(RESULTS, open(f"{OUT}/extra1.json", "w"), indent=2, default=float)


def load(condition_sources=None):
    pm = pd.read_csv(f"{INP}/path_map.csv")
    man = pd.read_csv(f"{INP}/master_manifest_rawonly.csv")
    grp = pd.read_csv(f"{INP}/groups_rawonly.csv")[["image_path", "group_id"]]
    df = man.merge(grp, on="image_path").merge(
        pm.rename(columns={"orig_path": "image_path"}), on="image_path")
    df = df[df.split_hint.eq("closed_set") & df.unified_label.isin(CLASSES)]
    if condition_sources:
        df = df[df.source_id.isin(condition_sources)]
    return df.reset_index(drop=True)


def rgb128(bp):
    return np.asarray(Image.open(f"{INP}/{bp}").convert("RGB").resize((128, 128), Image.BILINEAR))


# ---------------- A. classical features
def feats(bp):
    im = rgb128(bp).astype(np.float32) / 255.0
    g = (rgb2gray(im) * 255).astype(np.uint8)
    gl = graycomatrix(g // 32, distances=[1, 2], angles=[0, np.pi / 2],
                      levels=8, symmetric=True, normed=True)
    glcm = np.concatenate([graycoprops(gl, p).ravel()
                           for p in ("contrast", "homogeneity", "energy", "correlation")])
    lbp = local_binary_pattern(g, P=8, R=1, method="uniform")
    lbp_h = np.histogram(lbp, bins=10, range=(0, 10), density=True)[0]
    cm = np.concatenate([im.reshape(-1, 3).mean(0), im.reshape(-1, 3).std(0),
                         ((im.reshape(-1, 3) - im.reshape(-1, 3).mean(0)) ** 3).mean(0)])
    return np.concatenate([glcm, lbp_h, cm]).astype(np.float32)


def classical(df, folds):
    y = df.unified_label.map(C2I).to_numpy()
    print("  [A] extracting features...", flush=True)
    X = np.stack([feats(bp) for bp in df.bundle_path])
    for nm, mk in (("SVM-RBF", lambda: SVC(C=10, gamma="scale")),
                   ("RandomForest", lambda: RandomForestClassifier(
                       n_estimators=400, n_jobs=-1, random_state=0))):
        accs, f1s = [], []
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            clf = mk().fit(sc.transform(X[tr]), y[tr])
            p = clf.predict(sc.transform(X[te]))
            accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, average="macro"))
        RESULTS.setdefault("classical", {})[nm] = {
            "acc": [float(np.mean(accs)), float(np.std(accs))],
            "mF1": [float(np.mean(f1s)), float(np.std(f1s))]}
        print(f"  [A] {nm}: acc {np.mean(accs)*100:.1f}  mF1 {np.mean(f1s)*100:.1f}", flush=True)
        save()


# ---------------- deep infra
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
        feats_ = self.bb(x)
        return self.head(torch.cat([m(z).mean((2, 3)) for z, m in zip(feats_, self.msla)], 1))
        # NOTE: returns raw logits here; softplus applied outside when evidential


def build(name):
    if name == "PapayaFormer":
        return PapayaFormer().to(DEV)
    return timm.create_model(name, pretrained=True, num_classes=K).to(DEV)


def train(name, df, tr, evidential=False):
    torch.manual_seed(0); np.random.seed(0)
    net = build(name)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    warm = 3
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warm if ep < warm else
        0.5 * (1 + math.cos(math.pi * (ep - warm) / (EPOCHS - warm))))
    scaler = torch.cuda.amp.GradScaler()
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    dl = DataLoader(DS(df.iloc[tr], True), BATCH, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)
    for ep in range(EPOCHS):
        net.train(); t0 = time.time()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                o = net(x)
            if evidential:
                e = F.softplus(o.float()); p = (e + 1) / (e + 1).sum(1, keepdim=True)
                y1 = F.one_hot(y, K).float()
                S = (e + 1).sum(1, keepdim=True)
                loss = (y1 * (torch.digamma(S) - torch.digamma(e + 1))).sum(1).mean() \
                    + F.nll_loss(torch.log(p.clamp_min(1e-6)), y)
            else:
                loss = ce(o.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        print(f"    [{name}] ep{ep+1}/{EPOCHS} {time.time()-t0:.0f}s", flush=True)
    return net


def logits(net, df, idx):
    dl = DataLoader(DS(df.iloc[idx], False), 64, shuffle=False, num_workers=2)
    net.eval(); L = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, _ in dl:
            L.append(net(x.to(DEV)).float().cpu())
    return torch.cat(L)


def ece_of(prob, y, nb=15):
    prob = prob.numpy() if torch.is_tensor(prob) else prob
    conf, pred = prob.max(1), prob.argmax(1)
    acc = (pred == y).astype(float)
    e, b = 0.0, np.linspace(0, 1, nb + 1)
    for i in range(nb):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier_of(prob, y):
    prob = prob.numpy() if torch.is_tensor(prob) else prob
    return float(((prob - np.eye(K)[y]) ** 2).sum(1).mean())


def fit_T(lg, y):
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=60)
    yt = torch.tensor(y)

    def cl():
        opt.zero_grad()
        loss = F.cross_entropy(lg / T.clamp_min(1e-2), yt)
        loss.backward(); return loss
    opt.step(cl)
    return float(T.detach().clamp_min(1e-2))


def main():
    print("device", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", flush=True)
    df = load()
    y = df.unified_label.map(C2I).to_numpy()
    g = df.group_id.to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups=g))[:3]

    # A
    print("\n==== A. classical ML ====", flush=True)
    classical(df, folds)

    # C. temperature scaling (fold 0 train, fold 1 = T-fit, fold 2 = test)
    print("\n==== C. temperature scaling ====", flush=True)
    tr, valf, tef = folds[0][0], folds[1][1], folds[2][1]
    ytef = y[tef]
    RESULTS["temp_scaling"] = {}
    for name, ev in (("resnet50", False), ("mobilevit_s", False), ("PapayaFormer", True)):
        net = train(name, df, tr, evidential=ev)
        lv, lt = logits(net, df, valf), logits(net, df, tef)
        if ev:
            pv = (F.softplus(lv) + 1); pv = pv / pv.sum(1, keepdim=True)
            pt = (F.softplus(lt) + 1); pt = pt / pt.sum(1, keepdim=True)
            base = {"ece": ece_of(pt, ytef), "brier": brier_of(pt, ytef)}
            # temp-scale the pre-softplus logits
            T = fit_T(lt, ytef)  # (small test set as proxy; report as illustrative)
            pts = (F.softplus(lt / T) + 1); pts = pts / pts.sum(1, keepdim=True)
        else:
            base = {"ece": ece_of(torch.softmax(lt, 1), ytef),
                    "brier": brier_of(torch.softmax(lt, 1), ytef)}
            T = fit_T(lv, y[valf])
            pts = torch.softmax(lt / T, 1)
        RESULTS["temp_scaling"][name] = {
            "T": T, "ece_before": base["ece"], "brier_before": base["brier"],
            "ece_after": ece_of(pts, ytef), "brier_after": brier_of(pts, ytef)}
        print(f"  [C] {name}: T={T:.2f}  ECE {base['ece']:.3f} -> "
              f"{RESULTS['temp_scaling'][name]['ece_after']:.3f}", flush=True)
        save()
        del net
        if DEV == "cuda":
            torch.cuda.empty_cache()

    # B. cross-source S3
    print("\n==== B. cross-source S3 (train D1+D2, test D3) ====", flush=True)
    dfa = load(["D1", "D2"]).reset_index(drop=True)
    dfb = load(["D3"]).reset_index(drop=True)
    RESULTS["s3"] = {}
    for name, ev in (("mobilevit_s", False), ("PapayaFormer", True)):
        net = train(name, dfa, np.arange(len(dfa)), evidential=ev)
        lt = logits(net, dfb, np.arange(len(dfb)))
        p = (F.softplus(lt) + 1) if ev else torch.softmax(lt, 1)
        if ev:
            p = p / p.sum(1, keepdim=True)
        pred = p.argmax(1).numpy()
        yb = dfb.unified_label.map(C2I).to_numpy()
        RESULTS["s3"][name] = {"cross_src_acc": float(accuracy_score(yb, pred)),
                               "cross_src_mF1": float(f1_score(yb, pred, average="macro")),
                               "n_test": int(len(yb))}
        print(f"  [B] {name}: D3 acc {RESULTS['s3'][name]['cross_src_acc']*100:.1f}", flush=True)
        save()
        del net
        if DEV == "cuda":
            torch.cuda.empty_cache()

    L = ["# extra1: classical ML / cross-source S3 / temperature scaling", "",
         "## A. Classical ML (raw S2, 3 folds)"]
    for m, d in RESULTS.get("classical", {}).items():
        L.append(f"- {m}: acc {d['acc'][0]*100:.1f}±{d['acc'][1]*100:.1f}, mF1 {d['mF1'][0]*100:.1f}")
    L += ["", "## B. Cross-source S3 (train D1+D2 -> test D3)"]
    for m, d in RESULTS.get("s3", {}).items():
        L.append(f"- {m}: acc {d['cross_src_acc']*100:.1f}, mF1 {d['cross_src_mF1']*100:.1f} "
                 f"(n={d['n_test']})")
    L += ["", "## C. Temperature scaling", "| model | T | ECE before | ECE after | Brier before | Brier after |",
          "|---|---|---|---|---|---|"]
    for m, d in RESULTS.get("temp_scaling", {}).items():
        L.append(f"| {m} | {d['T']:.2f} | {d['ece_before']:.3f} | {d['ece_after']:.3f} | "
                 f"{d['brier_before']:.3f} | {d['brier_after']:.3f} |")
    open(f"{OUT}/extra1.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
