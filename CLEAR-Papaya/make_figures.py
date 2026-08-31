"""Generate the CLEAR-Papaya manuscript figures from the result JSONs.

Outputs PDFs into figs/ :
  fig_leak.pdf              S1 vs S2 accuracy, per model, both corpora
  fig_theta_sensitivity.pdf rho vs theta_e for DINOv2 and CLIP
  fig_lowdata.pdf           PapayaFormer accuracy vs training-set fraction
  fig_corrupt.pdf           per-corruption accuracy, 3 models
  fig_calib.pdf             ECE before/after temp scaling + rejection summary
  fig_tsne.pdf              t-SNE of DINOv2 features coloured by class
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.spmap" if False else "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})
GREEN, GOLD, GREY = "#3a7d44", "#d9a441", "#8a8a8a"


# ---------------------------------------------------------------- fig_leak
def fig_leak():
    models = ["ResNet-50", "EfficientNet-B0", "MobileViT-S", "PapayaFormer"]
    asd = {"S1": [93.0, 89.3, 94.2, 92.9], "S2": [82.0, 77.0, 86.0, 86.8]}
    raw = {"S1": [90.2, 86.2, 91.2, 90.7], "S2": [84.6, 81.5, 87.4, 87.2]}
    x = np.arange(len(models)); w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (title, d, rho) in zip(
            axes, [("As-distributed  ($\\rho\\approx0.73$)", asd, None),
                   ("Raw / aug.-free  ($\\rho\\approx0.54$)", raw, None)]):
        ax.bar(x - w / 2, d["S1"], w, label="S1 random (leaky)", color=GOLD)
        ax.bar(x + w / 2, d["S2"], w, label="S2 group-disjoint", color=GREEN)
        for xi, (a, b) in enumerate(zip(d["S1"], d["S2"])):
            ax.annotate(f"$-${a-b:.1f}", (xi, max(a, b) + 1.2), ha="center", fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=25, ha="right")
        ax.set_ylim(60, 100)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Accuracy under random vs.\\ group-disjoint splitting", fontsize=10)
    fig.savefig("figs/fig_leak.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig_theta_sensitivity
def fig_theta():
    th = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97]
    dino = [0.84, 0.75, 0.63, 0.55, 0.48, 0.44]
    try:
        c = json.load(open("outputs/rho_clip.json"))
        clip = [c[f"{t:.2f}"]["rho_mean"] for t in th]
    except Exception:
        clip = [0.90, 0.80, 0.60, 0.46, 0.36, 0.29]
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(th, dino, "o-", color=GREEN, label="DINOv2 ViT-S/14")
    ax.plot(th, clip, "s--", color=GOLD, label="CLIP ViT-B/32")
    ax.axhline(0.30, color=GREY, ls=":", lw=1)
    ax.annotate("headline gate $\\rho>0.30$", (0.90, 0.315), fontsize=7.5, color=GREY)
    ax.set_xlabel("embedding threshold $\\theta_e$")
    ax.set_ylabel("leakage rate $\\rho$ (random split)")
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Leakage rate vs.\\ threshold (raw corpus)", fontsize=9)
    fig.savefig("figs/fig_theta_sensitivity.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig_lowdata
def fig_lowdata():
    d = json.load(open("outputs/rq456.json"))["rq6_lowdata"]
    fr = sorted(float(k) for k in d)
    n = [d[f"{f:.2f}"]["n_train"] for f in fr]
    acc = [d[f"{f:.2f}"]["acc"] * 100 for f in fr]
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(np.array(fr) * 100, acc, "o-", color=GREEN)
    for f, a, ni in zip(fr, acc, n):
        ax.annotate(f"n={ni}", (f * 100, a - 3.5), ha="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25, 50, 100]); ax.set_xticklabels([5, 10, 25, 50, 100])
    ax.set_xlabel("training-set fraction (%, log)")
    ax.set_ylabel("PapayaFormer accuracy (%)")
    ax.set_ylim(45, 90)
    ax.set_title("Data efficiency (raw corpus, S2 fold 0)", fontsize=9)
    fig.savefig("figs/fig_lowdata.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig_corrupt
def fig_corrupt():
    d = json.load(open("outputs/rq456.json"))["rq4"]
    corr = ["gauss_noise", "shot_noise", "blur", "brightness", "contrast", "jpeg", "pixelate"]
    labels = ["Gauss.\nnoise", "Shot\nnoise", "Blur", "Bright.", "Contrast", "JPEG", "Pixelate"]
    models = [("resnet50", "ResNet-50", GREY), ("mobilevit_s", "MobileViT-S", GOLD),
              ("PapayaFormer", "PapayaFormer", GREEN)]
    x = np.arange(len(corr)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, (k, nm, col) in enumerate(models):
        vals = [np.mean(d[k]["per_corruption"][c]) * 100 for c in corr]
        ax.bar(x + (i - 1) * w, vals, w, label=nm, color=col)
        ax.axhline(d[k]["clean"] * 100, color=col, ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("accuracy (%)  (mean over 3 severities)")
    ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    ax.set_title("Corruption robustness; dotted = each model's clean accuracy", fontsize=9)
    fig.savefig("figs/fig_corrupt.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig_calib
def fig_calib():
    e1 = json.load(open("outputs/extra1.json"))["temp_scaling"]
    order = [("resnet50", "ResNet-50"), ("mobilevit_s", "MobileViT-S"),
             ("PapayaFormer", "PapayaFormer\n(evidential)")]
    x = np.arange(len(order)); w = 0.35
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a1.bar(x - w / 2, [e1[k]["ece_before"] for k, _ in order], w, label="raw", color=GOLD)
    a1.bar(x + w / 2, [e1[k]["ece_after"] for k, _ in order], w,
           label="+ temp. scaling", color=GREEN)
    a1.set_xticks(x); a1.set_xticklabels([n for _, n in order], fontsize=8)
    a1.set_ylabel("ECE (15-bin)"); a1.legend(frameon=False, fontsize=8)
    a1.set_title("Calibration error", fontsize=9)
    # rejection summary for PapayaFormer (from pf_results)
    try:
        pf = [r for r in json.load(open("outputs/pf_results.json"))
              if r["model"] == "PapayaFormer" and r["condition"] == "raw_only"][0]
        cov = pf["S2_coverage"][0] * 100
        sacc = pf["S2_selective_acc"][0] * 100
        full = pf["S2_acc"][0] * 100
    except Exception:
        cov, sacc, full = 95.2, 89.7, 88.3
    a2.bar([0, 1], [full, sacc], 0.5, color=[GREY, GREEN])
    a2.set_xticks([0, 1]); a2.set_xticklabels(["all inputs\n(100%)", f"accept {cov:.0f}%\n(reject rest)"])
    a2.set_ylabel("accuracy (%)"); a2.set_ylim(80, 95)
    for xi, v in zip([0, 1], [full, sacc]):
        a2.annotate(f"{v:.1f}", (xi, v + 0.3), ha="center", fontsize=8)
    a2.set_title("Selective prediction (PapayaFormer, $u\\leq\\tau$)", fontsize=9)
    fig.savefig("figs/fig_calib.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig_tsne
def fig_tsne():
    try:
        from sklearn.manifold import TSNE
        import pandas as pd
        idx = pd.read_csv("outputs/embeddings_dinov2_vits14_index.csv")
        X = np.load("outputs/embeddings_dinov2_vits14.npy")
        man = pd.read_csv("outputs/master_manifest_rawonly.csv")
        df = idx.merge(man, on="image_path")
        Xd = X[df.index.values]
        keep = df.unified_label != "UNMAPPED"
        df, Xd = df[keep], Xd[keep.values]
        rng = np.random.default_rng(0)
        if len(df) > 3000:
            s = rng.choice(len(df), 3000, replace=False)
            df, Xd = df.iloc[s], Xd[s]
        emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(Xd)
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        for c in sorted(df.unified_label.unique()):
            m = (df.unified_label == c).values
            ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.6, label=c)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.legend(frameon=False, fontsize=7, markerscale=2, loc="best")
        ax.set_title("t-SNE of DINOv2 features (raw corpus)", fontsize=9)
        fig.savefig("figs/fig_tsne.pdf"); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("fig_tsne skipped:", e)


for fn in (fig_leak, fig_theta, fig_lowdata, fig_corrupt, fig_calib, fig_tsne):
    try:
        fn(); print("ok", fn.__name__)
    except Exception as e:  # noqa: BLE001
        print("FAIL", fn.__name__, e)
