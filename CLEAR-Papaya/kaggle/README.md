# Kaggle GPU kernels

Every heavy experiment is a **single self-contained script** that runs as a Kaggle Notebook
(script type), reads one attached image bundle, and writes result JSON/MD (+ figures) to
`/kaggle/working/`. Each script:

* auto-installs a `sm_60`-compatible `torch` if it lands on a P100,
* resolves the dataset path by globbing `/kaggle/input/**/path_map.csv` (mount name varies),
* saves results **incrementally**, so a 12-hour timeout never loses completed runs.

| Kernel | Produces |
|:--|:--|
| `kernel_baselines`    | ResNet-50 / EfficientNet-B0 / MobileViT-S, S1 vs S2, both corpora |
| `kernel_papayaformer` | PapayaFormer + baselines on shared folds, ECE/Brier, rejection, paired stats |
| `kernel_sig`          | 5 seeds × 5 folds significance (paired *t* + Wilcoxon + Holm) |
| `kernel_ablation`     | MSLA / evidential-head / γ-init ablation |
| `kernel_backbones`    | DenseNet-121, ConvNeXt-T, MobileNetV3-L, ViT-B/16, DeiT3-S, Swin-T |
| `kernel_rq456`        | corruption suite, open-set (Phytophthora), low-data curve, latency, MCC/κ/AUROC |
| `kernel_rq3`          | Clever-Hans background probe (V_full / V_leaf / V_bg) |
| `kernel_extra1`       | classical GLCM+LBP→SVM/RF, cross-source S3, temperature scaling |
| `kernel_pcs3`         | per-class table + S3 per-class breakdown + confusion matrices |
| `kernel_xai`          | Grad-CAM++ / MSLA deletion–insertion AUC, stability, qualitative grid |

## Run one

```bash
# build the image bundle once (resizes D1/D2/D3 to 256 px, ships CSVs)
#   -> upload as a private Kaggle Dataset, then:
kaggle kernels push -p kaggle/kernel_sig
kaggle kernels status  <your-slug>/clear-papaya-sig
kaggle kernels output  <your-slug>/clear-papaya-sig -p results/
```

Edit the `id` in each `kernel-metadata.json` to your own Kaggle slug, and set
`dataset_sources` to your uploaded bundle.
