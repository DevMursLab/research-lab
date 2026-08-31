"""Task 2.2 (local CPU variant) — DINOv2 embeddings via timm, no torch.hub.

Writes the exact files 07_similarity_graph.py expects:
    outputs/embeddings_dinov2_vits14.npy         float32 [N, 384], L2-normalised
    outputs/embeddings_dinov2_vits14_index.csv   image_path in row order

Run:  python src/06b_embeddings_timm.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import OUT_DIR, ROOT, require

MODEL_TAG = "dinov2_vits14"
TIMM_NAME = "vit_small_patch14_dinov2.lvd142m"


class ImgDS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        fp = ROOT / self.paths[i]
        try:
            im = Image.open(fp).convert("RGB")
        except Exception:  # noqa: BLE001
            im = Image.new("RGB", (224, 224))
        return self.tf(im), i


def main():
    man = require(OUT_DIR / "master_manifest.csv", "03_build_manifest.py")
    paths = pd.read_csv(man)["image_path"].tolist()
    n = len(paths)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"device={dev}  N={n}  model={TIMM_NAME}")

    # dynamic_img_size lets the ViT accept 224px input (pos-embed interpolated)
    # instead of the checkpoint-native 518px -> ~5x less CPU compute.
    model = timm.create_model(TIMM_NAME, pretrained=True, num_classes=0,
                              dynamic_img_size=True).eval().to(dev)
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dl = DataLoader(ImgDS(paths, tf), batch_size=32, num_workers=4,
                    shuffle=False, pin_memory=False)

    feats = [None] * n
    done = 0
    t0 = time.time()
    with torch.no_grad():
        for xb, idx in dl:
            z = model(xb.to(dev))
            z = torch.nn.functional.normalize(z, dim=1).cpu().numpy()
            for k, j in zip(range(len(idx)), idx.tolist()):
                feats[j] = z[k]
            done += len(idx)
            if done % 320 == 0 or done == n:
                el = time.time() - t0
                eta = el / done * (n - done)
                print(f"  {done}/{n}  elapsed={el/60:.1f}m  eta={eta/60:.1f}m", flush=True)

    arr = np.stack(feats).astype("float32")
    np.save(OUT_DIR / f"embeddings_{MODEL_TAG}.npy", arr)
    pd.DataFrame({"image_path": paths}).to_csv(
        OUT_DIR / f"embeddings_{MODEL_TAG}_index.csv", index=False)
    print(f"wrote outputs/embeddings_{MODEL_TAG}.npy  shape={arr.shape}")


if __name__ == "__main__":
    main()
