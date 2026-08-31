"""Task 2.2 — self-supervised embeddings for every image.  [GPU / Colab]

DINOv2 ViT-S/14 (Facebook, torch.hub) is the primary encoder.
CLIP ViT-B/32 is an optional robustness check (--model clip_vitb32; needs open-clip-torch).

Writes:
    outputs/embeddings_<model>.npy    float32 [N, D], L2-normalised
    outputs/embeddings_<model>_index.csv   image_path in row order

Colab:
    !pip install torch torchvision timm open-clip-torch imagehash pandas tqdm
    !python src/06_embeddings.py --model dinov2_vits14 --batch 64
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from common import OUT_DIR, ROOT, require


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


def build_dinov2():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def embed(x):
        return model(x)  # CLS token, [B, 384]

    return model, tf, embed


def build_clip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")

    def embed(x):
        return model.encode_image(x)

    return model, preprocess, embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dinov2_vits14",
                    choices=["dinov2_vits14", "clip_vitb32"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    man = require(OUT_DIR / "master_manifest.csv", "03_build_manifest.py")
    paths = pd.read_csv(man)["image_path"].tolist()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  model={args.model}  N={len(paths)}")
    if dev == "cpu":
        print("WARNING: CPU embedding of a full dataset is slow. Prefer Colab GPU.")

    model, tf, embed = build_dinov2() if args.model == "dinov2_vits14" else build_clip()
    model.eval().to(dev)

    dl = DataLoader(ImgDS(paths, tf), batch_size=args.batch, num_workers=args.workers,
                    shuffle=False, pin_memory=(dev == "cuda"))

    feats = [None] * len(paths)
    with torch.no_grad():
        for xb, idx in tqdm(dl, desc="embedding"):
            xb = xb.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
                z = embed(xb).float()
            z = torch.nn.functional.normalize(z, dim=1).cpu().numpy()
            for k, j in zip(range(len(idx)), idx.tolist()):
                feats[j] = z[k]

    arr = np.stack(feats).astype("float32")
    np.save(OUT_DIR / f"embeddings_{args.model}.npy", arr)
    pd.DataFrame({"image_path": paths}).to_csv(
        OUT_DIR / f"embeddings_{args.model}_index.csv", index=False)
    print(f"wrote outputs/embeddings_{args.model}.npy  shape={arr.shape}")


if __name__ == "__main__":
    main()
