"""Shared config and IO helpers for CLEAR-Papaya Phase 1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------ audit params (paper Section 4.6)
THETA_P = 5          # pHash Hamming-distance threshold for near-duplicates
THETA_E = 0.95       # embedding cosine threshold for "same physical leaf" (calibrate!)
KNN_K = 20           # neighbours per image when building the embedding similarity graph
N_RANDOM_SPLITS = 10 # repeats for the leakage-rate estimate
SPLIT_FRACS = (0.70, 0.15, 0.15)  # train / val / test
N_FOLDS = 5          # StratifiedGroupKFold for S2
SEED = 0

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ------------------------------------------------------------------ label unification
# Map every raw class string (lower-cased, stripped) to a canonical class.
# Extend this as you discover new spellings in the downloaded datasets.
CANONICAL_CLASSES = [
    "healthy",
    "prsv",              # papaya ringspot virus
    "anthracnose",
    "powdery_mildew",
    "bacterial_leaf_spot",
    "mite_or_deficiency",
]

LABEL_MAP = {
    # healthy
    "healthy": "healthy", "health": "healthy", "fresh": "healthy", "normal": "healthy",
    "healthy leaf": "healthy", "healthy_leaf": "healthy",
    "good papaya": "healthy",  # D3 (Sethy) label for healthy fruit
    # PRSV / ringspot / mosaic
    "prsv": "prsv", "ring spot": "prsv", "ringspot": "prsv", "ring_spot": "prsv",
    "papaya ring spot": "prsv", "papaya ringspot virus": "prsv", "ring spot virus": "prsv",
    "mosaic": "prsv", "mosaic virus": "prsv", "leaf curl": "prsv", "curl": "prsv",
    "papaya_ringspot": "prsv", "virus": "prsv",
    "ring spot diease": "prsv",  # D3 (Sethy) typo: "Diease"
    # anthracnose
    "anthracnose": "anthracnose", "anthracnos": "anthracnose",
    "brown spot": "anthracnose", "brownspot": "anthracnose",
    "anthracanose diease": "anthracnose",  # D3 (Sethy) typo: "Anthracanose Diease"
    # powdery mildew
    "powdery mildew": "powdery_mildew", "powdery_mildew": "powdery_mildew",
    "powdery": "powdery_mildew", "mildew": "powdery_mildew", "powdary mildew": "powdery_mildew",
    "powdery mildery diease": "powdery_mildew",  # D3 (Sethy) typo: "Powdery Mildery Diease"
    # bacterial leaf spot
    "bacterial leaf spot": "bacterial_leaf_spot", "bacterial spot": "bacterial_leaf_spot",
    "bacterial_spot": "bacterial_leaf_spot", "bacterial": "bacterial_leaf_spot",
    "bl spot": "bacterial_leaf_spot", "leaf spot": "bacterial_leaf_spot",
    "black spot diease": "bacterial_leaf_spot",  # D3 (Sethy): "Black Spot Diease" ~= bacterial leaf spot
    "bacterialspot": "bacterial_leaf_spot",  # D1/D2 folder name "BacterialSpot" (no space/underscore)
    # mite / nutrient deficiency / physiological
    "mite": "mite_or_deficiency", "spider mite": "mite_or_deficiency",
    "red mite": "mite_or_deficiency", "mites": "mite_or_deficiency",
    "mite disease": "mite_or_deficiency",
    "mealybug": "mite_or_deficiency",  # D1: pest damage, grouped with mite (user decision 2026-08-28)
    "nutrient deficiency": "mite_or_deficiency", "deficiency": "mite_or_deficiency",
    "mineral deficiency": "mite_or_deficiency", "nutrient": "mite_or_deficiency",
    "yellow leaf": "mite_or_deficiency", "yellowing": "mite_or_deficiency",
}

# open-set only (never in training) — kept separate on purpose
OPEN_SET_LABELS = {
    "herbicide": "herbicide_injury", "herbicide injury": "herbicide_injury",
    "abiotic": "herbicide_injury", "non-papaya": "non_papaya", "other": "non_papaya",
}


def canon_label(raw: str) -> str | None:
    """Return canonical class for a raw label string, or None if unmapped."""
    key = " ".join(str(raw).strip().lower().replace("_", " ").replace("-", " ").split())
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    key2 = key.replace(" ", "_")
    return LABEL_MAP.get(key2)


def iter_images(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"  wrote {path.relative_to(ROOT)}")


def load_json(path: Path):
    return json.loads(path.read_text())


def require(path: Path, produced_by: str):
    if not path.exists():
        sys.exit(f"ERROR: missing {path.relative_to(ROOT)} — run {produced_by} first.")
    return path
