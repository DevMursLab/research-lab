# Data placement

The raw images are **not** stored in this repository (licence + size). Download the three
source datasets from Mendeley Data and lay them out as below, then run the pipeline in
[`../src/`](../src/).

| Folder | Dataset | DOI | Notes |
|:--|:--|:--|:--|
| `data/D1/` | Papaya Leaf Disease Image Dataset | [`10.17632/3kwgxg4stb.1`](https://doi.org/10.17632/3kwgxg4stb.1) | **raw images only** — do *not* copy the `~18k` pre-augmented folder |
| `data/D2/` | BDPapayaLeaf | [`10.17632/p997fvf526.1`](https://doi.org/10.17632/p997fvf526.1) | original images (`2 159`), not the annotated crops |
| `data/D3/` | Papaya Diseases Dataset (Sethy) | [`10.17632/yvcwypp8rz.1`](https://doi.org/10.17632/yvcwypp8rz.1) | India source; only corpus with Powdery Mildew; `Phytophthora` used as the open-set class |

All three are released under **CC BY 4.0**.

## Expected layout

```
data/
├── D1/
│   ├── Anthracnose/*.png
│   ├── BacterialSpot/*.png
│   ├── Healthy/*.png
│   └── ...                 # one folder per raw class label
├── D2/
│   ├── Anthracnose/*.jpg
│   └── ...
└── D3/
    ├── Anthracnose Diease/*.jpg
    └── ...
```

The parent folder name is read as the raw label; label unification to the six canonical
classes is in [`../src/common.py`](../src/common.py) (`LABEL_MAP`).

## Then

```bash
python ../src/02_verify_integrity.py   # MD5, resolution, corrupt-file check
python ../src/03_build_manifest.py     # unified manifest + label map
python ../src/04_dataset_stats.py      # class distribution (prints LaTeX)
python ../src/05_hashes.py             # MD5 + 64-bit pHash for every image
```
