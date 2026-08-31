"""Task 1.2 — verify file integrity of every downloaded dataset.

Scans data/<source>/ recursively, opens every image, records corrupt/truncated
files and a per-source count. Writes outputs/integrity_report.json and a
README stub per source folder.
"""
from __future__ import annotations

import warnings
from collections import defaultdict

from PIL import Image, ImageFile

from common import DATA_DIR, OUT_DIR, ROOT, iter_images, save_json

ImageFile.LOAD_TRUNCATED_IMAGES = False  # we WANT truncated files to raise


def check_image(path):
    try:
        with Image.open(path) as im:
            im.verify()                       # header / structure
        with Image.open(path) as im:
            im.load()                         # full decode
            w, h = im.size
            if w < 8 or h < 8:
                return "too_small", (w, h)
        return "ok", (w, h)
    except Exception as e:                     # noqa: BLE001
        return f"corrupt:{type(e).__name__}", None


def main():
    sources = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    if not sources:
        print(f"No source folders under {DATA_DIR}. Download datasets into data/D1, data/D2, ...")
        return

    report = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        for src in sources:
            imgs = list(iter_images(src))
            status = defaultdict(list)
            for p in imgs:
                st, _ = check_image(p)
                status[st.split(":")[0] if st.startswith("corrupt") else st].append(str(p.relative_to(ROOT)))
            n_ok = len(status.get("ok", []))
            n_bad = len(imgs) - n_ok
            report[src.name] = {
                "path": str(src.relative_to(ROOT)),
                "n_files": len(imgs),
                "n_ok": n_ok,
                "n_bad": n_bad,
                "bad_files": {k: v for k, v in status.items() if k != "ok"},
            }
            print(f"{src.name:8s}  files={len(imgs):6d}  ok={n_ok:6d}  bad={n_bad}")

            readme = src / "README.md"
            if not readme.exists():
                readme.write_text(
                    f"# {src.name}\n\n"
                    f"- Source URL: TODO\n- DOI: TODO\n- Licence: TODO\n"
                    f"- Downloaded: TODO\n- Files found: {len(imgs)}\n"
                    f"- Published image count (from source description): TODO\n"
                    f"- Count matches published? TODO\n"
                )

    save_json(report, OUT_DIR / "integrity_report.json")
    print("\nNEXT: for each source, fill data/<source>/README.md and confirm the file")
    print("count matches the published description. Note any mismatch — reviewers check this.")


if __name__ == "__main__":
    main()
