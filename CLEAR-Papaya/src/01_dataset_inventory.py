"""Task 1.1 — inventory every public papaya leaf disease dataset.

Creates outputs/dataset_inventory.csv as a template. FILL IT BY HAND after your
Scholar / Kaggle / Mendeley / Roboflow / GitHub / Zenodo / UCI search.

Roadmap gate 1.3: a row with no DOI or no verifiable licence => download_tested=no
and role='EXCLUDE'.
"""
from __future__ import annotations

import csv
from common import OUT_DIR

TEMPLATE_ROWS = [
    # dataset_id, dataset_name, url, doi, image_count, classes, licence, download_tested, role, notes
    ["D1", "Papaya leaf disease (Mendeley Data)", "", "", "", "", "", "no", "primary_train_test", "VERIFY at data.mendeley.com"],
    ["D2", "Papaya leaf disease (Kaggle)", "", "", "", "", "", "no", "primary_train_test", "VERIFY slug + licence on Kaggle"],
    ["D3", "Papaya leaf (Roboflow Universe / GitHub)", "", "", "", "", "", "no", "cross_source_test", "VERIFY export licence"],
    ["D4", "PlantVillage", "", "", "54306", "38 (no papaya)", "", "no", "ood_control", "no papaya class — OOD probe only"],
    ["D5", "PlantDoc (field-condition)", "https://github.com/pratikkayal/PlantDoc-Dataset", "", "", "multi-crop field", "", "no", "ood_background_realism", "VERIFY"],
]

HEADER = ["dataset_id", "dataset_name", "url", "doi", "image_count", "classes",
          "licence", "download_tested", "role", "notes"]


def main():
    out = OUT_DIR / "dataset_inventory.csv"
    if out.exists():
        print(f"{out} already exists — not overwriting. Edit it directly.")
        return
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(TEMPLATE_ROWS)
    print(f"wrote template {out}")
    print("\nNEXT: open it, add every dataset you found, verify DOI + licence + count")
    print("at the primary source. Acceptable licences: CC BY, CC BY-SA, CC0, Apache-2.0, MIT.")
    print("NC (non-commercial) => flag as 'problematic'. No licence / no DOI => role='EXCLUDE'.")


if __name__ == "__main__":
    main()
