"""
VeriFace - AI-content detector v2
Extracts the julienlucas/midjourney-dalle-sd-dataset (real vs. AI-generated,
MIXED across Midjourney, DALL-E, and Stable Diffusion in one training set)
into the standard labels.csv/faces schema. Training on a mixed-generator
set directly, rather than SDXL alone, is the fix for the cross-generator
collapse seen earlier (0.9999 same-generator AUROC -> 0.52 cross-generator).

Dataset schema (confirmed): {"image": Image, "label": ClassLabel[fake,real]}
  - train: 5,000 images (2,500 real / 2,500 fake) -> split further into
    train/val here
  - test: 1,000 images (500/500) -> kept as the held-out test split

Usage:
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" prepare_multigenerator_faces.py
"""

import csv
import hashlib
from pathlib import Path
from datasets import load_dataset

# ============ CONFIG ============
DATASET_NAME = "julienlucas/midjourney-dalle-sd-dataset"
OUTPUT_DIR = r"C:\Data Razorpay\data\ai_faces_v2_processed"
VAL_FRACTION = 0.15  # carved out of the 5,000-image train split
# ==================================


def assign_val_split(item_id, val_fraction):
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16)
    r = (h % 10000) / 10000.0
    return "val" if r < val_fraction else "train"


def main():
    output_dir = Path(OUTPUT_DIR)
    faces_dir = output_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_NAME} ...")
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"hf://datasets/{DATASET_NAME}/data/train-*.parquet",
            "test": f"hf://datasets/{DATASET_NAME}/data/test-*.parquet",
        },
    )

    # Resolve int->name label mapping from schema (same lesson learned from
    # cross_generator_eval.py - don't assume string labels)
    label_feature = ds["train"].features["label"]
    int_to_name = {i: label_feature.int2str(i) for i in range(label_feature.num_classes)}
    print(f"Label mapping: {int_to_name}\n")

    rows = []

    # --- Train split (further divided into train/val) ---
    print(f"Processing train split ({len(ds['train'])} images)...")
    for idx, item in enumerate(ds["train"]):
        image = item["image"].convert("RGB")
        raw_label = item["label"]
        label = int_to_name[raw_label] if isinstance(raw_label, int) else raw_label

        item_id = f"mgtrain_{idx}"
        filename = f"mgface_{item_id}.jpg"
        image.resize((224, 224)).save(faces_dir / filename)

        rows.append({
            "filename": filename,
            "video_id": item_id,
            "category": "multi-generator",
            "label": label,
            "split": assign_val_split(item_id, VAL_FRACTION),
        })

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(ds['train'])} done")

    # --- Test split (kept as-is, held out) ---
    print(f"\nProcessing test split ({len(ds['test'])} images)...")
    for idx, item in enumerate(ds["test"]):
        image = item["image"].convert("RGB")
        raw_label = item["label"]
        label = int_to_name[raw_label] if isinstance(raw_label, int) else raw_label

        item_id = f"mgtest_{idx}"
        filename = f"mgface_{item_id}.jpg"
        image.resize((224, 224)).save(faces_dir / filename)

        rows.append({
            "filename": filename,
            "video_id": item_id,
            "category": "multi-generator",
            "label": label,
            "split": "test",
        })

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(ds['test'])} done")

    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "video_id", "category", "label", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    print(f"\nDone. {len(rows)} images saved to {faces_dir}")
    print(f"Labels CSV: {csv_path}")
    print(f"Label breakdown: {dict(Counter(r['label'] for r in rows))}")
    print(f"Split breakdown: {dict(Counter(r['split'] for r in rows))}")
    print("\nNext: point train.py's CSV_PATH/FACES_DIR/CHECKPOINT_DIR at this "
          "new data (use a new checkpoint dir, e.g. checkpoints_ai_v2) and "
          "retrain. This model should generalize across generators much "
          "better than the SDXL-only version, since it saw Midjourney, "
          "DALL-E, and SD examples during training, not just one family.")


if __name__ == "__main__":
    main()