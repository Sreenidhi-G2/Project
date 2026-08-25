"""
VeriFace - Phase 2 extension
Covers a DIFFERENT fraud vector than the main face-swap/reenactment model:
a user submitting a fully AI-generated (diffusion-model) selfie/photo for
KYC, e.g. generated via Gemini, GPT image generation, Midjourney, DALL-E,
or Stable Diffusion. This is architecturally distinct from face-swap
deepfakes (FaceForensics++/Celeb-DF) - diffusion models leave different
generative fingerprints than face-swap/reenactment artifacts, so the main
model is NOT expected to catch this reliably. This script preps a second,
dedicated dataset for a second specialist detector covering this vector.

Uses bitmind's paired FFHQ (real) + SDXL-generated (fake) faces dataset on
Hugging Face - pre-cropped/aligned faces, no manual download/access-request
needed.

STEP 1 (run this first, does nothing but print the schema):
  Set INSPECT_ONLY = True below, run once, and paste the printed output
  back so we confirm field names before extracting the full dataset.

Usage:
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m pip install datasets
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" prepare_ai_generated_faces.py
"""

import os
import csv
import hashlib
from pathlib import Path
from datasets import load_dataset

# ============ CONFIG ============
DATASET_NAME_FAKE = "bitmind/ffhq-256___stable-diffusion-xl-base-1.0_training_faces"
DATASET_NAME_REAL = "bitmind/ffhq-256"
OUTPUT_DIR = r"C:\Data Razorpay\data\ai_faces_processed"
MAX_IMAGES_PER_CLASS = 2000  # 2000 real + 2000 fake = 4000 total
TRAIN_SPLIT, VAL_SPLIT = 0.7, 0.15
INSPECT_ONLY = False  # schema confirmed - ready to extract
# ==================================


def assign_split(item_id):
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16)
    r = (h % 10000) / 10000.0
    if r < TRAIN_SPLIT:
        return "train"
    elif r < TRAIN_SPLIT + VAL_SPLIT:
        return "val"
    else:
        return "test"


def extract_class(dataset_name, config, label, output_faces_dir, max_images,
                   rows_accumulator):
    print(f"\nLoading {label} images from: {dataset_name}"
          f"{f' (config: {config})' if config else ''} ...")
    ds = load_dataset(dataset_name, config) if config else load_dataset(dataset_name)
    split_name = list(ds.keys())[0]
    data = ds[split_name]

    n = min(max_images, len(data))
    print(f"  Using {n} of {len(data)} available images.")

    for idx in range(n):
        item = data[idx]
        image = item["image"]
        item_id = f"{label}_{idx}"
        filename = f"aiface_{item_id}.jpg"
        image.convert("RGB").resize((224, 224)).save(output_faces_dir / filename)

        rows_accumulator.append({
            "filename": filename,
            "video_id": item_id,
            "category": "SDXL" if label == "fake" else "original",
            "label": label,
            "split": assign_split(item_id),
        })

        if (idx + 1) % 500 == 0:
            print(f"    {idx + 1}/{n} done")


def main():
    output_dir = Path(OUTPUT_DIR)
    faces_dir = output_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    extract_class(DATASET_NAME_REAL, None, "real", faces_dir,
                  MAX_IMAGES_PER_CLASS, rows)
    extract_class(DATASET_NAME_FAKE, "base_transforms", "fake", faces_dir,
                  MAX_IMAGES_PER_CLASS, rows)

    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "video_id", "category", "label", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Saved {len(rows)} images to {faces_dir}")
    print(f"Labels CSV: {csv_path}")
    from collections import Counter
    print(f"Label breakdown: {dict(Counter(r['label'] for r in rows))}")
    print(f"Split breakdown: {dict(Counter(r['split'] for r in rows))}")
    print("\nThis CSV/faces folder uses the SAME schema as your FF++ "
          "pipeline, so you can reuse train.py/evaluate.py unchanged - "
          "just point CSV_PATH/FACES_DIR/CHECKPOINT_DIR at this new data "
          "and a new checkpoint filename for this second, specialist model.")


if __name__ == "__main__":
    main()