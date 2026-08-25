"""
VeriFace - Diagnostic script
Breaks down validation performance BY MANIPULATION METHOD, not just
aggregate. This tests whether the near-chance overall val AUROC is being
dragged down by inherently harder methods (Face2Face, NeuralTextures are
known in the literature to be much subtler than Deepfakes/FaceSwap) versus
a genuine bug affecting everything equally.

Usage:
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" diagnose_by_category.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import FaceCropDataset, get_eval_transforms, LABEL_MAP
from model import DeepfakeDetector

# ============ CONFIG - match your train.py paths ============
CSV_PATH = r"C:\Data Razorpay\data\processed\labels.csv"
FACES_DIR = r"C:\Data Razorpay\data\processed\faces"
CHECKPOINT_PATH = r"C:\Data Razorpay\ml\checkpoints\best_model.pt"
IMAGE_SIZE = 224
BATCH_SIZE = 32
# ================================================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeepfakeDetector(freeze_backbone_layers=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val AUROC at save time: {checkpoint['val_auroc']:.4f}\n")

    # Full val dataframe (need 'category' column, which the Dataset class
    # doesn't expose directly, so read it ourselves)
    df = pd.read_csv(CSV_PATH)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    faces_dir = Path(FACES_DIR)
    transform = get_eval_transforms(IMAGE_SIZE)

    import cv2

    def predict_batch(rows):
        probs = []
        for _, row in rows.iterrows():
            img_path = faces_dir / row["filename"]
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = transform(image=image)
            tensor = augmented["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(tensor)
                prob = torch.sigmoid(logit).cpu().item()
            probs.append(prob)
        return probs

    print("Computing predictions on full val set (this may take a minute)...")
    val_df["pred_prob"] = predict_batch(val_df)
    val_df["true_label"] = val_df["label"].map(LABEL_MAP)

    # Overall AUROC (sanity check - should roughly match training log)
    overall_auroc = roc_auc_score(val_df["true_label"], val_df["pred_prob"])
    print(f"\nOverall val AUROC: {overall_auroc:.4f}\n")

    # Per-method breakdown: real vs each fake method individually
    real_df = val_df[val_df["label"] == "real"]
    print(f"{'Method':<20} {'N (real+fake)':<15} {'AUROC':<10}")
    print("-" * 45)

    methods = [c for c in val_df["category"].unique() if c != "original"]
    results = {}
    for method in sorted(methods):
        method_df = val_df[val_df["category"] == method]
        combined = pd.concat([real_df, method_df])
        if combined["true_label"].nunique() < 2:
            print(f"{method:<20} skipped (only one class present)")
            continue
        method_auroc = roc_auc_score(combined["true_label"], combined["pred_prob"])
        results[method] = method_auroc
        print(f"{method:<20} {len(combined):<15} {method_auroc:.4f}")

    print("\n--- Interpretation ---")
    spread = max(results.values()) - min(results.values())
    if spread > 0.15:
        print(f"Large spread ({spread:.3f}) across methods detected.")
        print("This suggests method difficulty (not a bug) is driving the "
              "low aggregate AUROC - some methods are genuinely much "
              "subtler (Face2Face/NeuralTextures are known to be harder "
              "than Deepfakes/FaceSwap in the literature). This is a real, "
              "reportable finding for your project, not a broken pipeline.")
    else:
        print(f"Small spread ({spread:.3f}) across methods - performance is "
              "uniformly weak across ALL manipulation types.")
        print("This points AWAY from method-difficulty as the explanation "
              "and suggests a pipeline/data issue instead (e.g. label "
              "mapping, face-crop quality inconsistency between train/val, "
              "or an identity-leakage/split problem). Next step: manually "
              "inspect 10-15 val face crops per method side by side with "
              "their labels to check for anything visually off.")


if __name__ == "__main__":
    main()