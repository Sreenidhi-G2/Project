"""
VeriFace - Phase 2, Day 5 (part 1)
Evaluates the trained model on the held-out TEST split - this is the
number you report as your baseline, since val was used to pick the best
checkpoint (val is "seen" in that sense) while test is fully untouched.

Usage:
  python evaluate.py
"""

import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from dataset import get_eval_transforms, LABEL_MAP
from model import DeepfakeDetector

# ============ CONFIG ============
CSV_PATH = r"C:\Data Razorpay\data\ai_faces_processed\labels.csv"
FACES_DIR = r"C:\Data Razorpay\data\ai_faces_processed\faces"
CHECKPOINT_PATH = r"C:\Data Razorpay\ml\checkpoints_ai_v2\best_model.pt"
IMAGE_SIZE = 224
REPORT_PATH = r"C:\Data Razorpay\ml\checkpoints_ai_v2\sdxl_crosscheck_report.md"
# ==================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeepfakeDetector(freeze_backbone_layers=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val AUROC at save time: {checkpoint['val_auroc']:.4f}\n")

    df = pd.read_csv(CSV_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Test set: {len(test_df)} images "
          f"({(test_df['label'] == 'real').sum()} real, "
          f"{(test_df['label'] == 'fake').sum()} fake)\n")

    faces_dir = Path(FACES_DIR)
    transform = get_eval_transforms(IMAGE_SIZE)

    probs, labels = [], []
    print("Running inference on test set...")
    for idx, row in test_df.iterrows():
        img_path = faces_dir / row["filename"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = transform(image=image)
        tensor = augmented["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor)
            prob = torch.sigmoid(logit).cpu().item()
        probs.append(prob)
        labels.append(LABEL_MAP[row["label"]])

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(test_df)} done")

    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= 0.5).astype(int)

    test_auroc = roc_auc_score(labels, probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=["real", "fake"])

    print("\n" + "=" * 50)
    print("HELD-OUT TEST SET RESULTS (same-dataset baseline)")
    print("=" * 50)
    print(f"AUROC:     {test_auroc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred, order=[real, fake]):")
    print(cm)
    print(f"\n{report}")

    # Save a markdown snippet you can paste straight into your README
    md = f"""# VeriFace - Same-Dataset Baseline (FaceForensics++ test split)

Checkpoint: epoch {checkpoint['epoch']}, val AUROC {checkpoint['val_auroc']:.4f}

| Metric | Value |
|---|---|
| Test AUROC | {test_auroc:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1 | {f1:.4f} |
| Test set size | {len(test_df)} images ({(test_df['label'] == 'real').sum()} real, {(test_df['label'] == 'fake').sum()} fake) |

Confusion matrix (rows=true, cols=pred, order=[real, fake]):
```
{cm}
```

Note: this is a same-dataset (FaceForensics++) held-out test result. Cross-dataset
generalization (evaluating on Celeb-DF/DFDC, which the model never saw during
training) is reported separately, and is expected to be lower - see
cross_dataset_eval.py results.
"""
    with open(REPORT_PATH, "w") as f:
        f.write(md)
    print(f"\nMarkdown report saved to: {REPORT_PATH}") 
    print("Copy this into your README/metrics_report.md.")


if __name__ == "__main__":
    main()