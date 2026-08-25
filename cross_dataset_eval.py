"""
VeriFace - Phase 2, Day 5 (part 2)
Cross-dataset generalization test: evaluates the FaceForensics++-trained
model on Celeb-DF, a dataset it has NEVER seen during training. This is
the credibility piece - same-dataset test AUROC alone is not a reliable
signal of real-world performance, since deepfake detectors are notorious
for overfitting to dataset-specific compression artifacts.

Also includes a threshold sweep: for a fraud-prevention use case, missing
a real deepfake (false negative) is typically far costlier than flagging
a real user for manual review (false positive). The default 0.5 threshold
optimizes for balanced accuracy, not this asymmetric cost - this sweep
shows you the recall/precision tradeoff at different thresholds so you
can pick (and justify) an operating point suited to the KYC use case.

BEFORE RUNNING: you need face crops extracted from Celeb-DF videos in the
same format as your FF++ pipeline. Point CELEBDF_FACES_DIR at wherever you
ran a similar frame-extraction + face-crop pass on the Celeb-DF videos,
with a labels CSV in the same format (filename, label) where label is
"real" or "fake". If you haven't extracted Celeb-DF faces yet, reuse
extract_faces.py from Phase 1 - just point BASE_DIR/CATEGORIES at the
Celeb-DF folder structure (Celeb-real -> real, Celeb-synthesis -> fake).

Usage:
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" cross_dataset_eval.py
"""

import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

from dataset import get_eval_transforms, LABEL_MAP
from model import DeepfakeDetector

# ============ CONFIG ============
CHECKPOINT_PATH = r"C:\Data Razorpay\ml\checkpoints\best_model.pt"
CELEBDF_LABELS_CSV = r"C:\Data Razorpay\data\celebdf_processed\labels.csv"
CELEBDF_FACES_DIR = r"C:\Data Razorpay\data\celebdf_processed\faces"
IMAGE_SIZE = 224
REPORT_PATH = r"C:\Data Razorpay\ml\checkpoints\cross_dataset_report.md"
THRESHOLDS_TO_TEST = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# ==================================


def evaluate_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)
    # false negative rate = missed fakes / total actual fakes - the costly
    # error for a fraud-prevention use case
    fn = cm[1][0] if cm.shape == (2, 2) else 0
    total_fake = (labels == 1).sum()
    fnr = fn / total_fake if total_fake > 0 else 0
    return precision, recall, f1, fnr, cm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeepfakeDetector(freeze_backbone_layers=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(trained on FaceForensics++, val AUROC {checkpoint['val_auroc']:.4f})\n")

    if not Path(CELEBDF_LABELS_CSV).exists():
        print(f"ERROR: {CELEBDF_LABELS_CSV} not found.")
        print("You need to extract face crops from Celeb-DF first, in the "
              "same format as Phase 1. Reuse extract_faces.py, pointing "
              "BASE_DIR at your Celeb-DF download and CATEGORIES at:")
        print('  "real": (r"Celeb-real", "real")')
        print('  "fake": (r"Celeb-synthesis", "fake")')
        print("Then set OUTPUT_DIR to a new folder like "
              r'"C:\Data Razorpay\data\celebdf_processed" so it does not '
              "overwrite your FF++ processed data.")
        return

    df = pd.read_csv(CELEBDF_LABELS_CSV)
    print(f"Celeb-DF eval set: {len(df)} images "
          f"({(df['label'] == 'real').sum()} real, "
          f"{(df['label'] == 'fake').sum()} fake)\n")

    faces_dir = Path(CELEBDF_FACES_DIR)
    transform = get_eval_transforms(IMAGE_SIZE)

    probs, labels = [], []
    print("Running inference on Celeb-DF...")
    for idx, row in df.iterrows():
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
            print(f"  {idx + 1}/{len(df)} done")

    probs = np.array(probs)
    labels = np.array(labels)

    cross_auroc = roc_auc_score(labels, probs)
    print("\n" + "=" * 55)
    print("CROSS-DATASET GENERALIZATION RESULT (Celeb-DF)")
    print("=" * 55)
    print(f"Cross-dataset AUROC: {cross_auroc:.4f}")
    print(f"(For reference: same-dataset FF++ test AUROC was reported "
          f"separately by evaluate.py - compare the two.)\n")

    print("--- Threshold sweep (precision/recall/F1/false-negative-rate) ---")
    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}{'FN Rate':<10}")
    sweep_rows = []
    for t in THRESHOLDS_TO_TEST:
        p, r, f1, fnr, cm = evaluate_at_threshold(labels, probs, t)
        print(f"{t:<12}{p:<12.4f}{r:<12.4f}{f1:<10.4f}{fnr:<10.4f}")
        sweep_rows.append((t, p, r, f1, fnr))

    print("\nFN Rate = fraction of real deepfakes the model missed (labeled "
          "them 'real'). For a KYC fraud filter, this is usually the costlier "
          "error than a false positive (flagging a genuine user for review). "
          "Consider operating at a LOWER threshold than 0.5 if you want to "
          "prioritize catching fraud over minimizing review-queue volume - "
          "explicitly justify whichever threshold you pick in your README.")

    # Markdown report
    sweep_table = "\n".join(
        f"| {t} | {p:.4f} | {r:.4f} | {f1:.4f} | {fnr:.4f} |"
        for t, p, r, f1, fnr in sweep_rows
    )
    md = f"""# VeriFace - Cross-Dataset Generalization (trained on FF++, tested on Celeb-DF)

Checkpoint: epoch {checkpoint['epoch']} (FF++ val AUROC {checkpoint['val_auroc']:.4f})

**Cross-dataset AUROC: {cross_auroc:.4f}**

This model was trained entirely on FaceForensics++ and evaluated here on
Celeb-DF, a dataset it never saw during training. A drop relative to the
same-dataset FF++ test AUROC is expected and well-documented in the
deepfake-detection literature - it reflects the model partially relying on
FF++-specific compression/artifact signatures that don't fully transfer to
Celeb-DF's different generation pipeline and video characteristics.

## Threshold sweep (operating-point tradeoff for a fraud-prevention use case)

| Threshold | Precision | Recall | F1 | False-Negative Rate |
|---|---|---|---|---|
{sweep_table}

False-Negative Rate = fraction of actual deepfakes the model missed. In a
KYC/fraud-prevention deployment, a missed deepfake (false negative) is
typically more costly than a false positive (a genuine user flagged for
manual review), so the operating threshold should be chosen with that
asymmetry in mind rather than defaulting to 0.5.
"""
    with open(REPORT_PATH, "w") as f:
        f.write(md)
    print(f"\nMarkdown report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()