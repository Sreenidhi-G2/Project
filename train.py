"""
VeriFace - Phase 2, Days 3-4
Training script for the image deepfake classifier.

Usage (Windows, from ml/training/ directory):
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" train.py

Tracks AUROC (not just accuracy - accuracy is misleading on imbalanced
data) on the val split each epoch, saves the best checkpoint, and stops
early if val AUROC doesn't improve for PATIENCE epochs.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from pathlib import Path
import time
import json

from dataset import FaceCropDataset
from model import DeepfakeDetector

# ============ CONFIG - edit paths if yours differ ============

CSV_PATH = r"C:\Data Razorpay\data\ai_faces_v2_processed\labels.csv"
FACES_DIR = r"C:\Data Razorpay\data\ai_faces_v2_processed\faces"
CHECKPOINT_DIR = r"C:\Data Razorpay\ml\checkpoints_ai_v2"

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 5e-5        # lowered from 1e-4 - was letting the model move too fast per step
UNFREEZE_AFTER_EPOCH = 8    # raised from 3 - ~500 source videos is not enough to safely
                             # unfreeze a 19M-param backbone early; give the head much
                             # longer to stabilize on frozen features first
PATIENCE = 5                # slightly more patience since LR is now lower
NUM_WORKERS = 4             # lower to 0 if you hit multiprocessing issues on Windows
IMAGE_SIZE = 224
WEIGHT_DECAY = 1e-4          # raised from 1e-5 - extra regularization pressure
# ================================================================


def compute_class_weights(dataset):
    """Datasets here are often imbalanced (more manipulated than real
    videos, since 4 fake methods vs 1 real source). Weight the loss
    inversely to class frequency so the model doesn't just learn to
    always predict 'fake'."""
    labels = dataset.df["label"].map({"real": 0, "fake": 1}).values
    n_real = (labels == 0).sum()
    n_fake = (labels == 1).sum()
    total = n_real + n_fake
    # pos_weight for BCEWithLogitsLoss: weight applied to the positive (fake) class
    pos_weight = n_real / max(n_fake, 1)
    print(f"Class balance -> real: {n_real}, fake: {n_fake}, "
          f"pos_weight: {pos_weight:.3f}")
    return torch.tensor(pos_weight, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auroc = roc_auc_score(all_labels, all_probs)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", zero_division=0
    )

    return {
        "loss": avg_loss, "auroc": auroc,
        "precision": precision, "recall": recall, "f1": f1,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected. Training will be slow - "
              "consider reducing NUM_EPOCHS or BATCH_SIZE.")

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = FaceCropDataset(CSV_PATH, FACES_DIR, "train", IMAGE_SIZE)
    val_ds = FaceCropDataset(CSV_PATH, FACES_DIR, "val", IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = DeepfakeDetector(freeze_backbone_layers=True).to(device)

    pos_weight = compute_class_weights(train_ds).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val_auroc = 0.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        if epoch == UNFREEZE_AFTER_EPOCH + 1:
            print(">> Unfreezing full backbone for fine-tuning.")
            model.unfreeze_all()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE / 10, weight_decay=WEIGHT_DECAY
            )

        train_metrics = run_epoch(model, train_loader, criterion, optimizer,
                                   device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer,
                                 device, train=False)

        elapsed = time.time() - start
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} ({elapsed:.0f}s)")
        print(f"  Train - loss: {train_metrics['loss']:.4f}, "
              f"AUROC: {train_metrics['auroc']:.4f}")
        print(f"  Val   - loss: {val_metrics['loss']:.4f}, "
              f"AUROC: {val_metrics['auroc']:.4f}, "
              f"precision: {val_metrics['precision']:.4f}, "
              f"recall: {val_metrics['recall']:.4f}, "
              f"f1: {val_metrics['f1']:.4f}")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auroc": best_val_auroc,
            }, checkpoint_path)
            print(f"  -> New best val AUROC: {best_val_auroc:.4f}. "
                  f"Saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement ({epochs_without_improvement}/{PATIENCE})")

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered after epoch {epoch}.")
            break

    # Save training history for your metrics_report.md
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val AUROC: {best_val_auroc:.4f}")
    print(f"Best checkpoint: {checkpoint_dir / 'best_model.pt'}")
    print(f"Full history saved to: {history_path}")
    print("\nNext step: run evaluate.py on the test split for your "
          "baseline metrics, then cross_dataset_eval.py on Celeb-DF/DFDC "
          "for the generalization number (Phase 2, Day 5).")


if __name__ == "__main__":
    main()