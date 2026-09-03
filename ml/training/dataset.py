"""
VeriFace - Phase 2
PyTorch Dataset for the face-crop images produced by extract_faces.py.

Reads labels.csv (filename, video_id, category, label, split) and serves
(image, label) pairs for a given split, with augmentation applied only
to the training split.
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


LABEL_MAP = {"real": 0, "fake": 1}


def get_train_transforms(image_size=224):
    """Augmentations chosen deliberately to mimic real-world upload
    conditions (compression, blur, lighting) rather than generic augmentation
    - this matters because deepfake detectors are notorious for overfitting
    to dataset-specific compression artifacts rather than real manipulation
    signatures. Training with varied compression helps close that gap."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(40, 100), p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size=224):
    """No augmentation for val/test - we want a clean, honest measurement."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class FaceCropDataset(Dataset):
    def __init__(self, csv_path, faces_dir, split, image_size=224):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.faces_dir = Path(faces_dir)
        self.transform = (
            get_train_transforms(image_size) if split == "train"
            else get_eval_transforms(image_size)
        )

        if len(self.df) == 0:
            raise ValueError(
                f"No rows found for split='{split}' in {csv_path}. "
                "Check that labels.csv has a 'split' column with "
                "train/val/test values."
            )

        print(f"[{split}] {len(self.df)} images "
              f"({(self.df['label'] == 'real').sum()} real, "
              f"{(self.df['label'] == 'fake').sum()} fake)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.faces_dir / row["filename"]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = LABEL_MAP[row["label"]]

        augmented = self.transform(image=image)
        image_tensor = augmented["image"]

        return image_tensor, label