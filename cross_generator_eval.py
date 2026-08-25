"""
VeriFace - AI-generated-face detector: cross-generator/pipeline sanity check

Tests the SDXL-trained specialist model on a DIFFERENT dataset (Midjourney/
DALL-E/SD mix, from an entirely different collection pipeline than the
bitmind FFHQ/SDXL pair it was trained on). This directly tests the
suspicion that the 0.9999 val AUROC reflects a dataset-pipeline shortcut
(e.g. compression/export differences between the two bitmind datasets)
rather than genuine AI-generated-face detection ability.

CAVEAT: this test dataset is general images, not necessarily face-cropped
portraits like the model's training data. A lower score here could reflect
either (a) genuine non-generalization, or (b) input-distribution mismatch
(full scenes vs face crops). Note this honestly in your README rather than
over-interpreting a single number.

Usage:
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m pip install datasets
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" cross_generator_eval.py
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from datasets import load_dataset

from dataset import get_eval_transforms, LABEL_MAP
from model import DeepfakeDetector

# ============ CONFIG ============
CHECKPOINT_PATH = r"C:\Data Razorpay\ml\checkpoints_ai_generated\best_model.pt"
TEST_DATASET_NAME = "julienlucas/midjourney-dalle-sd-dataset"
IMAGE_SIZE = 224
REPORT_PATH = r"C:\Data Razorpay\ml\checkpoints_ai_generated\cross_generator_report.md"
# ==================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeepfakeDetector(freeze_backbone_layers=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(trained on FFHQ vs SDXL, val AUROC {checkpoint['val_auroc']:.4f})\n")

    print(f"Loading {TEST_DATASET_NAME} test split ...")
    ds = load_dataset(
        "parquet",
        data_files={
            "test": f"hf://datasets/{TEST_DATASET_NAME}/data/test-00000-of-00001.parquet"
        },
    )
    test_data = ds["test"]
    print(f"Test set: {len(test_data)} images\n")

    # The 'label' field turned out to be an integer-coded ClassLabel, not a
    # plain "real"/"fake" string as the dataset card suggested - read the
    # actual int->name mapping from the schema rather than guessing.
    label_feature = test_data.features["label"]
    if hasattr(label_feature, "int2str"):
        int_to_name = {i: label_feature.int2str(i) for i in range(label_feature.num_classes)}
    else:
        int_to_name = None
    print(f"Label schema: {label_feature}")
    if int_to_name:
        print(f"Integer -> name mapping: {int_to_name}\n")

    def resolve_label(raw_label):
        if isinstance(raw_label, str):
            return LABEL_MAP[raw_label]
        if int_to_name is not None:
            name = int_to_name[raw_label].lower()
        else:
            # Fallback if schema doesn't expose names: print once and ask
            raise ValueError(
                f"Could not resolve label {raw_label} - schema didn't "
                "expose class names. Paste the 'Label schema' line printed "
                "above so we can hardcode the correct mapping."
            )
        return LABEL_MAP[name]

    transform = get_eval_transforms(IMAGE_SIZE)

    probs, labels = [], []
    print("Running inference...")
    for idx, item in enumerate(test_data):
        image = item["image"].convert("RGB")
        image_np = np.array(image)
        augmented = transform(image=image_np)
        tensor = augmented["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor)
            prob = torch.sigmoid(logit).cpu().item()
        probs.append(prob)
        labels.append(resolve_label(item["label"]))

        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(test_data)} done")

    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= 0.5).astype(int)

    auroc = roc_auc_score(labels, probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    print("\n" + "=" * 55)
    print("CROSS-GENERATOR/PIPELINE RESULT (Midjourney/DALL-E/SD mix)")
    print("=" * 55)
    print(f"AUROC:     {auroc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\n--- Interpretation ---")
    if auroc >= 0.80:
        print("Strong result: the model generalizes well beyond the SDXL "
              "training pipeline. The 0.9999 val AUROC appears to reflect "
              "genuine, transferable AI-generated-face detection signal, "
              "not just a dataset-pipeline shortcut.")
    elif auroc >= 0.60:
        print("Moderate drop, similar in spirit to the FF++ -> Celeb-DF "
              "gap: some genuine signal transfers, but there is a real "
              "generalization gap across generators/pipelines. Report both "
              "numbers honestly - this is a legitimate, expected finding.")
    else:
        print("Large drop toward chance level. This supports the concern "
              "that the near-perfect 0.9999 val AUROC was driven by a "
              "dataset-pipeline shortcut (compression/export artifacts "
              "distinguishing the two bitmind source datasets) rather than "
              "genuine AI-generated-face detection. Also consider: this "
              "test set is general images, not face crops, which alone "
              "could explain some of the drop - note both possibilities "
              "honestly rather than picking one to report.")

    md = f"""# VeriFace - AI-Generated-Face Detector: Cross-Generator Check

Checkpoint: epoch {checkpoint['epoch']} (trained on FFHQ-real vs SDXL-fake, val AUROC {checkpoint['val_auroc']:.4f})

Tested on: {TEST_DATASET_NAME} (Midjourney/DALL-E/Stable Diffusion mix, different collection pipeline)

| Metric | Value |
|---|---|
| AUROC | {auroc:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1 | {f1:.4f} |

**Caveat:** this test set contains general images, not necessarily face-cropped
portraits like the training data - a lower score may partly reflect input-
distribution mismatch rather than pure generalization failure. Report this
honestly alongside the number rather than over-claiming either way.
"""
    with open(REPORT_PATH, "w") as f:
        f.write(md)
    print(f"\nMarkdown report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()