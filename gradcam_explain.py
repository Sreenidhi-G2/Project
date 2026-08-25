"""
VeriFace - Phase 4, Day 8
Explainability: given a face image, produces a Grad-CAM heatmap showing
WHICH facial region drove the fake/real decision, plus a simple
natural-language reason string. This turns a bare score into something a
fraud analyst could actually act on, rather than a black-box number.

Usage:
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m pip install grad-cam
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" gradcam_explain.py "path\\to\\face_image.jpg"
"""

import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import get_eval_transforms
from model import DeepfakeDetector


class BinaryLogitTarget:
    """Custom Grad-CAM target for a binary classifier with a single scalar
    logit output (our model.py squeezes to shape (batch,), not the standard
    (batch, num_classes) that pytorch-grad-cam's default targets=None mode
    expects). This just backprops from the raw logit directly - higher
    logit = more 'fake', so gradients highlight regions that push the
    prediction toward fake."""
    def __call__(self, model_output):
        return model_output

# ============ CONFIG ============
CHECKPOINT_PATH = r"C:\Data Razorpay\ml\checkpoints\best_model.pt"
IMAGE_SIZE = 224
OUTPUT_DIR = r"C:\Data Razorpay\ml\gradcam_outputs"
# ==================================


def get_region_name(cam_map):
    """Very simple heuristic: divide the 224x224 heatmap into 5 rough face
    regions (upper/forehead, eyes, nose/cheeks, mouth, jaw/lower) and report
    whichever region has the highest average activation. This is NOT precise
    facial landmark localization - it's a coarse, cheap approximation good
    enough for a one-line "reason" string, which is the point: interpretable
    at a glance, not clinically precise."""
    h, w = cam_map.shape
    regions = {
        "forehead/upper-face": cam_map[0:int(h * 0.25), :],
        "eye region": cam_map[int(h * 0.25):int(h * 0.45), :],
        "nose/cheek region": cam_map[int(h * 0.45):int(h * 0.65), :],
        "mouth region": cam_map[int(h * 0.65):int(h * 0.85), :],
        "jaw/chin region": cam_map[int(h * 0.85):h, :],
    }
    scores = {name: region.mean() for name, region in regions.items()}
    top_region = max(scores, key=scores.get)
    return top_region, scores


def main():
    if len(sys.argv) < 2:
        print("Usage: python gradcam_explain.py <path_to_face_image>")
        return

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = DeepfakeDetector(freeze_backbone_layers=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Read and preprocess image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"ERROR: could not read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    transform = get_eval_transforms(IMAGE_SIZE)
    augmented = transform(image=image_resized)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    # Get prediction first
    with torch.no_grad():
        logit = model(input_tensor)
        prob = torch.sigmoid(logit).cpu().item()
    verdict = "FAKE" if prob >= 0.5 else "REAL"

    # Grad-CAM: target the last conv layer of the EfficientNet backbone
    # (conv_head is the standard final feature-expansion conv in timm's
    # EfficientNet implementations, right before global pooling)
    target_layer = model.backbone.conv_head
    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor, targets=[BinaryLogitTarget()])[0]

    # Normalize the display image to [0,1] float for overlay
    display_image = image_resized.astype(np.float32) / 255.0
    cam_overlay = show_cam_on_image(display_image, grayscale_cam, use_rgb=True)

    # Region-based reason string
    top_region, region_scores = get_region_name(grayscale_cam)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_gradcam.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

    print("\n" + "=" * 50)
    print("EXPLAINABILITY RESULT")
    print("=" * 50)
    print(f"Prediction: {verdict} (fake probability: {prob:.4f})")
    print(f"Most activated region: {top_region}")
    print(f"Region activation scores:")
    for name, score in sorted(region_scores.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score:.4f}")

    if verdict == "FAKE":
        reason = (f"Flagged as likely manipulated - visual inconsistency "
                  f"concentrated around the {top_region}.")
    else:
        reason = (f"Classified as likely authentic - no strong localized "
                  f"inconsistency detected (highest, still modest, "
                  f"activation near the {top_region}).")
    print(f"\nReason string: \"{reason}\"")
    print(f"\nHeatmap overlay saved to: {output_path}")


if __name__ == "__main__":
    main()