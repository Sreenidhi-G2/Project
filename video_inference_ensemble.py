"""
VeriFace - Ensemble video inference
Runs BOTH specialist detectors on every sampled frame:
  1. Face-swap/reenactment detector (trained on FaceForensics++)
  2. AI-generated-content detector (trained on Midjourney/DALL-E/SD mix)

Each specialist targets a DIFFERENT fraud vector (a real video that's been
face-swapped, vs. a fully synthetic AI-generated video/photo like Gemini
output). Neither model reliably covers the other's vector - see the
project's cross-dataset/cross-generator findings. Running both and taking
the more severe verdict is the realistic mitigation: an ensemble of
specialists, not a single generalist model, matching how real-world
fraud-detection pipelines are typically built.

Usage:
  python video_inference_ensemble.py "path\\to\\video.mp4"
"""

import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN

from dataset import get_eval_transforms
from model import DeepfakeDetector

# ============ CONFIG ============
FACESWAP_CHECKPOINT = r"C:\Data Razorpay\ml\checkpoints\best_model.pt"
AI_GENERATED_CHECKPOINT = r"C:\Data Razorpay\ml\checkpoints_ai_v2\best_model.pt"
IMAGE_SIZE = 224
FRAMES_PER_SECOND = 2
MAX_FRAMES = 30

# Face-swap specialist thresholds (from Celeb-DF threshold sweep)
FACESWAP_REVIEW_LOW = 0.3
FACESWAP_REVIEW_HIGH = 0.6
FACESWAP_MAX_ESCALATION = 0.8

# AI-generated specialist thresholds (from multi-generator test split, 0.5
# default since this model's calibration wasn't tuned to a fraud-recall
# threshold sweep like the face-swap model was - noted as future work)
AI_GEN_REVIEW_LOW = 0.35
AI_GEN_REVIEW_HIGH = 0.65
AI_GEN_MAX_ESCALATION = 0.85
# ==================================


def load_model(checkpoint_path, device):
    model = DeepfakeDetector(freeze_backbone_layers=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def get_verdict(mean_score, max_score, review_low, review_high, max_escalation):
    if max_score >= max_escalation and mean_score < review_high:
        return "REVIEW", "escalated (high-confidence outlier frame)"
    elif mean_score < review_low:
        return "REAL", "auto-approved"
    elif mean_score >= review_high:
        return "FAKE", "auto-flagged"
    else:
        return "REVIEW", "routed to manual review"


SEVERITY_ORDER = {"REAL": 0, "REVIEW": 1, "FAKE": 2}


def combine_verdicts(faceswap_result, ai_gen_result):
    """The overall verdict is whichever specialist raised the more severe
    concern - an ensemble should not let a calm signal from one specialist
    override a real concern raised by the other."""
    fs_severity = SEVERITY_ORDER[faceswap_result["verdict"]]
    ai_severity = SEVERITY_ORDER[ai_gen_result["verdict"]]

    if ai_severity > fs_severity:
        winning_specialist = "AI-generated-content detector"
        overall_verdict = ai_gen_result["verdict"]
    elif fs_severity > ai_severity:
        winning_specialist = "Face-swap/reenactment detector"
        overall_verdict = faceswap_result["verdict"]
    else:
        winning_specialist = "Both specialists agree"
        overall_verdict = faceswap_result["verdict"]

    return overall_verdict, winning_specialist


def score_video_both_models(video_path, faceswap_model, ai_gen_model, mtcnn,
                             transform, device):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(fps / FRAMES_PER_SECOND), 1)

    faceswap_scores = []
    ai_gen_scores = []
    frame_idx = 0
    faces_found = 0

    while cap.isOpened() and faces_found < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                boxes, probs = mtcnn.detect(rgb_frame)
            except Exception:
                boxes = None

            if boxes is not None and len(boxes) > 0:
                best_idx = probs.argmax()
                box = boxes[best_idx]
                x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                face_crop = rgb_frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE))
                    augmented = transform(image=face_resized)
                    tensor = augmented["image"].unsqueeze(0).to(device)

                    with torch.no_grad():
                        fs_logit = faceswap_model(tensor)
                        fs_prob = torch.sigmoid(fs_logit).cpu().item()

                        ai_logit = ai_gen_model(tensor)
                        ai_prob = torch.sigmoid(ai_logit).cpu().item()

                    faceswap_scores.append(fs_prob)
                    ai_gen_scores.append(ai_prob)
                    faces_found += 1

        frame_idx += 1

    cap.release()

    if not faceswap_scores:
        return {"error": "No faces detected in any sampled frame."}

    def summarize(scores, review_low, review_high, max_escalation):
        arr = np.array(scores)
        mean_score = float(arr.mean())
        max_score = float(arr.max())
        variance = float(arr.var())
        verdict, action = get_verdict(mean_score, max_score, review_low,
                                       review_high, max_escalation)
        return {
            "mean_score": mean_score, "max_score": max_score,
            "variance": variance, "verdict": verdict, "action": action,
        }

    faceswap_result = summarize(faceswap_scores, FACESWAP_REVIEW_LOW,
                                 FACESWAP_REVIEW_HIGH, FACESWAP_MAX_ESCALATION)
    ai_gen_result = summarize(ai_gen_scores, AI_GEN_REVIEW_LOW,
                               AI_GEN_REVIEW_HIGH, AI_GEN_MAX_ESCALATION)

    overall_verdict, winning_specialist = combine_verdicts(faceswap_result, ai_gen_result)

    return {
        "num_frames_analyzed": len(faceswap_scores),
        "faceswap": faceswap_result,
        "ai_generated": ai_gen_result,
        "overall_verdict": overall_verdict,
        "driven_by": winning_specialist,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python video_inference_ensemble.py <path_to_video.mp4>")
        return

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading both specialist models...")
    faceswap_model, _ = load_model(FACESWAP_CHECKPOINT, device)
    ai_gen_model, _ = load_model(AI_GENERATED_CHECKPOINT, device)
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
    transform = get_eval_transforms(IMAGE_SIZE)

    print(f"Analyzing video: {video_path}")
    result = score_video_both_models(video_path, faceswap_model, ai_gen_model,
                                      mtcnn, transform, device)

    if "error" in result:
        print(f"\nERROR: {result['error']}")
        return

    print("\n" + "=" * 55)
    print("ENSEMBLE VIDEO RESULT")
    print("=" * 55)
    print(f"Frames analyzed: {result['num_frames_analyzed']}\n")

    fs = result["faceswap"]
    print("[Face-swap/reenactment specialist]")
    print(f"  mean={fs['mean_score']:.4f}  max={fs['max_score']:.4f}  "
          f"var={fs['variance']:.4f}  -> {fs['verdict']} ({fs['action']})")

    ai = result["ai_generated"]
    print("\n[AI-generated-content specialist]")
    print(f"  mean={ai['mean_score']:.4f}  max={ai['max_score']:.4f}  "
          f"var={ai['variance']:.4f}  -> {ai['verdict']} ({ai['action']})")

    print(f"\nOVERALL VERDICT: {result['overall_verdict']}")
    print(f"Driven by: {result['driven_by']}")


if __name__ == "__main__":
    main()