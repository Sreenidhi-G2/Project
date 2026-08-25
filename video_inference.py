"""
VeriFace - Phase 3, Day 6
Video-level inference: samples frames from a video, runs each through the
trained face-swap detector, and aggregates per-frame scores into a single
video-level verdict. This matches the actual liveness-check/video-KYC use
case, where Razorpay would receive a full video clip, not a single frame.

Aggregation strategy (deliberately simple as a baseline, per the plan):
  - mean of per-frame fake-probabilities
  - max of per-frame fake-probabilities (catches brief manipulation bursts
    that mean-aggregation could dilute)
  - per-frame variance (a video that's PART manipulated should show
    spikier per-frame scores than one that's uniformly real or fake)

Usage:
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" video_inference.py "path\\to\\video.mp4"
"""

import sys
import cv2
import importlib
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN

torch = importlib.import_module("torch")

from dataset import get_eval_transforms
from model import DeepfakeDetector

# ============ CONFIG ============
CHECKPOINT_PATH = r"C:\Data Razorpay\ml\checkpoints\best_model.pt"
IMAGE_SIZE = 224
FRAMES_PER_SECOND = 2       # denser than training extraction (1 fps) since
                             # inference-time we want good temporal coverage
MAX_FRAMES = 30              # cap so a long video doesn't take forever
DECISION_THRESHOLD = 0.3     # kept for backward reference / single-cutoff mode
REVIEW_LOW_THRESHOLD = 0.3   # below this: auto-approve as REAL
REVIEW_HIGH_THRESHOLD = 0.6  # above this: auto-flag as FAKE
                              # between the two: route to manual review
                              # rather than force a binary call - reflects
                              # that a single cutoff either lets fraud
                              # through (threshold too high) or over-flags
                              # genuine users in unfamiliar conditions
                              # (threshold too low, as seen on webcam test
                              # footage - see README known limitations)
MAX_SCORE_ESCALATION_THRESHOLD = 0.8
                              # a single frame scoring this high is itself
                              # suspicious, even if diluted by other
                              # ambiguous frames in the mean - added after
                              # a fully-synthetic (Gemini-generated) test
                              # video scored mean=0.28 (auto-approved) but
                              # max=0.92 (a fake-modality gap the mean alone
                              # missed - see README known limitations)
# ==================================


def get_verdict(mean_score, max_score):
    if max_score >= MAX_SCORE_ESCALATION_THRESHOLD and mean_score < REVIEW_HIGH_THRESHOLD:
        return "REVIEW", "escalated to manual review (high-confidence outlier frame)"
    elif mean_score < REVIEW_LOW_THRESHOLD:
        return "REAL", "auto-approved"
    elif mean_score >= REVIEW_HIGH_THRESHOLD:
        return "FAKE", "auto-flagged"
    else:
        return "REVIEW", "routed to manual review"


def load_model(device):
    model = DeepfakeDetector(freeze_backbone_layers=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def score_video(video_path, model, mtcnn, transform, device):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(fps / FRAMES_PER_SECOND), 1)

    frame_scores = []
    frame_indices = []
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
                        logit = model(tensor)
                        prob = torch.sigmoid(logit).cpu().item()

                    frame_scores.append(prob)
                    frame_indices.append(frame_idx)
                    faces_found += 1

        frame_idx += 1

    cap.release()

    if not frame_scores:
        return {
            "error": "No faces detected in any sampled frame. Video may be "
                     "too low quality, face may be too small/occluded, or "
                     "video may not contain a clear frontal face.",
        }

    frame_scores = np.array(frame_scores)
    mean_score = float(frame_scores.mean())
    max_score = float(frame_scores.max())
    variance = float(frame_scores.var())

    verdict, action = get_verdict(mean_score, max_score)

    return {
        "video_path": str(video_path),
        "num_frames_analyzed": len(frame_scores),
        "frame_indices": frame_indices,
        "per_frame_scores": frame_scores.tolist(),
        "mean_score": mean_score,
        "max_score": max_score,
        "score_variance": variance,
        "review_low_threshold": REVIEW_LOW_THRESHOLD,
        "review_high_threshold": REVIEW_HIGH_THRESHOLD,
        "max_score_escalation_threshold": MAX_SCORE_ESCALATION_THRESHOLD,
        "verdict": verdict,
        "action": action,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python video_inference.py <path_to_video.mp4>")
        return

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_model(device)
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
    transform = get_eval_transforms(IMAGE_SIZE)

    print(f"Analyzing video: {video_path}")
    result = score_video(video_path, model, mtcnn, transform, device)

    if "error" in result:
        print(f"\nERROR: {result['error']}")
        return

    print("\n" + "=" * 50)
    print("VIDEO-LEVEL RESULT")
    print("=" * 50)
    print(f"Frames analyzed:  {result['num_frames_analyzed']}")
    print(f"Mean fake score:  {result['mean_score']:.4f}")
    print(f"Max fake score:   {result['max_score']:.4f}")
    print(f"Score variance:   {result['score_variance']:.4f}")
    print(f"Review band:      [{result['review_low_threshold']}, "
          f"{result['review_high_threshold']})")
    print(f"VERDICT: {result['verdict']}  ({result['action']})")

    if result["score_variance"] > 0.05:
        print("\nNote: high per-frame variance detected. This can indicate "
              "a PARTIALLY manipulated video (manipulation only in some "
              "frames/segments) rather than uniformly real or fake - worth "
              "flagging for manual review even if the mean score alone "
              "wouldn't trigger a FAKE verdict.")


if __name__ == "__main__":
    main()
