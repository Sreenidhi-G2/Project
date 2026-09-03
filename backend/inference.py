"""
VeriFace Backend - inference.py
Loads both specialist models ONCE at import time (not per-request), and
provides shared scoring functions for both the image and video endpoints.
Reuses the same logic already validated in video_inference_ensemble.py and
gradcam_explain.py.
"""

import io
import cv2
import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

import sys
sys.path.append(r"C:\Data Razorpay\ml\training")
from dataset import get_eval_transforms
from model import DeepfakeDetector

# ============ CONFIG ============
FACESWAP_CHECKPOINT = r"C:\Data Razorpay\ml\checkpoints\best_model.pt"
AI_GENERATED_CHECKPOINT = r"C:\Data Razorpay\ml\checkpoints_ai_v2\best_model.pt"
IMAGE_SIZE = 224
FRAMES_PER_SECOND = 2
MAX_FRAMES = 30
GRADCAM_OUTPUT_DIR = Path(r"C:\Data Razorpay\backend\gradcam_outputs")

FACESWAP_REVIEW_LOW, FACESWAP_REVIEW_HIGH, FACESWAP_MAX_ESCALATION = 0.3, 0.6, 0.8
AI_GEN_REVIEW_LOW, AI_GEN_REVIEW_HIGH, AI_GEN_MAX_ESCALATION = 0.35, 0.65, 0.85
SEVERITY_ORDER = {"REAL": 0, "REVIEW": 1, "FAKE": 2}
# ==================================


class BinaryLogitTarget:
    def __call__(self, model_output):
        return model_output


class ModelBundle:
    """Loaded once at app startup and reused across all requests - loading
    a ~20M-parameter model per-request would make every API call slow and
    is unnecessary."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[inference] Using device: {self.device}")

        self.faceswap_model = self._load_model(FACESWAP_CHECKPOINT)
        self.ai_gen_model = self._load_model(AI_GENERATED_CHECKPOINT)
        self.mtcnn = MTCNN(keep_all=False, device=self.device, post_process=False)
        self.transform = get_eval_transforms(IMAGE_SIZE)

        # Grad-CAM for BOTH specialists - whichever one drives a verdict
        # should be able to produce its own supporting heatmap, not just
        # the face-swap model
        self.cam_faceswap = GradCAM(
            model=self.faceswap_model,
            target_layers=[self.faceswap_model.backbone.conv_head],
        )
        self.cam_ai_generated = GradCAM(
            model=self.ai_gen_model,
            target_layers=[self.ai_gen_model.backbone.conv_head],
        )

        GRADCAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print("[inference] Both specialist models loaded and ready.")

    def _load_model(self, checkpoint_path):
        model = DeepfakeDetector(freeze_backbone_layers=False).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def detect_and_crop_face(self, rgb_frame):
        """Returns a 224x224 RGB face crop, or None if no face detected."""
        try:
            boxes, probs = self.mtcnn.detect(rgb_frame)
        except Exception:
            return None
        if boxes is None or len(boxes) == 0:
            return None
        best_idx = probs.argmax()
        x1, y1, x2, y2 = [int(max(0, v)) for v in boxes[best_idx]]
        face_crop = rgb_frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        return cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE))

    def score_face_crop(self, face_rgb_224):
        """Runs both specialist models on one pre-cropped 224x224 RGB face.
        Returns (faceswap_prob, ai_gen_prob)."""
        augmented = self.transform(image=face_rgb_224)
        tensor = augmented["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            fs_prob = torch.sigmoid(self.faceswap_model(tensor)).cpu().item()
            ai_prob = torch.sigmoid(self.ai_gen_model(tensor)).cpu().item()
        return fs_prob, ai_prob

    def gradcam_for_face_crop(self, face_rgb_224, request_id, specialist="faceswap"):
        """Generates a Grad-CAM heatmap overlay for a face crop, saves it,
        returns the output file path and the most-activated region name.
        specialist: "faceswap" or "ai_generated" - picks which model's CAM
        to use, since either specialist can be the one that drove a verdict."""
        cam_obj = self.cam_faceswap if specialist == "faceswap" else self.cam_ai_generated

        augmented = self.transform(image=face_rgb_224)
        tensor = augmented["image"].unsqueeze(0).to(self.device)
        grayscale_cam = cam_obj(input_tensor=tensor, targets=[BinaryLogitTarget()])[0]

        display_image = face_rgb_224.astype(np.float32) / 255.0
        overlay = show_cam_on_image(display_image, grayscale_cam, use_rgb=True)

        output_path = GRADCAM_OUTPUT_DIR / f"{request_id}_{specialist}_gradcam.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        region_name = self._most_activated_region(grayscale_cam)
        return str(output_path), region_name

    @staticmethod
    def _most_activated_region(cam_map):
        h, w = cam_map.shape
        regions = {
            "forehead/upper-face": cam_map[0:int(h * 0.25), :],
            "eye region": cam_map[int(h * 0.25):int(h * 0.45), :],
            "nose/cheek region": cam_map[int(h * 0.45):int(h * 0.65), :],
            "mouth region": cam_map[int(h * 0.65):int(h * 0.85), :],
            "jaw/chin region": cam_map[int(h * 0.85):h, :],
        }
        scores = {name: region.mean() for name, region in regions.items()}
        return max(scores, key=scores.get)

    def get_verdict(self, mean_score, max_score, review_low, review_high, max_escalation):
        if max_score >= max_escalation and mean_score < review_high:
            return "REVIEW", "escalated (high-confidence outlier frame)"
        elif mean_score < review_low:
            return "REAL", "auto-approved"
        elif mean_score >= review_high:
            return "FAKE", "auto-flagged"
        else:
            return "REVIEW", "routed to manual review"

    def combine_verdicts(self, fs_verdict, ai_verdict):
        fs_sev = SEVERITY_ORDER[fs_verdict]
        ai_sev = SEVERITY_ORDER[ai_verdict]
        if ai_sev > fs_sev:
            return ai_verdict, "AI-generated-content detector"
        elif fs_sev > ai_sev:
            return fs_verdict, "Face-swap/reenactment detector"
        else:
            return fs_verdict, "Both specialists agree"


def read_image_bytes_to_rgb(image_bytes):
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_image)


def score_image_bytes(bundle: ModelBundle, image_bytes: bytes, request_id: str):
    rgb_image = read_image_bytes_to_rgb(image_bytes)
    face_crop = bundle.detect_and_crop_face(rgb_image)

    if face_crop is None:
        return {"error": "No face detected in the uploaded image."}

    fs_prob, ai_prob = bundle.score_face_crop(face_crop)

    fs_verdict, fs_action = bundle.get_verdict(
        fs_prob, fs_prob, FACESWAP_REVIEW_LOW, FACESWAP_REVIEW_HIGH, FACESWAP_MAX_ESCALATION
    )
    ai_verdict, ai_action = bundle.get_verdict(
        ai_prob, ai_prob, AI_GEN_REVIEW_LOW, AI_GEN_REVIEW_HIGH, AI_GEN_MAX_ESCALATION
    )
    overall_verdict, driven_by = bundle.combine_verdicts(fs_verdict, ai_verdict)

    gradcam_path, region = bundle.gradcam_for_face_crop(face_crop, request_id)

    reason = (
        f"Flagged as {overall_verdict.lower()} - primary signal from "
        f"{driven_by.lower()}, with visual inconsistency concentrated "
        f"around the {region}."
        if overall_verdict != "REAL" else
        f"Classified as likely authentic - no strong concern raised by "
        f"either specialist detector."
    )

    return {
        "overall_verdict": overall_verdict,
        "driven_by": driven_by,
        "reason": reason,
        "faceswap_score": fs_prob,
        "faceswap_verdict": fs_verdict,
        "ai_generated_score": ai_prob,
        "ai_generated_verdict": ai_verdict,
        "gradcam_region": region,
        "gradcam_path": gradcam_path,
    }


def score_video_file(bundle: ModelBundle, video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(fps / FRAMES_PER_SECOND), 1)

    fs_scores, ai_scores = [], []
    frame_idx, faces_found = 0, 0

    while cap.isOpened() and faces_found < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_crop = bundle.detect_and_crop_face(rgb_frame)
            if face_crop is not None:
                fs_prob, ai_prob = bundle.score_face_crop(face_crop)
                fs_scores.append(fs_prob)
                ai_scores.append(ai_prob)
                faces_found += 1
        frame_idx += 1
    cap.release()

    if not fs_scores:
        return {"error": "No faces detected in any sampled frame."}

    fs_arr, ai_arr = np.array(fs_scores), np.array(ai_scores)

    fs_verdict, fs_action = bundle.get_verdict(
        float(fs_arr.mean()), float(fs_arr.max()),
        FACESWAP_REVIEW_LOW, FACESWAP_REVIEW_HIGH, FACESWAP_MAX_ESCALATION
    )
    ai_verdict, ai_action = bundle.get_verdict(
        float(ai_arr.mean()), float(ai_arr.max()),
        AI_GEN_REVIEW_LOW, AI_GEN_REVIEW_HIGH, AI_GEN_MAX_ESCALATION
    )
    overall_verdict, driven_by = bundle.combine_verdicts(fs_verdict, ai_verdict)

    return {
        "overall_verdict": overall_verdict,
        "driven_by": driven_by,
        "num_frames_analyzed": len(fs_scores),
        "faceswap_mean_score": float(fs_arr.mean()),
        "faceswap_max_score": float(fs_arr.max()),
        "faceswap_verdict": fs_verdict,
        "ai_generated_mean_score": float(ai_arr.mean()),
        "ai_generated_max_score": float(ai_arr.max()),
        "ai_generated_verdict": ai_verdict,
    }