"""
VeriFace - Representative Video Frame Sampler

Purpose:
    Extract a small number of representative frames from a video
    for the GPT supervisor.

Strategy:
    1. Divide video into temporal segments.
    2. Generate candidate frames in each segment.
    3. Detect the largest face in each candidate.
    4. Score candidates using:
         - face size
         - image sharpness
         - brightness quality
    5. Select the best frame from each segment.

The sampler intentionally does NOT run either deepfake detector.
The ML models remain responsible for prediction.
This module only prepares visual evidence for the supervisor.

Example:
    frames = sample_representative_frames(
        "video.mp4",
        output_dir="supervisor_frames",
        num_frames=8
    )
"""

import cv2
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224

# Number of candidate frames examined inside each segment.
CANDIDATES_PER_SEGMENT = 5

# Minimum face size relative to frame area.
# Very tiny faces are poor evidence for an LLM.
MIN_FACE_AREA_RATIO = 0.02

# ============================================================


class RepresentativeFrameSampler:

    def __init__(self, device="cpu"):
        self.device = device

        self.mtcnn = MTCNN(
            keep_all=True,
            device=device,
            post_process=False
        )

    def _detect_best_face(self, frame_rgb):
        """
        Detect faces and return the largest face.

        Returns:
            {
                "box": (x1, y1, x2, y2),
                "area_ratio": float
            }

        or None if no face exists.
        """

        try:
            boxes, probs = self.mtcnn.detect(frame_rgb)
        except Exception:
            return None

        if boxes is None or len(boxes) == 0:
            return None

        height, width = frame_rgb.shape[:2]

        best_face = None
        best_area = 0

        for box, prob in zip(boxes, probs):

            if prob is None or prob < 0.80:
                continue

            x1, y1, x2, y2 = box

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(width, int(x2))
            y2 = min(height, int(y2))

            face_width = max(0, x2 - x1)
            face_height = max(0, y2 - y1)

            area = face_width * face_height

            if area > best_area:
                best_area = area
                best_face = (x1, y1, x2, y2)

        if best_face is None:
            return None

        area_ratio = best_area / float(width * height)

        if area_ratio < MIN_FACE_AREA_RATIO:
            return None

        return {
            "box": best_face,
            "area_ratio": area_ratio
        }

    @staticmethod
    def _sharpness(frame):
        """
        Laplacian variance is a simple measure of image sharpness.
        Higher = generally sharper.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        return float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )

    @staticmethod
    def _brightness_score(frame):
        """
        Returns a score between 0 and 1.

        Penalizes extremely dark or extremely bright frames.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        brightness = float(gray.mean())

        # Ideal brightness approximately around 128.
        distance = abs(brightness - 128.0)

        score = 1.0 - (distance / 128.0)

        return max(0.0, min(1.0, score))

    def _candidate_score(self, frame_rgb, face_info):
        """
        Score a candidate frame.

        Face size is weighted strongly because the supervisor
        needs to clearly see the face.

        Sharpness is also important.

        Brightness is a smaller factor.
        """

        face_area = face_info["area_ratio"]

        sharpness = self._sharpness(frame_rgb)

        brightness = self._brightness_score(frame_rgb)

        # Normalize sharpness approximately.
        # Values above ~1000 are already quite sharp.
        sharpness_score = min(sharpness / 1000.0, 1.0)

        # Face area:
        # 2% -> 0
        # 20%+ -> 1
        face_score = np.clip(
            (face_area - 0.02) / 0.18,
            0.0,
            1.0
        )

        # Final score.
        score = (
            0.50 * face_score +
            0.35 * sharpness_score +
            0.15 * brightness
        )

        return float(score)

    def _read_frame(self, cap, frame_index):
        """
        Seek and read a frame.
        """

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()

        if not success:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def sample(
        self,
        video_path,
        output_dir,
        num_frames=8
    ):
        """
        Extract representative frames.

        Returns:
            [
                {
                    "path": "...",
                    "frame_index": int,
                    "timestamp_seconds": float,
                    "score": float,
                    "face_area_ratio": float
                },
                ...
            ]
        """

        video_path = str(video_path)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            cap.release()
            raise ValueError("Could not determine video length.")

        duration = total_frames / fps

        # If the video is shorter than the requested number
        # of frames, don't try to produce more than necessary.
        actual_num_frames = min(
            num_frames,
            total_frames
        )

        # ----------------------------------------------------
        # Divide video into temporal segments.
        # ----------------------------------------------------

        segment_edges = np.linspace(
            0,
            total_frames - 1,
            actual_num_frames + 1
        ).astype(int)

        selected_frames = []

        # ----------------------------------------------------
        # Find best frame in each segment.
        # ----------------------------------------------------

        for segment_idx in range(actual_num_frames):

            start = segment_edges[segment_idx]
            end = segment_edges[segment_idx + 1]

            if end <= start:
                candidate_indices = [start]
            else:
                candidate_indices = np.linspace(
                    start,
                    end,
                    CANDIDATES_PER_SEGMENT
                ).astype(int)

            best_candidate = None

            for frame_index in candidate_indices:

                frame_rgb = self._read_frame(
                    cap,
                    int(frame_index)
                )

                if frame_rgb is None:
                    continue

                face_info = self._detect_best_face(
                    frame_rgb
                )

                if face_info is None:
                    continue

                score = self._candidate_score(
                    frame_rgb,
                    face_info
                )

                candidate = {
                    "frame": frame_rgb,
                    "frame_index": int(frame_index),
                    "score": score,
                    "face_area_ratio": face_info["area_ratio"]
                }

                if (
                    best_candidate is None
                    or candidate["score"] >
                    best_candidate["score"]
                ):
                    best_candidate = candidate

            # No usable frame in this segment.
            if best_candidate is None:
                continue

            # ------------------------------------------------
            # Save selected frame.
            # ------------------------------------------------

            frame_rgb = best_candidate["frame"]

            frame_bgr = cv2.cvtColor(
                frame_rgb,
                cv2.COLOR_RGB2BGR
            )

            output_path = (
                output_dir /
                f"frame_{segment_idx:02d}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                frame_bgr,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    90
                ]
            )

            timestamp = (
                best_candidate["frame_index"] / fps
            )

            selected_frames.append({
                "path": str(output_path),
                "frame_index":
                    best_candidate["frame_index"],
                "timestamp_seconds":
                    round(timestamp, 3),
                "score":
                    round(best_candidate["score"], 4),
                "face_area_ratio":
                    round(
                        best_candidate["face_area_ratio"],
                        4
                    )
            })

        cap.release()

        return selected_frames


# ============================================================
# Convenience function
# ============================================================

def sample_representative_frames(
    video_path,
    output_dir,
    num_frames=8,
    device=None
):
    """
    Simple public API for FastAPI/supervisor code.
    """

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sampler = RepresentativeFrameSampler(
        device=device
    )

    return sampler.sample(
        video_path=video_path,
        output_dir=output_dir,
        num_frames=num_frames
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract representative video frames "
                    "for VeriFace supervisor."
    )

    parser.add_argument(
        "video",
        help="Path to input video"
    )

    parser.add_argument(
        "--output",
        default="supervisor_frames",
        help="Output directory"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="Number of representative frames"
    )

    args = parser.parse_args()

    results = sample_representative_frames(
        video_path=args.video,
        output_dir=args.output,
        num_frames=args.frames
    )

    print("\n" + "=" * 60)
    print("REPRESENTATIVE FRAMES")
    print("=" * 60)

    print(f"Requested: {args.frames}")
    print(f"Selected:  {len(results)}\n")

    for item in results:
        print(
            f"Frame {item['frame_index']:6d} | "
            f"time={item['timestamp_seconds']:7.2f}s | "
            f"score={item['score']:.3f} | "
            f"face={item['face_area_ratio']:.3f} | "
            f"{item['path']}"
        )