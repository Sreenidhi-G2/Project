"""
VeriFace - Phase 2, Day 5 (cross-dataset data prep)
Extracts face crops from the Celeb-DF videos listed in the OFFICIAL
List_of_testing_videos.txt benchmark split (518 videos: 70 YouTube-real +
108 Celeb-real + 340 Celeb-synthesis). Using the official list means your
cross-dataset AUROC is directly comparable to published Celeb-DF benchmark
numbers, rather than an arbitrary subset.

List format: "<label> <relative_path>" where 1 = real, 0 = fake, e.g.:
  1 YouTube-real/00170.mp4
  0 Celeb-synthesis/id1_id0_0007.mp4

Usage:
  "C:\\Users\\Sreendihi G\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" extract_celebdf_faces.py
"""

import os
import cv2
import csv
from pathlib import Path
from facenet_pytorch import MTCNN
import torch

# ============ CONFIG ============
CELEBDF_BASE_DIR = r"C:\Data Razorpay\data\celebdf"
TEST_LIST_PATH = r"C:\Data Razorpay\data\celebdf\List_of_testing_videos.txt"
OUTPUT_DIR = r"C:\Data Razorpay\data\celebdf_processed"

FRAMES_PER_SECOND = 1
FACE_SIZE = 224
MAX_FRAMES_PER_VIDEO = 30
# ==================================


def parse_test_list(path):
    """Returns list of (label_str, relative_path) tuples."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            label_int, rel_path = parts
            label = "real" if label_int == "1" else "fake"
            entries.append((label, rel_path))
    return entries


def extract_faces_from_video(video_path, mtcnn, out_dir, video_id, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(fps / FRAMES_PER_SECOND), 1)

    saved_files = []
    frame_idx = 0
    saved_count = 0

    while cap.isOpened() and saved_count < max_frames:
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
                    face_resized = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
                    face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)

                    out_filename = f"{video_id}_frame{frame_idx}.jpg"
                    out_path = out_dir / out_filename
                    cv2.imwrite(str(out_path), face_bgr)
                    saved_files.append(out_filename)
                    saved_count += 1

        frame_idx += 1

    cap.release()
    return saved_files


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)

    output_dir = Path(OUTPUT_DIR)
    faces_dir = output_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    entries = parse_test_list(TEST_LIST_PATH)
    print(f"Parsed {len(entries)} entries from official test list "
          f"({sum(1 for l, _ in entries if l == 'real')} real, "
          f"{sum(1 for l, _ in entries if l == 'fake')} fake)\n")

    rows = []
    missing = []
    for i, (label, rel_path) in enumerate(entries):
        video_path = Path(CELEBDF_BASE_DIR) / rel_path
        if not video_path.exists():
            missing.append(rel_path)
            continue

        video_id = Path(rel_path).stem.replace("/", "_")
        video_id = f"{Path(rel_path).parent.name}_{video_id}"

        saved_files = extract_faces_from_video(
            video_path, mtcnn, faces_dir, video_id, MAX_FRAMES_PER_VIDEO
        )

        for fname in saved_files:
            rows.append({"filename": fname, "video_id": video_id, "label": label})

        if (i + 1) % 25 == 0 or (i + 1) == len(entries):
            print(f"  {i + 1}/{len(entries)} videos processed")

    if missing:
        print(f"\nWARNING: {len(missing)} videos from the list were not "
              f"found on disk. First few missing:")
        for m in missing[:5]:
            print(f"  {m}")

    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "video_id", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Extracted {len(rows)} face crops from "
          f"{len(entries) - len(missing)} videos.")
    print(f"Faces saved to: {faces_dir}")
    print(f"Labels CSV saved to: {csv_path}")
    print("\nNext step: run cross_dataset_eval.py (CELEBDF_LABELS_CSV and "
          "CELEBDF_FACES_DIR in that script already point here).")


if __name__ == "__main__":
    main()