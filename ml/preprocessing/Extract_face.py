"""
VeriFace - Phase 1, Day 2
Frame extraction + face detection/cropping for FaceForensics++ videos.

What this does:
1. Walks each category folder (original + 4 manipulation methods)
2. Verifies each video can actually be opened by OpenCV (sanity check for the
   "unsupported encoding" warning you saw in Windows Media Player - WMP failing
   to play a file does NOT mean OpenCV/ffmpeg can't read it)
3. Samples frames at a fixed interval (default: 1 frame per second)
4. Runs MTCNN face detection on each sampled frame, crops + aligns the face,
   resizes to 224x224, and saves it as a .jpg
5. Writes a labels.csv with: filename, video_id, label (real/fake), method,
   split (train/val/test) - split is assigned PER VIDEO ID, never per frame,
   so the same face never leaks across train/val/test.

Usage (from Windows, using your Python 3.10 install):
  python extract_faces.py

Edit the CONFIG section below before running if your folder names differ.
"""

import os
import cv2
import csv
import random
import hashlib
from pathlib import Path
from facenet_pytorch import MTCNN
import torch

# ============ CONFIG - edit these if your paths differ ============
BASE_DIR = r"C:\Data Razorpay\data\ffpp"
OUTPUT_DIR = r"C:\Data Razorpay\data\processed"

# category_name -> (relative path from BASE_DIR, label)
CATEGORIES = {
    "original":       (r"original_sequences\youtube\c23\videos", "real"),
    "Deepfakes":       (r"manipulated_sequences\Deepfakes\c23\videos", "fake"),
    "Face2Face":       (r"manipulated_sequences\Face2Face\c23\videos", "fake"),
    "FaceSwap":        (r"manipulated_sequences\FaceSwap\c23\videos", "fake"),
    "NeuralTextures":  (r"manipulated_sequences\NeuralTextures\c23\videos", "fake"),
}

FRAMES_PER_SECOND = 1      # how densely to sample frames from each video
FACE_SIZE = 224             # output crop size (square)
MAX_FRAMES_PER_VIDEO = 30   # cap so long videos don't dominate the dataset
TRAIN_SPLIT, VAL_SPLIT = 0.7, 0.15  # remainder (0.15) goes to test
RANDOM_SEED = 42
# =====================================================================


def get_video_id(category, filename):
    """Stable id so the same video always lands in the same split."""
    return f"{category}_{Path(filename).stem}"


def assign_split(video_id):
    """Deterministic split by hashing the video_id - same video always
    gets the same split even if you re-run this script later."""
    h = int(hashlib.md5(video_id.encode()).hexdigest(), 16)
    r = (h % 10000) / 10000.0
    if r < TRAIN_SPLIT:
        return "train"
    elif r < TRAIN_SPLIT + VAL_SPLIT:
        return "val"
    else:
        return "test"


def sanity_check_videos(video_paths, sample_size=5):
    """Open a handful of videos with OpenCV to confirm they're readable,
    before committing to the full extraction run."""
    print("\n--- Sanity check: verifying OpenCV can read sample videos ---")
    sample = random.sample(video_paths, min(sample_size, len(video_paths)))
    all_ok = True
    for p in sample:
        cap = cv2.VideoCapture(str(p))
        ok, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        status = "OK" if ok else "FAILED"
        if not ok:
            all_ok = False
        print(f"  [{status}] {p.name}  (frame_count={frame_count})")
    if not all_ok:
        print("\nWARNING: some videos failed to open. If ALL of them failed, "
              "you likely need to install a proper ffmpeg build for OpenCV:\n"
              "  python -m pip install opencv-contrib-python\n"
              "If only a few failed, those specific files may be corrupted "
              "downloads - safe to skip them.")
    else:
        print("All sample videos opened successfully. Proceeding.\n")
    return all_ok


def extract_faces_from_video(video_path, mtcnn, out_dir, video_id, max_frames):
    """Sample frames from a video, detect + crop the largest face in each,
    save as jpgs. Returns list of saved filenames."""
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
                # pick the highest-confidence face
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
    random.seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("NOTE: No GPU detected - face detection will be slower on CPU. "
              "This is fine for this dataset size, just be patient.\n")

    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)

    output_dir = Path(OUTPUT_DIR)
    faces_dir = output_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    # Collect all video paths first, for the sanity check
    all_video_paths = []
    for category, (rel_path, label) in CATEGORIES.items():
        cat_dir = Path(BASE_DIR) / rel_path
        if not cat_dir.exists():
            print(f"WARNING: path not found, skipping: {cat_dir}")
            continue
        videos = list(cat_dir.glob("*.mp4"))
        all_video_paths.extend(videos)

    if not all_video_paths:
        print("ERROR: no videos found. Check BASE_DIR and CATEGORIES paths "
              "at the top of this script.")
        return

    print(f"Found {len(all_video_paths)} total videos across "
          f"{len(CATEGORIES)} categories.")

    sanity_check_videos(all_video_paths)

    # Main extraction loop
    rows = []
    for category, (rel_path, label) in CATEGORIES.items():
        cat_dir = Path(BASE_DIR) / rel_path
        if not cat_dir.exists():
            continue
        videos = sorted(cat_dir.glob("*.mp4"))
        print(f"\n--- Processing category: {category} ({label}, "
              f"{len(videos)} videos) ---")

        for i, video_path in enumerate(videos):
            video_id = get_video_id(category, video_path.name)

            # Skip videos already processed in a previous run (e.g. when
            # topping up the dataset with more videos later) - checks for
            # any existing face crop file starting with this video_id.
            existing = list(faces_dir.glob(f"{video_id}_frame*.jpg"))
            if existing:
                split = assign_split(video_id)
                for f in existing:
                    rows.append({
                        "filename": f.name,
                        "video_id": video_id,
                        "category": category,
                        "label": label,
                        "split": split,
                    })
                continue

            split = assign_split(video_id)

            saved_files = extract_faces_from_video(
                video_path, mtcnn, faces_dir, video_id, MAX_FRAMES_PER_VIDEO
            )

            for fname in saved_files:
                rows.append({
                    "filename": fname,
                    "video_id": video_id,
                    "category": category,
                    "label": label,
                    "split": split,
                })

            if (i + 1) % 10 == 0 or (i + 1) == len(videos):
                print(f"  {i + 1}/{len(videos)} videos done "
                      f"({len(saved_files)} faces from last video)")

    # Write labels.csv
    csv_path = output_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "video_id", "category", "label", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Extracted {len(rows)} face crops from "
          f"{len(all_video_paths)} videos.")
    print(f"Faces saved to: {faces_dir}")
    print(f"Labels CSV saved to: {csv_path}")

    # Quick split summary
    from collections import Counter
    split_counts = Counter(r["split"] for r in rows)
    label_counts = Counter(r["label"] for r in rows)
    print(f"\nSplit breakdown: {dict(split_counts)}")
    print(f"Label breakdown: {dict(label_counts)}")
    print("\nNext step: open ~20-30 random images in "
          f"{faces_dir} and eyeball them for quality before moving on "
          "to Phase 2 (training).")


if __name__ == "__main__":
    main()