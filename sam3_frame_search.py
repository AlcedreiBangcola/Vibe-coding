import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image

from transformers import Sam3Processor, Sam3Model

# -----------------------------
# Device + model init
# -----------------------------

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"[INIT] Using device: {DEVICE}")

print("[INIT] Loading SAM3 image model...")
model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
model.eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")


# -----------------------------
# Helper: run SAM3 on one frame
# -----------------------------

def sam3_matches(frame_rgb: np.ndarray, text: str, score_threshold: float = 0.5) -> bool:
    """
    Run SAM3 on a single RGB frame (H, W, 3 uint8).
    Returns True if any object has score >= score_threshold.
    """
    image = Image.fromarray(frame_rgb)

    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    scores = results["scores"]  # tensor [num_objects]
    if scores is None or scores.numel() == 0:
        return False

    keep = scores >= score_threshold
    return bool(keep.any().item())


# -----------------------------
# Main: scan video & save hits
# -----------------------------

def search_video(
    video_path: str,
    text_query: str,
    frame_step: int = 30,
    score_threshold: float = 0.5,
    output_dir: str | Path = "hits",
) -> None:
    """
    Scan a video with SAM3 (image mode) by sampling frames.

    - video_path: path to video file
    - text_query: e.g. "yellow car", "person in red shirt"
    - frame_step: process every Nth frame (e.g. 30 ≈ 1 frame per second if 30 fps)
    - score_threshold: minimum score for SAM3 to count a detection
    - output_dir: folder where matching frames are saved as images
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] FPS: {fps:.2f}, total frames: {total_frames}")
    print(f"[INFO] Sampling every {frame_step} frames")
    print(f"[INFO] Query: {text_query!r}, score threshold: {score_threshold}")

    frame_idx = 0
    hits = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Only process every Nth frame
        if frame_idx % frame_step == 0:
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # OPTIONAL: downscale to speed up / save memory
            # frame_rgb = cv2.resize(frame_rgb, (640, 360))

            time_sec = frame_idx / fps
            mm = int(time_sec // 60)
            ss = time_sec % 60
            timestamp = f"{mm:02d}:{ss:05.2f}"

            print(f"[FRAME] idx={frame_idx}, t={timestamp} → running SAM3...", end="", flush=True)
            try:
                match = sam3_matches(frame_rgb, text_query, score_threshold=score_threshold)
            except Exception as e:
                print(f" ERROR: {e}")
                frame_idx += 1
                continue

            if match:
                hits += 1
                filename = f"{video_path.stem}_f{frame_idx:06d}.jpg"
                out_path = out_dir / filename
                # Save original BGR frame as JPEG
                cv2.imwrite(str(out_path), frame_bgr)
                print(f" MATCH → saved {out_path}")
            else:
                print(" no match")

        frame_idx += 1

    cap.release()

    print("\n[RESULTS]")
    print(f"Processed frames: ~{total_frames // frame_step} (step={frame_step})")
    print(f"Matching frames saved: {hits}")
    print(f"Output folder: {out_dir.resolve()}")


# -----------------------------
# CLI entrypoint
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample video frames and run SAM3 image segmentation to find matches."
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--query",
        required=True,
        help='Text query, e.g. "yellow car", "person in red shirt"',
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=30,
        help="Process every Nth frame (default: 30 ≈ 1 fps for 30 fps video)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum SAM3 score to count as a match (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        default="hits",
        help="Directory to save matching frames (default: hits)",
    )

    args = parser.parse_args()

    search_video(
        video_path=args.video,
        text_query=args.query,
        frame_step=args.frame_step,
        score_threshold=args.score_threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
