from pathlib import Path
import os
import uuid
import shutil
from typing import List, Dict, Any, Optional

import torch
import cv2  # NEW: use OpenCV instead of PyAV/torchvision for video loading
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from transformers import Sam3VideoModel, Sam3VideoProcessor

# -----------------------------
# Device & model initialization
# -----------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Use bfloat16 only on CUDA; otherwise stay in float32 for safety (good for Mac)
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

print(f"[INIT] Using device: {DEVICE}, dtype: {DTYPE}")

print("[INIT] Loading SAM3 Video model...")
MODEL = Sam3VideoModel.from_pretrained("facebook/sam3")
MODEL = MODEL.to(device=DEVICE, dtype=DTYPE)
MODEL.eval()

PROCESSOR = Sam3VideoProcessor.from_pretrained("facebook/sam3")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="SAM3 CCTV Search")


# -----------------------------
# Video loading with OpenCV
# -----------------------------

def load_video_opencv(path: str, max_frames: Optional[int] = None):
    """
    Load video frames using OpenCV and return (frames, info).

    frames: list of RGB numpy arrays (H, W, 3) uint8
    info: dict with at least "video_fps"
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0  # fallback

    frames = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        # Convert BGR (OpenCV) -> RGB (what processors expect)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if not frames:
        raise RuntimeError("No frames could be read from the video.")

    info = {"video_fps": fps}
    return frames, info


# -----------------------------
# Core search function
# -----------------------------

def search_video_for_concept(
    video_path: str,
    text: str,
    score_threshold: float = 0.5,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run SAM3 Video on a video file and return frames where `text` appears.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"[SEARCH] Loading video (OpenCV): {video_path}")
    video_frames, info = load_video_opencv(str(video_path), max_frames=max_frames)
    fps = float(info.get("video_fps", 25.0))
    total_frames = len(video_frames)
    print(f"[SEARCH] Loaded {total_frames} frames at ~{fps:.2f} fps")

    # Initialize a SAM3 video session
    print(f"[SEARCH] Initializing SAM3 video session with query: {text!r}")
    inference_session = PROCESSOR.init_video_session(
        video=video_frames,
        inference_device=DEVICE,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=DTYPE,
    )

    inference_session = PROCESSOR.add_text_prompt(
        inference_session=inference_session,
        text=text,
    )

    hits: List[Dict[str, Any]] = []

    print("[SEARCH] Running propagate_in_video_iterator...")
    for model_outputs in MODEL.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=total_frames,
    ):
        frame_idx = model_outputs.frame_idx
        processed = PROCESSOR.postprocess_outputs(inference_session, model_outputs)

        scores = processed["scores"]  # tensor [num_objects]
        if scores is None or scores.numel() == 0:
            continue

        keep_mask = scores >= score_threshold
        if keep_mask.any():
            num_matches = int(keep_mask.sum().item())
            time_sec = frame_idx / fps

            hits.append(
                {
                    "frame_idx": frame_idx,
                    "time_sec": float(time_sec),
                    "num_matches": num_matches,
                    "scores": [float(s) for s in scores[keep_mask]],
                    "object_ids": [int(i) for i in processed["object_ids"][keep_mask]],
                }
            )

    return {
        "fps": fps,
        "total_frames": total_frames,
        "text": text,
        "score_threshold": score_threshold,
        "hits": hits,
    }


# -----------------------------
# Simple HTML UI
# -----------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8" />
    <title>SAM3 CCTV Search</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 2rem auto;
            max-width: 900px;
            line-height: 1.5;
        }}
        h1 {{
            margin-bottom: 0.25rem;
        }}
        .subtitle {{
            color: #555;
            margin-bottom: 1.5rem;
        }}
        form {{
            border: 1px solid #ddd;
            padding: 1rem 1.25rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }}
        label {{
            display: block;
            margin-top: 0.75rem;
            font-weight: 600;
        }}
        input[type="text"],
        input[type="number"],
        input[type="file"] {{
            width: 100%;
            padding: 0.4rem;
            margin-top: 0.25rem;
            box-sizing: border-box;
        }}
        button {{
            margin-top: 1rem;
            padding: 0.5rem 1.25rem;
            border-radius: 999px;
            border: none;
            cursor: pointer;
            background-color: #2563eb;
            color: white;
            font-weight: 600;
        }}
        button:hover {{
            background-color: #1d4ed8;
        }}
        .results {{
            margin-top: 1rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.75rem;
        }}
        th, td {{
            border: 1px solid #e5e7eb;
            padding: 0.4rem 0.5rem;
            text-align: left;
            font-size: 0.9rem;
        }}
        th {{
            background-color: #f9fafb;
        }}
        .message {{
            margin-top: 0.75rem;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            background-color: #eff6ff;
            color: #1d4ed8;
            font-size: 0.9rem;
        }}
        .error {{
            background-color: #fef2f2;
            color: #b91c1c;
        }}
        .small {{
            font-size: 0.8rem;
            color: #6b7280;
        }}
    </style>
</head>
<body>
    <h1>SAM3 CCTV Search</h1>
    <div class="subtitle">
        Upload a video, type what you're looking for (e.g. <b>"yellow car"</b>),
        and SAM3 will find frames where it appears.
    </div>

    <form action="/search" method="post" enctype="multipart/form-data">
        <label for="video_file">Video file (mp4, etc.)</label>
        <input type="file" name="video_file" accept="video/*" required />

        <label for="query">Search query</label>
        <input type="text" name="query" placeholder="yellow car, person in red shirt, blue truck..." required />

        <label for="score_threshold">Score threshold (default: 0.5)</label>
        <input type="number" step="0.05" name="score_threshold" value="0.5" />

        <label for="max_frames">Max frames to process (optional, e.g. 300 for quick test)</label>
        <input type="number" name="max_frames" />

        <button type="submit">Search video</button>
        <div class="small">
            Note: First request may take a while while the model loads & processes frames.
        </div>
    </form>

    {message_block}
    {results_block}
</body>
</html>
"""


def render_page(message_block: str = "", results_block: str = "") -> str:
    return HTML_TEMPLATE.format(
        message_block=message_block, results_block=results_block
    )


# -----------------------------
# Routes
# -----------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return render_page()


@app.post("/search", response_class=HTMLResponse)
async def search(
    video_file: UploadFile = File(...),
    query: str = Form(...),
    score_threshold: float = Form(0.5),
    max_frames: Optional[str] = Form(None),
) -> str:
    try:
        # Save uploaded file to disk
        ext = os.path.splitext(video_file.filename or "")[1]
        if not ext:
            ext = ".mp4"
        out_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"

        with open(out_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        max_frames_int = None
        if max_frames:
            try:
                max_frames_int = int(max_frames)
            except ValueError:
                max_frames_int = None

        # Run search
        results = search_video_for_concept(
            video_path=str(out_path),
            text=query,
            score_threshold=score_threshold,
            max_frames=max_frames_int,
        )

        hits = results["hits"]
        if not hits:
            message = (
                f"No frames found containing <b>{query}</b> "
                f"(score ≥ {score_threshold})."
            )
            message_block = f'<div class="message">{message}</div>'
            return render_page(message_block=message_block)

        # Build HTML table of hits
        rows = []
        for h in hits:
            t = h["time_sec"]
            mm = int(t // 60)
            ss = t % 60
            timestamp = f"{mm:02d}:{ss:05.2f}"
            scores_str = ", ".join(f"{s:.3f}" for s in h["scores"])

            rows.append(
                f"<tr>"
                f"<td>{h['frame_idx']}</td>"
                f"<td>{timestamp}</td>"
                f"<td>{h['num_matches']}</td>"
                f"<td>{scores_str}</td>"
                f"<td>{h['object_ids']}</td>"
                f"</tr>"
            )

        table_html = (
            '<div class="results">'
            f"<b>Results for query:</b> {query}<br/>"
            f"Score threshold: {score_threshold} &middot; "
            f"FPS: {results['fps']:.2f} &middot; "
            f"Frames processed: {results['total_frames']}<br/>"
            "<table>"
            "<thead><tr>"
            "<th>Frame</th><th>Timestamp</th><th># Matches</th>"
            "<th>Scores</th><th>Object IDs</th>"
            "</tr></thead>"
            "<tbody>"
            + "".join(rows)
            + "</tbody></table>"
            "</div>"
        )

        message_block = (
            '<div class="message">'
            "Search completed. Scroll down to see the frames where the object appears."
            "</div>"
        )

        return render_page(message_block=message_block, results_block=table_html)

    except Exception as e:
        error_block = (
            f'<div class="message error">Error while processing video: {e}</div>'
        )
        return render_page(message_block=error_block)
