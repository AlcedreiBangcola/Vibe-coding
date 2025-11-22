from pathlib import Path
import os
import uuid
import shutil
import base64
import csv
import time
from typing import List, Dict, Any, Optional

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

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

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

HITS_DIR = Path("hits")
HITS_DIR.mkdir(exist_ok=True)

# sessions older than this (in days) will be deleted by "Clean old sessions"
SESSION_MAX_AGE_DAYS = 3

app = FastAPI(title="SAM3 Frame-based CCTV Search")


def slugify_query(text_query: str) -> str:
    """Turn a query into a safe folder name like 'black_car'."""
    safe = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in text_query.strip().lower()
    )
    return safe or "query"


# -----------------------------
# SAM3 + highlighting helpers
# -----------------------------

def apply_mask_overlay(
    frame_rgb: np.ndarray,
    masks: torch.Tensor,
    scores: torch.Tensor,
    label_text: str,
) -> np.ndarray:
    """
    Take an RGB frame and [K, H, W] masks, plus scores, and return a BGR image
    with a light red transparent overlay + per-object boxes and labels.

    Each detected object gets:
      - its own soft red mask
      - its own bounding box
      - its own label like "black car (0.87)" drawn INSIDE the box
    """
    image = Image.fromarray(frame_rgb).convert("RGBA")

    # masks: [K, H, W] -> boolean
    masks_np = (masks.detach().cpu().numpy() > 0.5)  # [K, H, W] bool
    scores_np = scores.detach().cpu().numpy()        # [K]

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, mask_bool in enumerate(masks_np):
        # -------- soft overlay (light red) --------
        mask_img = Image.fromarray((mask_bool.astype("uint8") * 255))
        overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))  # red
        # alpha=80 → very transparent red tint, object still visible
        overlay.putalpha(mask_img.point(lambda v: 80 if v > 0 else 0))
        image = Image.alpha_composite(image, overlay)

        # -------- bounding box for this object --------
        ys, xs = np.where(mask_bool)
        if ys.size == 0 or xs.size == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline=(255, 0, 0, 255),
            width=3,
        )

        # -------- label text for this object --------
        score_val = float(scores_np[i])
        # e.g. "black car (0.87)"
        text = f"{label_text} ({score_val:.2f})"

        if font is not None:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        else:
            text_w, text_h = draw.textlength(text), 14  # fallback

        pad = 4

        # Place label INSIDE the box at the top-left corner
        label_left = x_min + 1
        label_top = y_min + 1
        label_right = label_left + text_w + 2 * pad
        label_bottom = label_top + text_h + 2 * pad

        # Clamp label inside image bounds
        img_w, img_h = image.size
        if label_right > img_w:
            shift = label_right - img_w
            label_left -= shift
            label_right -= shift
        if label_bottom > img_h:
            shift = label_bottom - img_h
            label_top -= shift
            label_bottom -= shift

        # Solid background so text is readable even over video
        draw.rectangle(
            [(label_left, label_top), (label_right, label_bottom)],
            fill=(255, 0, 0, 220),
        )

        text_x = label_left + pad
        text_y = label_top + pad

        # Draw label text (white)
        draw.text(
            (text_x, text_y),
            text,
            fill=(255, 255, 255, 255),
            font=font,
        )

    # Back to BGR for OpenCV
    image_rgb = image.convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    return image_bgr

def sam3_highlight_frame(
    frame_rgb: np.ndarray,
    text: str,
    score_threshold: float = 0.5,
) -> tuple[bool, Optional[np.ndarray], Optional[torch.Tensor]]:
    """
    Run SAM3 on a single RGB frame.
    Returns:
        has_match: bool
        highlighted_bgr: np.ndarray | None
        kept_scores: torch.Tensor | None
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

    scores = results["scores"]      # [N]
    masks = results["masks"]        # [N, H, W]

    if scores is None or scores.numel() == 0 or masks is None:
        return False, None, None

    keep = scores >= score_threshold
    if not keep.any():
        return False, None, None

    kept_masks = masks[keep]               # [K, H, W]
    kept_scores = scores[keep]             # [K]

    highlighted_bgr = apply_mask_overlay(frame_rgb, kept_masks, kept_scores, label_text=text)
    return True, highlighted_bgr, kept_scores


# -----------------------------
# Main video scan logic
# -----------------------------

def search_video_for_concept(
    video_path: str,
    text_query: str,
    frame_step: int = 30,
    score_threshold: float = 0.5,
    output_dir: Path = HITS_DIR,
) -> Dict[str, Any]:
    """
    Scan a video with SAM3 (image mode) by sampling frames.

    Saves highlighted frames under:
        hits/<query_slug>/<session_id>/...

    Also writes a results.csv in that folder.

    Returns a dict with fps, total_frames, and a list of hits.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    query_slug = slugify_query(text_query)

    session_id = uuid.uuid4().hex
    query_root = output_dir / query_slug
    session_dir = query_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] FPS: {fps:.2f}, total frames: {total_frames}")
    print(f"[INFO] Sampling every {frame_step} frames")
    print(f"[INFO] Query: {text_query!r}, slug: {query_slug}")
    print(f"[INFO] Saving to: {session_dir}")

    frame_idx = 0
    hits: List[Dict[str, Any]] = []

    MAX_SIDE = 720  # downscale longest side to this (for speed & memory)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Downscale for SAM3
            h, w = frame_rgb.shape[:2]
            scale = min(1.0, MAX_SIDE / max(h, w))
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            time_sec = frame_idx / fps
            mm = int(time_sec // 60)
            ss = time_sec % 60
            timestamp = f"{mm:02d}:{ss:05.2f}"

            print(
                f"[FRAME] idx={frame_idx}, t={timestamp} → running SAM3...",
                end="",
                flush=True,
            )

            try:
                has_match, highlighted_bgr, kept_scores = sam3_highlight_frame(
                    frame_rgb, text_query, score_threshold=score_threshold
                )
            except Exception as e:
                print(f" ERROR: {e}")
                frame_idx += 1
                continue

            if has_match and highlighted_bgr is not None and kept_scores is not None:
                filename = f"{video_path.stem}_f{frame_idx:06d}.jpg"
                out_path = session_dir / filename
                cv2.imwrite(str(out_path), highlighted_bgr)
                print(f" MATCH → saved {out_path}")

                scores_list = [float(s) for s in kept_scores]
                hits.append(
                    {
                        "frame_idx": frame_idx,
                        "time_sec": float(time_sec),
                        "image_path": out_path,
                        "scores": scores_list,
                        "num_objects": int(len(scores_list)),
                    }
                )
            else:
                print(" no match")

        frame_idx += 1

    cap.release()

    processed_frames_est = max(1, total_frames // max(1, frame_step))
    print("\n[RESULTS]")
    print(f"Processed frames: ~{processed_frames_est} (step={frame_step})")
    print(f"Matching frames saved: {len(hits)}")
    print(f"Output folder: {session_dir.resolve()}")

    # Write CSV summary
    csv_path = session_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query",
                "frame_idx",
                "time_sec",
                "num_objects",
                "scores",
                "image_filename",
            ]
        )
        for h in hits:
            writer.writerow(
                [
                    text_query,
                    h["frame_idx"],
                    h["time_sec"],
                    h["num_objects"],
                    ";".join(f"{s:.4f}" for s in h["scores"]),
                    h["image_path"].name,
                ]
            )

    return {
        "fps": fps,
        "total_frames": total_frames,
        "frame_step": frame_step,
        "text": text_query,
        "score_threshold": score_threshold,
        "hits": hits,
        "session_dir": session_dir,
        "query_slug": query_slug,
        "session_id": session_id,
        "csv_path": csv_path,
    }


# -----------------------------
# HTML UI (dark theme + loading + clean sessions)
# -----------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8" />
    <title>SAM3 CCTV Frame Search</title>
    <style>
        :root {{
            color-scheme: dark;
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(circle at top, #0f172a 0, #020617 45%, #000 100%);
            color: #e5e7eb;
        }}
        .page {{
            max-width: 1180px;
            margin: 0 auto 4rem auto;
            padding: 1.8rem 1.4rem 3rem 1.4rem;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.2rem;
        }}
        .title {{
            font-size: 1.7rem;
            font-weight: 700;
        }}
        .badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.6);
            color: #7dd3fc;
            font-size: 0.78rem;
            font-weight: 600;
        }}
        .subtitle {{
            color: #9ca3af;
            margin-bottom: 1.75rem;
            font-size: 0.9rem;
        }}
        .layout {{
            display: grid;
            grid-template-columns: minmax(0, 340px) minmax(0, 1fr);
            gap: 1.5rem;
        }}
        @media (max-width: 900px) {{
            .layout {{
                grid-template-columns: minmax(0, 1fr);
            }}
        }}
        .card {{
            background: radial-gradient(circle at top left, #111827 0, #020617 70%);
            border-radius: 18px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(31, 41, 55, 0.9);
            position: relative;
            overflow: hidden;
        }}
        .card::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.07) 0, transparent 50%);
            pointer-events: none;
        }}
        .card-header {{
            padding: 0.9rem 1.1rem 0.4rem 1.1rem;
            border-bottom: 1px solid rgba(31, 41, 55, 0.9);
            font-weight: 600;
            font-size: 0.9rem;
            color: #e5e7eb;
            position: relative;
            z-index: 1;
        }}
        .card-body {{
            padding: 0.9rem 1.1rem 1.1rem 1.1rem;
            position: relative;
            z-index: 1;
        }}
        form label {{
            display: block;
            margin-top: 0.7rem;
            font-size: 0.8rem;
            font-weight: 600;
            color: #9ca3af;
        }}
        input[type="text"],
        input[type="number"],
        input[type="file"],
        select {{
            width: 100%;
            padding: 0.45rem 0.6rem;
            margin-top: 0.25rem;
            border-radius: 9px;
            border: 1px solid #374151;
            font-size: 0.83rem;
            background: #020617;
            color: #e5e7eb;
        }}
        input::file-selector-button {{
            border-radius: 999px;
            border: none;
            padding: 0.3rem 0.8rem;
            margin-right: 0.6rem;
            background: #1d4ed8;
            color: #e5e7eb;
            font-size: 0.75rem;
            cursor: pointer;
        }}
        input:focus,
        select:focus {{
            outline: none;
            border-color: #60a5fa;
            box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.4);
        }}
        button {{
            margin-top: 0.9rem;
            padding: 0.6rem 1.25rem;
            border-radius: 999px;
            border: none;
            cursor: pointer;
            background: linear-gradient(to right, #2563eb, #7c3aed);
            color: white;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        button:hover {{
            background: linear-gradient(to right, #1d4ed8, #6d28d9);
        }}
        .btn-secondary {{
            background: transparent;
            border: 1px solid #4b5563;
            color: #e5e7eb;
            padding: 0.45rem 1rem;
            font-size: 0.8rem;
        }}
        .btn-secondary:hover {{
            background: rgba(55, 65, 81, 0.4);
        }}
        .small {{
            font-size: 0.74rem;
            color: #6b7280;
            margin-top: 0.5rem;
        }}
        .message {{
            margin-top: 0.3rem;
            padding: 0.5rem 0.75rem;
            border-radius: 12px;
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.6);
            color: #bfdbfe;
            font-size: 0.83rem;
        }}
        .error {{
            background: rgba(248, 113, 113, 0.16);
            border-color: rgba(248, 113, 113, 0.8);
            color: #fecaca;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.6rem;
            font-size: 0.8rem;
        }}
        .summary-item {{
            background: rgba(15, 23, 42, 0.9);
            border-radius: 10px;
            padding: 0.5rem 0.65rem;
            border: 1px solid #1f2937;
        }}
        .summary-label {{
            color: #9ca3af;
            font-size: 0.73rem;
        }}
        .summary-value {{
            font-weight: 600;
            margin-top: 0.1rem;
        }}
        .results-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .result-card {{
            border-radius: 12px;
            background-color: rgba(15, 23, 42, 0.95);
            border: 1px solid #1f2937;
            padding: 0.55rem;
            font-size: 0.83rem;
        }}
        .result-card img {{
            width: 100%;
            border-radius: 10px;
            display: block;
            margin-bottom: 0.45rem;
            cursor: pointer;
        }}
        .result-meta {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            column-gap: 0.5rem;
            row-gap: 0.15rem;
        }}
        .meta-label {{
            color: #9ca3af;
            font-size: 0.74rem;
        }}
        .meta-value {{
            font-weight: 500;
        }}
        /* Modal for expanded image */
        .modal {{
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.88);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }}
        .modal img {{
            max-width: 90%;
            max-height: 90%;
            border-radius: 12px;
            box-shadow: 0 16px 50px rgba(0, 0, 0, 0.75);
        }}
        .csv-link {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            margin-top: 0.7rem;
            font-size: 0.8rem;
            color: #93c5fd;
            text-decoration: none;
        }}
        .csv-link:hover {{
            text-decoration: underline;
        }}
        .csv-icon {{
            font-size: 1rem;
        }}
        .maintenance-card {{
            margin-top: 1rem;
        }}
        /* Loading overlay */
        .loading-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.88);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            flex-direction: column;
            gap: 0.8rem;
        }}
        .spinner {{
            width: 40px;
            height: 40px;
            border-radius: 999px;
            border: 3px solid rgba(148, 163, 184, 0.4);
            border-top-color: #60a5fa;
            animation: spin 0.8s linear infinite;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        .loading-text {{
            font-size: 0.9rem;
            color: #e5e7eb;
        }}
        .loading-sub {{
            font-size: 0.75rem;
            color: #9ca3af;
        }}
    </style>
    <script>
        function openImage(src) {{
            const modal = document.getElementById('imgModal');
            const modalImg = document.getElementById('modalImg');
            modalImg.src = src;
            modal.style.display = 'flex';
        }}
        function closeModal() {{
            const modal = document.getElementById('imgModal');
            modal.style.display = 'none';
        }}
        function showLoading() {{
            const overlay = document.getElementById('loadingOverlay');
            if (overlay) {{
                overlay.style.display = 'flex';
            }}
        }}
    </script>
</head>
<body>
    <div class="page">
        <div class="header">
            <div class="title">SAM3 CCTV Frame Search</div>
            <div class="badge">Local prototype</div>
        </div>
        <div class="subtitle">
            Upload CCTV video, choose a mode, and type what you're looking for
            (e.g. <b>"black car"</b>). The system samples frames and shows only
            the images where SAM3 finds a match, with per-object overlays and
            confidence scores.
        </div>

        <div class="layout">
            <div>
                <div class="card">
                    <div class="card-header">Search settings</div>
                    <div class="card-body">
                        <form action="/search" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
                            <label>Video file</label>
                            <input type="file" name="video_file" accept="video/*" required />

                            <label>Search query</label>
                            <input type="text" name="query" placeholder="black car, person in red shirt, blue truck..." required />

                            <label>Mode</label>
                            <select name="mode">
                                <option value="standard" selected>Standard (frame_step=30, threshold=0.5)</option>
                                <option value="quick">Quick (frame_step=60, threshold=0.7)</option>
                                <option value="deep">Deep (frame_step=10, threshold=0.4)</option>
                                <option value="manual">Manual (use custom values below)</option>
                            </select>

                            <label>Frame step (Manual mode)</label>
                            <input type="number" name="frame_step" value="30" />

                            <label>Score threshold (Manual mode)</label>
                            <input type="number" step="0.05" name="score_threshold" value="0.5" />

                            <button type="submit">Search video</button>
                            <div class="small">
                                Quick = fewer frames & faster; Deep = more frames & more sensitive.
                            </div>
                        </form>
                    </div>
                </div>

                <div class="card maintenance-card">
                    <div class="card-header">Maintenance</div>
                    <div class="card-body">
                        <form action="/clean_sessions" method="post" onsubmit="showLoading()">
                            <div class="small">
                                Clean old sessions (older than {SESSION_MAX_AGE_DAYS} days) to free disk space.
                            </div>
                            <button type="submit" class="btn-secondary">Clean old sessions</button>
                        </form>
                    </div>
                </div>
            </div>

            <div>
                {message_block}
                {results_block}
            </div>
        </div>
    </div>

    <!-- Modal for expanded image -->
    <div id="imgModal" class="modal" onclick="closeModal()">
        <img id="modalImg" src="" alt="expanded frame" />
    </div>

    <!-- Loading overlay (progress indicator) -->
    <div id="loadingOverlay" class="loading-overlay">
        <div class="spinner"></div>
        <div class="loading-text">Processing video with SAM3…</div>
        <div class="loading-sub">This may take a while for longer videos.</div>
    </div>
</body>
</html>
""".replace("{SESSION_MAX_AGE_DAYS}", str(SESSION_MAX_AGE_DAYS))


def render_page(message_block: str = "", results_block: str = "") -> str:
    return HTML_TEMPLATE.format(
        message_block=message_block,
        results_block=results_block,
    )


# -----------------------------
# Routes
# -----------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return render_page()


@app.get("/download_csv", response_class=FileResponse)
async def download_csv(query_slug: str, session_id: str):
    """
    Serve the CSV summary for a given query/session.
    """
    csv_path = HITS_DIR / query_slug / session_id / "results.csv"
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="CSV not found for this session.")
    return FileResponse(
        path=csv_path,
        filename=f"{query_slug}_{session_id}_results.csv",
        media_type="text/csv",
    )


@app.post("/clean_sessions", response_class=HTMLResponse)
async def clean_sessions() -> str:
    """
    Delete session folders under hits/ that are older than SESSION_MAX_AGE_DAYS.
    """
    now = time.time()
    cutoff = SESSION_MAX_AGE_DAYS * 24 * 60 * 60
    deleted_sessions = 0

    for query_dir in HITS_DIR.iterdir():
        if not query_dir.is_dir():
            continue
        for session_dir in query_dir.iterdir():
            if not session_dir.is_dir():
                continue
            age_seconds = now - session_dir.stat().st_mtime
            if age_seconds > cutoff:
                shutil.rmtree(session_dir, ignore_errors=True)
                deleted_sessions += 1
        # remove empty query_dir
        try:
            if not any(query_dir.iterdir()):
                query_dir.rmdir()
        except OSError:
            pass

    if deleted_sessions == 0:
        message = "No sessions were old enough to clean."
    else:
        message = f"Cleaned {deleted_sessions} old session(s)."

    message_block = f'<div class="message">{message}</div>'
    return render_page(message_block=message_block)


@app.post("/search", response_class=HTMLResponse)
async def search(
    video_file: UploadFile = File(...),
    query: str = Form(...),
    mode: str = Form("standard"),
    frame_step: Optional[str] = Form("30"),
    score_threshold: float = Form(0.5),
) -> str:
    try:
        # Save uploaded video to disk
        ext = os.path.splitext(video_file.filename or "")[1]
        if not ext:
            ext = ".mp4"

        video_tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
        with open(video_tmp_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)

        # Decide settings based on mode
        mode_lower = mode.lower()
        if mode_lower == "quick":
            frame_step_int = 60
            threshold_used = 0.7
        elif mode_lower == "deep":
            frame_step_int = 10
            threshold_used = 0.4
        elif mode_lower == "standard":
            frame_step_int = 30
            threshold_used = 0.5
        else:  # manual
            try:
                frame_step_int = int(frame_step) if frame_step else 30
                if frame_step_int <= 0:
                    frame_step_int = 30
            except ValueError:
                frame_step_int = 30
            threshold_used = score_threshold

        # Run search
        results = search_video_for_concept(
            video_path=str(video_tmp_path),
            text_query=query,
            frame_step=frame_step_int,
            score_threshold=threshold_used,
            output_dir=HITS_DIR,
        )

        # Clean up uploaded video
        try:
            os.remove(video_tmp_path)
        except OSError:
            pass

        hits = results["hits"]
        query_slug = results["query_slug"]
        session_id = results["session_id"]

        if not hits:
            message = (
                f"No frames found containing <b>{query}</b> "
                f"(score ≥ {threshold_used})."
            )
            message_block = f'<div class="message error">{message}</div>'
            return render_page(message_block=message_block)

        # Summary block
        total_hits = len(hits)
        processed_est = max(1, results["total_frames"] // max(1, results["frame_step"]))
        total_objects = sum(h.get("num_objects", 0) for h in hits)
        all_scores = [s for h in hits for s in h.get("scores", [])]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        summary_html = f"""
        <div class="card" style="margin-bottom: 1rem;">
            <div class="card-header">Run summary</div>
            <div class="card-body">
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-label">Query</div>
                        <div class="summary-value">{query}</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-label">Mode</div>
                        <div class="summary-value">{mode_lower.capitalize()}</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-label">Hits (frames)</div>
                        <div class="summary-value">{total_hits}</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-label">Total objects detected</div>
                        <div class="summary-value">{total_objects}</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-label">Avg score</div>
                        <div class="summary-value">{avg_score:.2f}</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-label">Frames processed (approx)</div>
                        <div class="summary-value">{processed_est}</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-label">Frame step</div>
                        <div class="summary-value">{results["frame_step"]}</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-label">Threshold used</div>
                        <div class="summary-value">{threshold_used}</div>
                    </div>
                </div>
                <a class="csv-link" href="/download_csv?query_slug={query_slug}&session_id={session_id}">
                    <span class="csv-icon">📄</span>
                    <span>Download CSV summary</span>
                </a>
            </div>
        </div>
        """

        # Per-frame cards
        cards = []
        for h in hits:
            t = h["time_sec"]
            mm = int(t // 60)
            ss = t % 60
            timestamp = f"{mm:02d}:{ss:05.2f}"

            img_path: Path = h["image_path"]
            scores = h.get("scores", [])
            num_objects = h.get("num_objects", len(scores))
            scores_str = ", ".join(f"{s:.2f}" for s in scores)

            with open(img_path, "rb") as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            img_src = f"data:image/jpeg;base64,{img_b64}"

            cards.append(
                f"""
                <div class="result-card">
                    <img src="{img_src}" alt="frame {h['frame_idx']}" onclick="openImage('{img_src}')" />
                    <div class="result-meta">
                        <div>
                            <div class="meta-label">Frame</div>
                            <div class="meta-value">{h['frame_idx']}</div>
                        </div>
                        <div>
                            <div class="meta-label">Time</div>
                            <div class="meta-value">{timestamp}</div>
                        </div>
                        <div>
                            <div class="meta-label"># Objects</div>
                            <div class="meta-value">{num_objects}</div>
                        </div>
                        <div>
                            <div class="meta-label">Scores</div>
                            <div class="meta-value">{scores_str}</div>
                        </div>
                    </div>
                    <div class="small">Saved in: hits/{query_slug}/{session_id}/</div>
                </div>
                """
            )

        grid_html = (
            summary_html
            + '<div class="card"><div class="card-header">Matching frames</div>'
            + '<div class="card-body">'
            + '<div class="results-grid">'
            + "".join(cards)
            + "</div></div></div>"
        )

        message_block = (
            '<div class="message">'
            "Search completed. You can download a CSV summary, or click any image to see it larger. "
            "Red overlays and labels show each detected match and its confidence."
            "</div>"
        )

        return render_page(message_block=message_block, results_block=grid_html)

    except Exception as e:
        error_block = (
            f'<div class="message error">Error while processing video: {e}</div>'
        )
        return render_page(message_block=error_block)
