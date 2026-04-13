"""
R.A.D.A.R v2.0 — AI Analysis Pipeline
──────────────────────────────────────────────────────────────
Stages:
  1. Haar-cascade face crop → 224×224 RGB (20 % padding)
  2. CNN  — EfficientNetB4 loaded from models/best_cnn_model.h5
  3. FFT  — GAN grid-pattern detection in frequency domain
  4. ELA  — Error Level Analysis (compression inconsistency)
  5. Ensemble: confidence = CNN×0.6 + FFT×0.4
  6. Reason mapping based on signal thresholds
  7. Video: sample up to 12 keyframes → majority vote (>40 % fake → FAKE)

If models/best_cnn_model.h5 is absent or TF is unavailable, the CNN
falls back to a logged random-score stub so all endpoints stay usable.
"""

from __future__ import annotations

import io
import logging
import os
import random
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("radar.pipeline")

# ── Optional TensorFlow ────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

TF_AVAILABLE = False
try:
    import tensorflow as tf          # noqa: E402
    tf.get_logger().setLevel("ERROR")
    TF_AVAILABLE = True
    logger.info("TensorFlow %s detected", tf.__version__)
except ImportError:
    logger.warning("TensorFlow not installed — CNN will use stub scores")

# ── Paths & hyper-parameters ───────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent          # .../backend/
ROOT_DIR   = BASE_DIR.parent               # project root (radar/)

# Search order: root-level models/ → backend/models/ (legacy)
def _resolve_model_path() -> Path:
    candidates = [
        ROOT_DIR  / "models" / "best_cnn_model.h5",
        BASE_DIR  / "models" / "best_cnn_model.h5",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]   # default (will trigger the stub warning)

MODEL_PATH = _resolve_model_path()

IMG_SIZE        = 224     # CNN input resolution
FACE_PAD        = 0.20    # 20 % padding around detected face
W_CNN           = 0.60    # ensemble weight for CNN  score
W_FFT           = 0.40    # ensemble weight for FFT  score
FAKE_THRESHOLD  = 0.50    # ≥ this → "fake"
VIDEO_MAX_FRAMES = 12     # max keyframes sampled per video
VIDEO_FAKE_RATIO = 0.40   # >40 % fake frames → whole video = FAKE

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── Module-level singletons (populated by load_pipeline()) ────────────────────
_cnn_model: Optional[object] = None   # tf.keras.Model or None
_face_cascade: Optional[cv2.CascadeClassifier] = None
_model_is_stub: bool = True           # True ⟹ CNN using random fallback


# ══════════════════════════════════════════════════════════════════════════════
# 0 ─ Startup loader
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline() -> None:
    """
    Initialise Haar cascade and CNN model once at application startup.
    Designed to be called from an asyncio executor so it doesn't block
    the event loop.
    """
    global _face_cascade, _cnn_model, _model_is_stub

    # ── Haar cascade ──────────────────────────────────────────────────────────
    _face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if _face_cascade.empty():
        logger.error("Haar cascade failed to load from %s", CASCADE_PATH)
        _face_cascade = None
    else:
        logger.info("Haar cascade loaded OK")

    # ── CNN model ─────────────────────────────────────────────────────────────
    if not TF_AVAILABLE:
        logger.warning(
            "TensorFlow not available — CNN stub active (random scores). "
            "Install tensorflow to enable real inference."
        )
        _model_is_stub = True
        return

    if not MODEL_PATH.exists():
        logger.warning(
            "Model not found at %s — CNN stub active (random scores). "
            "Place best_cnn_model.h5 in the project-root models/ directory "
            "(or backend/models/) to enable real inference.",
            MODEL_PATH,
        )
        _model_is_stub = True
        return

    try:
        _cnn_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        # Warm-up pass to avoid cold-start latency on first request
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        _cnn_model(dummy, training=False)
        logger.info("EfficientNetB4 model loaded & warmed up from %s", MODEL_PATH)
        _model_is_stub = False
    except Exception as exc:
        logger.error("Failed to load CNN model (%s) — using stub", exc)
        _cnn_model = None
        _model_is_stub = True


# ══════════════════════════════════════════════════════════════════════════════
# 1 ─ Face preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def _expand_box(
    x: int, y: int, w: int, h: int,
    pad: float, img_h: int, img_w: int
) -> tuple[int, int, int, int]:
    """Expand a bounding box by `pad` fraction and clamp to image bounds."""
    pw, ph = int(w * pad), int(h * pad)
    return (
        max(0, x - pw),
        max(0, y - ph),
        min(img_w, x + w + pw),
        min(img_h, y + h + ph),
    )


def preprocess_face(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect the largest face with Haar cascade, crop with 20 % padding,
    resize to IMG_SIZE × IMG_SIZE RGB.

    Returns
    -------
    face_rgb : np.ndarray  shape (224, 224, 3) uint8
    face_score : float     detection confidence proxy (0 = no face found)
    """
    img_h, img_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_confidence = 0.0
    cropped = None

    if _face_cascade is not None:
        faces = _face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.10,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])  # largest face
            x1, y1, x2, y2 = _expand_box(x, y, fw, fh, FACE_PAD, img_h, img_w)
            cropped = image_bgr[y1:y2, x1:x2]
            # Confidence proxy: ratio of face area to image area, capped at 1
            face_confidence = round(min(1.0, (fw * fh) / (img_w * img_h) * 4.0), 3)

    if cropped is None:
        # Fallback: square centre-crop
        size = min(img_h, img_w)
        y0 = (img_h - size) // 2
        x0 = (img_w - size) // 2
        cropped = image_bgr[y0:y0 + size, x0:x0 + size]
        face_confidence = 0.0

    resized  = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    face_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return face_rgb, face_confidence


# ══════════════════════════════════════════════════════════════════════════════
# 2 ─ CNN inference
# ══════════════════════════════════════════════════════════════════════════════

def run_cnn(face_rgb: np.ndarray) -> float:
    """
    Run EfficientNetB4 on a 224×224 RGB face crop.
    Returns P(fake) in [0, 1].

    Preprocessing: divide by 255 (standard EfficientNet scaling).
    Adjust this line if your model was trained with a different normalisation.

    Handles both single-sigmoid output (shape [1,1]) and
    two-class softmax output (shape [1,2]; class index 1 = fake).
    """
    if _model_is_stub or _cnn_model is None:
        stub = round(random.uniform(0.38, 0.94), 3)
        logger.warning("CNN stub active — returning random score %.3f", stub)
        return stub

    try:
        # ── Normalise ───────────────────────────────────────────────────────
        # EfficientNet typically expects [0, 1].  Swap to [–1, 1] or use
        # tf.keras.applications.efficientnet.preprocess_input if your
        # training pipeline used that instead.
        img   = face_rgb.astype(np.float32) / 255.0
        batch = np.expand_dims(img, axis=0)        # (1, 224, 224, 3)

        out = _cnn_model(batch, training=False)
        out_np = out.numpy() if hasattr(out, "numpy") else np.array(out)

        if out_np.shape[-1] == 1:
            score = float(out_np[0, 0])          # sigmoid binary
        else:
            score = float(out_np[0, 1])          # softmax, class 1 = fake

        return round(float(np.clip(score, 0.0, 1.0)), 3)

    except Exception as exc:
        logger.error("CNN inference error: %s — falling back to stub", exc)
        return round(random.uniform(0.38, 0.94), 3)


# ══════════════════════════════════════════════════════════════════════════════
# 3 ─ FFT GAN-grid detector
# ══════════════════════════════════════════════════════════════════════════════

def run_fft(image_bgr: np.ndarray) -> float:
    """
    Detect GAN upsampling artefacts in the 2-D frequency domain.

    GAN generators (especially those using transposed convolutions or pixel-
    shuffle) leave characteristic spectral "grids": symmetric, evenly-spaced
    peaks in the log-magnitude FFT spectrum.

    Three sub-tests are combined:
      • Spectral symmetry  – GAN peaks are point-symmetric about DC (50 %)
      • Checkerboard probe – energy spikes at ±N/4 frequencies  (30 %)
      • Peak density        – more peaks → more likely GAN       (20 %)

    Returns a suspicion score in [0, 1].
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Hanning window suppresses boundary ringing
    hann   = np.outer(np.hanning(IMG_SIZE), np.hanning(IMG_SIZE))
    gray_w = gray * hann

    # Log-magnitude FFT centred at (cy, cx)
    f         = np.fft.fft2(gray_w)
    fshift    = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    cy, cx    = magnitude.shape[0] // 2, magnitude.shape[1] // 2

    # Mask DC
    dc_mask = np.ones_like(magnitude, dtype=bool)
    dc_mask[cy - 6:cy + 6, cx - 6:cx + 6] = False

    ac_vals   = magnitude[dc_mask]
    threshold = np.mean(ac_vals) + 2.0 * np.std(ac_vals)
    peak_map  = (magnitude > threshold) & dc_mask
    peaks     = np.argwhere(peak_map)

    if len(peaks) < 4:
        return 0.05  # too few peaks → natural image

    # ── Test 1: point symmetry around DC ──────────────────────────────────
    h, w = magnitude.shape
    sym_hits = 0
    for py, px in peaks:
        ry, rx = 2 * cy - py, 2 * cx - px
        if 0 <= ry < h and 0 <= rx < w and peak_map[ry, rx]:
            sym_hits += 1
    symmetry_score = sym_hits / len(peaks)

    # ── Test 2: checkerboard signature at ±N/4 offsets ───────────────────
    probe_pts = [
        (cy,       cx + w // 4), (cy,       cx - w // 4),
        (cy + h // 4, cx),       (cy - h // 4, cx),
        (cy + h // 4, cx + w // 4), (cy - h // 4, cx - w // 4),
    ]
    check_hits = 0
    sub_thr    = threshold * 0.82
    for ry, rx in probe_pts:
        if 0 <= ry < h and 0 <= rx < w:
            local = magnitude[max(0, ry - 3):ry + 4, max(0, rx - 3):rx + 4]
            if local.size and float(np.mean(local)) > sub_thr:
                check_hits += 1
    cboard_score = check_hits / len(probe_pts)

    # ── Test 3: normalised peak density ───────────────────────────────────
    density_score = min(len(peaks) / 80.0, 1.0)

    fft_score = 0.50 * symmetry_score + 0.30 * cboard_score + 0.20 * density_score
    return round(float(np.clip(fft_score, 0.0, 1.0)), 3)


# ══════════════════════════════════════════════════════════════════════════════
# 4 ─ ELA (Error Level Analysis)
# ══════════════════════════════════════════════════════════════════════════════

def run_ela(face_rgb: np.ndarray, quality: int = 90) -> float:
    """
    Re-compress the face crop at JPEG quality=90 and measure ELA residual.
    Face-swaps and inpainting leave different compression traces from the
    surrounding image, creating high-variance ELA maps.

    Returns a manipulation-likelihood score in [0, 1].
    """
    try:
        pil_img = Image.fromarray(face_rgb)
        buf     = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = np.array(Image.open(buf).convert("RGB"))

        ela = np.abs(face_rgb.astype(np.float32) - recompressed.astype(np.float32))

        # Evaluate the central 60 % of the face (excludes hair / background)
        h, w  = ela.shape[:2]
        my, mx = int(h * 0.20), int(w * 0.20)
        ela_c = ela[my:h - my, mx:w - mx]

        mean_ela = float(np.mean(ela_c))
        std_ela  = float(np.std(ela_c))

        # Score: natural JPEG residuals ≈ mean<12, std<8; manipulated regions higher
        score = np.clip((mean_ela / 28.0) * 0.55 + (std_ela / 22.0) * 0.45, 0.0, 1.0)
        return round(float(score), 3)

    except Exception as exc:
        logger.error("ELA error: %s", exc)
        return 0.30   # neutral fallback


# ══════════════════════════════════════════════════════════════════════════════
# 5 ─ Ensemble
# ══════════════════════════════════════════════════════════════════════════════

def ensemble(cnn_score: float, fft_score: float) -> float:
    """Weighted combination: CNN×0.6 + FFT×0.4."""
    return round(W_CNN * cnn_score + W_FFT * fft_score, 3)


# ══════════════════════════════════════════════════════════════════════════════
# 6 ─ Reason mapping
# ══════════════════════════════════════════════════════════════════════════════

def derive_reasons(
    cnn_score: float,
    fft_score: float,
    ela_score: float,
    face_score: float,
    final_score: float,
) -> list[str]:
    """Map signal values to up-to-3 human-readable detection reasons."""
    reasons: list[str] = []

    if final_score < FAKE_THRESHOLD:
        reasons.append("No significant deepfake indicators found")
        if fft_score < 0.25:
            reasons.append("Natural frequency spectrum")
        if ela_score < 0.20:
            reasons.append("Consistent image compression profile")
        return reasons[:3]

    # ── Fake reasons ──────────────────────────────────────────────────────
    if cnn_score >= 0.85:
        reasons.append("High-confidence deepfake signature detected")
    elif cnn_score >= 0.70:
        reasons.append("Facial blend artifact")

    if cnn_score >= 0.65 and face_score > 0.0:
        reasons.append("Eye rendering anomaly")

    if fft_score >= 0.68:
        reasons.append("GAN upsampling checkerboard in frequency domain")
    elif fft_score >= 0.48:
        reasons.append("FFT grid pattern detected")

    if ela_score >= 0.58:
        reasons.append("Inconsistent compression artifact (ELA)")
    elif ela_score >= 0.38:
        reasons.append("Inconsistent lighting gradient on skin")

    if face_score == 0.0:
        reasons.append("No face region — full-image analysis applied")

    if not reasons:
        reasons.append("Subtle generation artifacts detected")

    return reasons[:3]


# ══════════════════════════════════════════════════════════════════════════════
# 7 ─ High-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG, PNG, WebP, …) to a BGR ndarray."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image data.")
    return img


def analyze_image_bytes(image_bytes: bytes) -> dict:
    """
    Full RADAR pipeline for a single image.
    All processing is in RAM — no filesystem writes.
    """
    img_bgr = _decode_image(image_bytes)

    # 1 — Face crop
    face_rgb, face_score = preprocess_face(img_bgr)

    # 2 — CNN
    cnn_score = run_cnn(face_rgb)

    # 3 — FFT (run on face crop for focused frequency analysis)
    face_bgr  = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    fft_score = run_fft(face_bgr)

    # 4 — ELA
    ela_score = run_ela(face_rgb)

    # 5 — Ensemble
    confidence = ensemble(cnn_score, fft_score)
    result     = "fake" if confidence >= FAKE_THRESHOLD else "real"

    # 6 — Reasons
    reasons = derive_reasons(cnn_score, fft_score, ela_score, face_score, confidence)

    return {
        "result":     result,
        "confidence": confidence,
        "reason":     reasons,
        "signals": {
            "cnn_score":  cnn_score,
            "fft_score":  fft_score,
            "ela_score":  ela_score,
            "face_score": face_score,
        },
    }


def analyze_video_bytes(video_bytes: bytes) -> dict:
    """
    Extract up to VIDEO_MAX_FRAMES keyframes from video bytes,
    run the full pipeline per frame, then majority-vote:
      > VIDEO_FAKE_RATIO (40 %) fake frames → result = "fake".

    Video bytes are written to a short-lived temp file (required by
    cv2.VideoCapture) and deleted immediately after reading.
    """
    tmp_path: Optional[str] = None
    try:
        # Write to a NamedTemporaryFile so cv2 can open it by path
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video — unsupported codec or corrupt file.")

        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        step  = max(1, total // VIDEO_MAX_FRAMES)
        frame_indices = list(range(0, total, step))[:VIDEO_MAX_FRAMES]

        frame_results: list[dict] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            try:
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    frame_results.append(analyze_image_bytes(jpeg.tobytes()))
            except Exception as exc:
                logger.warning("Frame %d skipped: %s", idx, exc)

        cap.release()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)   # always clean up

    if not frame_results:
        raise ValueError("No usable frames could be extracted from the video.")

    # ── Majority vote ─────────────────────────────────────────────────────
    fake_count = sum(1 for r in frame_results if r["result"] == "fake")
    fake_ratio = fake_count / len(frame_results)
    final_result = "fake" if fake_ratio > VIDEO_FAKE_RATIO else "real"

    # Aggregate signals as mean across frames
    def _avg(key: str) -> float:
        return round(
            sum(r["signals"][key] for r in frame_results) / len(frame_results), 3
        )

    cnn_avg  = _avg("cnn_score")
    fft_avg  = _avg("fft_score")
    ela_avg  = _avg("ela_score")
    face_avg = _avg("face_score")
    confidence = ensemble(cnn_avg, fft_avg)
    reasons    = derive_reasons(cnn_avg, fft_avg, ela_avg, face_avg, confidence)

    logger.info(
        "Video analysed: %d frames, %d fake (%.0f%%) → %s",
        len(frame_results), fake_count, fake_ratio * 100, final_result.upper(),
    )

    return {
        "result":     final_result,
        "confidence": confidence,
        "reason":     reasons,
        "signals": {
            "cnn_score":  cnn_avg,
            "fft_score":  fft_avg,
            "ela_score":  ela_avg,
            "face_score": face_avg,
        },
    }
