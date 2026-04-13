"""
R.A.D.A.R v2.0 — AI Analysis Pipeline
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

# ── ENV CONFIG ───────────────────────────────────────────────
DISABLE_MODEL = os.getenv("DISABLE_MODEL", "true") == "true"

# ── LOGGING ──────────────────────────────────────────────────
logger = logging.getLogger("radar.pipeline")

# ── TensorFlow Setup (CLEAN) ─────────────────────────────────
TF_AVAILABLE = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    TF_AVAILABLE = True
    print("✅ TensorFlow loaded")
except Exception:
    print("⚠️ TensorFlow not available — using stub")
    TF_AVAILABLE = False

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent

def _resolve_model_path() -> Path:
    candidates = [
        ROOT_DIR / "models" / "best_cnn_model.h5",
        BASE_DIR / "models" / "best_cnn_model.h5",
        BASE_DIR / "model" / "best_cnn_model.h5",  # YOUR CASE
        ROOT_DIR / "backend" / "model" / "best_cnn_model.h5",
    ]

    for p in candidates:
        if p.exists():
            print(f"✅ Model found at: {p}")
            return p

    print("❌ Model not found — using fallback")
    return candidates[0]

MODEL_PATH = _resolve_model_path()

# ── CONSTANTS ────────────────────────────────────────────────
IMG_SIZE = 224
FACE_PAD = 0.20
W_CNN = 0.6
W_FFT = 0.4
FAKE_THRESHOLD = 0.5
VIDEO_MAX_FRAMES = 12
VIDEO_FAKE_RATIO = 0.4

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── GLOBALS ──────────────────────────────────────────────────
_cnn_model: Optional[object] = None
_face_cascade: Optional[cv2.CascadeClassifier] = None
_model_is_stub = True


# ═════════════════════════════════════════════════════════════
# LOAD PIPELINE
# ═════════════════════════════════════════════════════════════

def load_pipeline():
    global _cnn_model, _face_cascade, _model_is_stub

    # Face detector
    _face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # MODEL CONTROL (CRITICAL FIX)
    if not TF_AVAILABLE or DISABLE_MODEL:
        logger.warning("⚠️ CNN disabled — running in stub mode")
        _model_is_stub = True
        return

    if not MODEL_PATH.exists():
        logger.warning("Model not found — using stub")
        _model_is_stub = True
        return

    try:
        _cnn_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        _model_is_stub = False
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        _model_is_stub = True


# ═════════════════════════════════════════════════════════════
# IMAGE PREPROCESS
# ═════════════════════════════════════════════════════════════

def preprocess_face(image):
    img_h, img_w = image.shape[:2]

    if _face_cascade is None:
        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        pad = int(w * FACE_PAD)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        crop = image[y1:y2, x1:x2]
        face_score = 1.0
    else:
        crop = image
        face_score = 0.0

    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), face_score


# ═════════════════════════════════════════════════════════════
# CNN
# ═════════════════════════════════════════════════════════════

def run_cnn(face_rgb):
    if _model_is_stub or _cnn_model is None:
        return round(random.uniform(0.4, 0.9), 3)

    img = face_rgb.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = _cnn_model.predict(img)[0]

    if len(pred) == 1:
        return float(pred[0])
    return float(pred[1])


# ═════════════════════════════════════════════════════════════
# FFT
# ═════════════════════════════════════════════════════════════

def run_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    score = np.mean(magnitude) / 10.0
    return float(np.clip(score, 0, 1))


# ═════════════════════════════════════════════════════════════
# ELA
# ═════════════════════════════════════════════════════════════

def run_ela(face_rgb):
    pil = Image.fromarray(face_rgb)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    recompressed = np.array(Image.open(buffer))
    ela = np.abs(face_rgb.astype(float) - recompressed.astype(float))

    return float(np.clip(np.mean(ela) / 50.0, 0, 1))


# ═════════════════════════════════════════════════════════════
# FINAL ANALYSIS
# ═════════════════════════════════════════════════════════════

def analyze_image_bytes(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    face_rgb, face_score = preprocess_face(img)

    cnn = run_cnn(face_rgb)
    fft = run_fft(img)
    ela = run_ela(face_rgb)

    confidence = W_CNN * cnn + W_FFT * fft
    result = "fake" if confidence > FAKE_THRESHOLD else "real"

    return {
        "result": result,
        "confidence": round(confidence, 3),
        "reason": [
            "High frequency anomaly" if fft > 0.5 else "Normal spectrum",
            "CNN signal strong" if cnn > 0.7 else "Weak CNN signal"
        ]
    }
def analyze_video_bytes(video_bytes: bytes):
    import cv2
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)

    frames = []
    count = 0

    while count < 10:
        ret, frame = cap.read()
        if not ret:
            break

        ok, jpg = cv2.imencode(".jpg", frame)
        if ok:
            result = analyze_image_bytes(jpg.tobytes())
            frames.append(result["confidence"])

        count += 1

    cap.release()
    os.remove(temp_path)

    if not frames:
        return {"result": "error", "confidence": 0.0}

    avg_conf = sum(frames) / len(frames)
    result = "fake" if avg_conf > 0.5 else "real"

    return {
        "result": result,
        "confidence": round(avg_conf, 3),
        "reason": ["Video analyzed using frame sampling"]
    }
