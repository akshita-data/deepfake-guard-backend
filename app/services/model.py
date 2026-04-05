import numpy as np
import cv2
import requests
from tensorflow.keras.models import load_model
from app.utils.preprocess import preprocess_image, extract_fft_features

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_cnn_model.h5")

# 🔥 LOAD MODEL SAFELY
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None


# =========================
# IMAGE FROM FILE
# =========================
def predict_image_from_bytes(image_bytes):
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return error_response("Invalid image file")

        return run_pipeline(img)

    except Exception as e:
        return error_response(str(e))


# =========================
# IMAGE FROM URL
# =========================
def predict_image_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        img_bytes = response.content

        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return error_response("Invalid image URL")

        return run_pipeline(img)

    except Exception as e:
        return error_response(str(e))


# =========================
# CORE PIPELINE (CNN + FFT)
# =========================
def run_pipeline(img):
    if model is None:
        return error_response("Model not loaded")

    try:
        # =========================
        # PREPROCESS
        # =========================
        processed = preprocess_image(img)

        # =========================
        # CNN PREDICTION
        # =========================
        pred = model.predict(processed)[0][0]
        pred = float(pred)

        # =========================
        # FFT FEATURE
        # =========================
        fft_score = extract_fft_features(img)

        # =========================
        # DEBUG (VERY IMPORTANT)
        # =========================
        print("\n========== DEBUG ==========")
        print("RAW CNN OUTPUT:", pred)
        print("FFT SCORE:", fft_score)
        print("Processed shape:", processed.shape)
        print("Min/Max:", processed.min(), processed.max())

        # =========================
        # 🔥 CALIBRATION LOGIC
        # =========================
        # 🔥 config
        MODEL_REVERSED = True
        THRESHOLD = 0.6

        # calibration
        cnn_score = 1 - pred if MODEL_REVERSED else pred

        # weighted fusion
        final_score = (0.7 * cnn_score) + (0.3 * fft_score)

        # decision
        result = "fake" if final_score > THRESHOLD else "real"
        
        # =========================
        # REASONS (STRONG EXPLAINABILITY)
        # =========================
        reasons = []

        if cnn_score > 0.7 and result == "fake":
            reasons.append("CNN detected deepfake-like facial patterns")

        if fft_score > 0.6:
            reasons.append("Abnormal high-frequency signals detected")

        if not reasons:
            reasons.append("No strong manipulation signals detected")

        # =========================
        # FINAL DEBUG
        # =========================
        print("FINAL SCORE:", final_score)
        print("FINAL RESULT:", result)
        print("===========================\n")

        # =========================
        # RESPONSE
        # =========================
        return {
            "result": result,
            "confidence": round(float(final_score), 4),
            "reason": reasons,
            "signals": {
                "cnn_score": round(float(cnn_score), 4),
                "fft_score": round(float(fft_score), 4)
            }
        }

    except Exception as e:
        return error_response(str(e))

# =========================
# ERROR RESPONSE FORMAT
# =========================
def error_response(message):
    return {
        "result": "error",
        "confidence": 0,
        "reason": [message]
    }