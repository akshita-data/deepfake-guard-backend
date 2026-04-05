import cv2
import numpy as np

# =========================
# IMAGE PREPROCESSING (CNN)
# =========================
def preprocess_image(img):
    # 🔴 Convert BGR → RGB (CRITICAL FIX)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize (must match training size)
    img = cv2.resize(img, (224, 224))

    # Convert to float32 (important for TF)
    img = img.astype("float32")

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Debug (optional)
    print("Preprocessed shape:", img.shape)
    print("Min/Max:", img.min(), img.max())

    return img


# =========================
# FFT FEATURE EXTRACTION
# =========================
def extract_fft_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Magnitude spectrum
    magnitude = np.log(np.abs(fshift) + 1)

    # Normalize properly
    magnitude = magnitude / np.max(magnitude)

    # Score = average intensity
    score = float(np.mean(magnitude))

    # Debug
    print("FFT score:", score)

    return score