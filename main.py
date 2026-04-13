"""
R.A.D.A.R v2.0 — Real-time AI Detection & Analysis Runtime
FastAPI Backend — Real AI pipeline (Phase 2)
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from pipeline import analyze_image_bytes, analyze_video_bytes, load_pipeline

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("radar.api")

# ── Lifespan (model loading at startup) ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⚡ Loading R.A.D.A.R pipeline…")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_pipeline)   # non-blocking
    logger.info("✅ R.A.D.A.R v2.0 online — all systems ready")
    yield
    logger.info("R.A.D.A.R shutting down")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="R.A.D.A.R v2.0 API",
    description=(
        "Real-time AI Detection & Analysis Runtime — "
        "deepfake detection via CNN + FFT ensemble."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class UrlRequest(BaseModel):
    url: HttpUrl


class PredictionResponse(BaseModel):
    result: str
    confidence: float
    reason: list[str]
    signals: dict


# ── Helpers ────────────────────────────────────────────────────────────────────
ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp",
}
ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/mpeg", "video/webm",
    "video/quicktime", "video/x-msvideo", "video/x-matroska",
}


async def _run_pipeline(fn, *args):
    """Execute a synchronous pipeline function in the default thread-pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
async def root():
    """Returns API online status."""
    return {"status": "online"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict from image upload",
)
async def predict_image(file: UploadFile = File(...)):
    """
    Accept an image file upload and return a deepfake prediction.
    Pipeline: Haar face crop → EfficientNetB4 → FFT → ELA → ensemble.
    File is processed in RAM — nothing persisted to disk.
    Target latency: < 2 s.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type '{file.content_type}'. "
                f"Accepted: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}"
            ),
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return await _run_pipeline(analyze_image_bytes, contents)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Image analysis failed")
        raise HTTPException(status_code=500, detail="Internal analysis error.") from exc


@app.post(
    "/predict-video",
    response_model=PredictionResponse,
    summary="Predict from video upload",
)
async def predict_video(file: UploadFile = File(...)):
    """
    Accept a video file upload and return a deepfake prediction.
    Pipeline: sample keyframes → per-frame CNN+FFT → majority vote (>40 % fake = FAKE).
    Target latency: < 3 s.
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type '{file.content_type}'. "
                f"Accepted: {', '.join(sorted(ALLOWED_VIDEO_TYPES))}"
            ),
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return await _run_pipeline(analyze_video_bytes, contents)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Video analysis failed")
        raise HTTPException(status_code=500, detail="Internal analysis error.") from exc


@app.post(
    "/predict-url",
    response_model=PredictionResponse,
    summary="Predict from remote image URL",
)
async def predict_url(body: UrlRequest):
    """
    Fetch an image from a remote URL server-side and return a deepfake prediction.
    Nothing is persisted to disk.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(str(body.url))
            response.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="Request to remote URL timed out.")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Remote URL returned HTTP {exc.response.status_code}.",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}")

    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"URL did not return an image. Got content-type: '{content_type}'.",
        )

    image_bytes = response.content
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Fetched image is empty.")

    try:
        return await _run_pipeline(analyze_image_bytes, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("URL image analysis failed")
        raise HTTPException(status_code=500, detail="Internal analysis error.") from exc
