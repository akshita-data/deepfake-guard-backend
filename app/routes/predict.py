from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.services.model import predict_image_from_bytes, predict_image_from_url

router = APIRouter()


# ── Upload file ──────────────────────────────────────
@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    return predict_image_from_bytes(contents)


# ── URL scan ─────────────────────────────────────────
class ImageURL(BaseModel):
    url: str


@router.post("/predict-url")
async def predict_from_url(data: ImageURL):
    return predict_image_from_url(data.url)