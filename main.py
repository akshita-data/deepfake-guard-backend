from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict

app = FastAPI(title="DeepfakeGuard API")

# 🔥 CORS (CRITICAL for frontend + extension)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(predict.router, prefix="")


@app.get("/")
def root():
    return {"message": "DeepfakeGuard API is running"}