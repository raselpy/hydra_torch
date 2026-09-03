import io
import logging
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.pytorch
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.hydra_torch.data.data_modules import CIFAR10DataModule

logging.basicConfig(level=logging.INFO, force=True)
log = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

REGISTERED_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "hydra_torch_CIFAR10Model")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_transform = None


def _load_model():
    global _model, _transform

    # Model Registry loading requires a database-backed MLflow tracking
    # server (e.g. http://mlflow-server:5000 via docker-compose) — a
    # file:// store does not support the registry. See PLANNER.md
    # section 1j.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. Model Registry loading requires "
            "a database-backed MLflow tracking server — a file:// store "
            "does not support it. Run via docker-compose, which sets this "
            "automatically."
        )
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    log.info(f"Loading model from registry: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
    model.eval()
    model.to(DEVICE)

    # Reuse the EXACT SAME transform CIFAR10DataModule uses in training/eval,
    # so inference-time preprocessing matches what the model actually learned.
    # Note: this transform's normalization constants are actually MNIST's
    # stats, copy-pasted onto CIFAR10DataModule (a real, separate bug — see
    # PLANNER.md). The model was trained with this exact (wrong) normalization
    # baked in, so serving must reuse it as-is rather than "fixing" it here.
    transform = CIFAR10DataModule(batch_size=1).transform

    _model = model
    _transform = transform
    log.info(f"Model loaded from {model_uri} on {DEVICE}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="hydra_torch CIFAR10 classifier", lifespan=lifespan)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}") from e

    input_tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    predicted_idx = int(torch.argmax(probabilities).item())
    return PredictionResponse(
        predicted_class=CIFAR10_CLASSES[predicted_idx],
        confidence=float(probabilities[predicted_idx]),
        probabilities={CIFAR10_CLASSES[i]: float(p) for i, p in enumerate(probabilities)},
    )
