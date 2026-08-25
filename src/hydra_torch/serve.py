import io
import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image
from pydantic import BaseModel

# Load-bearing side-effecting import — registers Hydra ConfigStore schemas.
from src.config_schema import setup_config  # noqa: F401
from src.hydra_torch.data_modules import CIFAR10DataModule

log = logging.getLogger(__name__)

# This file lives at src/hydra_torch/serve.py — two levels under the repo
# root, unlike scripts/*.py which are three levels under it.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_THIS_DIR, "..", "..", "configs")

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

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/best.ckpt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_transform = None


def _load_model():
    global _model, _transform

    if not os.path.isfile(CHECKPOINT_PATH):
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Set the CHECKPOINT_PATH "
            f"env var, or train a model first (see README Quickstart)."
        )

    with initialize_config_dir(config_dir=_CONFIG_PATH, version_base=None):
        cfg = compose(config_name="config", overrides=["task=cifar10_classification", "data_module=cifar10"])
        model = instantiate(cfg.task.model)

    # weights_only=False is required: PyTorch Lightning checkpoints contain
    # non-tensor objects (optimizer states, callback states, hyperparameters),
    # not just raw tensors. Since PyTorch 2.6, torch.load defaults to
    # weights_only=True, which cannot unpickle those and would raise
    # UnpicklingError here. Safe to disable since this loads our own
    # checkpoint, not an untrusted third-party file.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["state_dict"]
    model_state_dict = {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(DEVICE)

    # Reuse the EXACT SAME transform CIFAR10DataModule uses in training/eval,
    # so inference-time preprocessing matches what the model actually learned.
    # Note: this transform's normalization constants are actually MNIST's
    # stats, copy-pasted onto CIFAR10DataModule (a real, separate bug — see
    # PLANNER.md). The model was trained with this exact (wrong) normalization
    # baked in, so serving must reuse it as-is rather than "fixing" it here —
    # using different stats at inference than at training would make
    # predictions worse, not better.
    transform = CIFAR10DataModule(batch_size=1).transform

    _model = model
    _transform = transform
    log.info(f"Model loaded from {CHECKPOINT_PATH} on {DEVICE}")


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
    return {"status": "ok", "model_loaded": _model is not None}


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
