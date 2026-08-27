"""FastAPI inference service for the CIFAR-10 classifier.

Endpoints:
    GET  /health   -> 200 once the checkpoint is loaded, 503 otherwise
                      (used for k8s liveness/readiness probes)
    POST /predict  -> multipart/form-data image upload, returns class
                      probabilities

Run locally:
    CHECKPOINT_PATH=checkpoints/classifier_v1.pt uvicorn serve:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from dataset import get_transforms
from model import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve")

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

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")

app = FastAPI(title="mlops-pytorch-pipeline serving", version="1.0.0")

# Populated by _load_model() at startup; stays None (and /health reports
# unhealthy) if the checkpoint can't be loaded, so k8s won't route traffic
# to a pod with no model.
_state: dict = {"model": None, "device": None, "eval_transform": None}


def _load_model() -> None:
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        logger.warning("Checkpoint not found at %s; /health will report unhealthy", checkpoint_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=True: this checkpoint only ever contains tensors/ints/
    # floats/strings we wrote ourselves in train.py, so the safer loader
    # (no arbitrary pickle execution) works and is the current best practice.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = get_model(
        architecture=checkpoint.get("architecture", "resnet18"),
        num_classes=checkpoint.get("num_classes", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _state["model"] = model
    _state["device"] = device
    _state["eval_transform"] = get_transforms(train=False)
    logger.info("Loaded checkpoint from %s onto %s", checkpoint_path, device)


@app.on_event("startup")
def on_startup() -> None:
    _load_model()


@app.get("/health")
def health():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - want a clean 400 for bad uploads
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    tensor = _state["eval_transform"](pil_image).unsqueeze(0).to(_state["device"])

    with torch.no_grad():
        logits = _state["model"](tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    predicted_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])

    return JSONResponse(
        {
            "predicted_class": CIFAR10_CLASSES[predicted_idx],
            "predicted_index": predicted_idx,
            "probabilities": {
                CIFAR10_CLASSES[i]: round(p, 6) for i, p in enumerate(probabilities)
            },
        }
    )
