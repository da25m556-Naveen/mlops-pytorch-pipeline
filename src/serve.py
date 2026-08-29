import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from model import get_model

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530

model: nn.Module | None = None


def resolve_checkpoint_path() -> Path:
    env_path = os.environ.get("CHECKPOINT_PATH")
    if env_path:
        return Path(env_path)
    default_path = Path("/app/checkpoints/classifier_v1.pt")
    if default_path.exists():
        return default_path
    return Path("checkpoints/classifier_v1.pt")


def load_model() -> nn.Module:
    architecture = os.environ.get("MODEL_ARCHITECTURE", "simple_cnn")
    num_classes = int(os.environ.get("NUM_CLASSES", "10"))
    checkpoint_path = resolve_checkpoint_path()

    net = get_model(architecture=architecture, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return net


def preprocess_image(raw_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(raw_bytes)).convert("L").resize((28, 28))
    pixels = torch.tensor(list(image.tobytes()), dtype=torch.float32)
    pixels = pixels.view(1, 1, 28, 28) / 255.0
    return (pixels - FASHION_MNIST_MEAN) / FASHION_MNIST_STD


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = load_model()
    except (FileNotFoundError, RuntimeError):
        model = None
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_bytes = await image.read()
    try:
        input_tensor = preprocess_image(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1).squeeze(0)

    predicted_idx = int(torch.argmax(probabilities).item())
    return {
        "predicted_class": FASHION_MNIST_CLASSES[predicted_idx],
        "probabilities": {
            FASHION_MNIST_CLASSES[i]: round(float(probabilities[i]), 4)
            for i in range(len(FASHION_MNIST_CLASSES))
        },
    }
