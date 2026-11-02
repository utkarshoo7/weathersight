"""
Weather Classification Inference Module
---------------------------------------
Performs single-image weather classification using a fine-tuned ResNet-18.

CUDA is used automatically when available; CPU remains the fallback.
"""

from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = Path("models/weather/weather_resnet18_ft.pth")

# Prefer the NVIDIA GPU when PyTorch can access CUDA.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MUST match the class order used when the checkpoint was trained.
RAW_CLASSES = ["clear", "rainy", "snowy", "overcast", "night"]


# =========================
# MODEL INITIALIZATION
# =========================

_model = None


def _load_model() -> torch.nn.Module:
    """
    Load the fine-tuned ResNet-18 checkpoint and cache it on the selected device.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Weather model not found: {MODEL_PATH}")

    # weights=None prevents torchvision from downloading pretrained ImageNet weights.
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(RAW_CLASSES))

    # The checkpoint is expected to be a plain state_dict of tensors.
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def _get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        _model = _load_model()
    return _model


# =========================
# PREPROCESSING
# =========================

_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# =========================
# CANONICAL MAPPING
# =========================

def _canonicalize_label(raw_label: str) -> str:
    """
    Map raw classifier outputs into the current downstream label set.

    Note: the existing project currently maps both 'overcast' and 'night'
    to 'clear'. This preserves the current behavior; the class semantics
    should be reviewed separately before changing the mapping.
    """
    if raw_label in ("overcast", "night"):
        return "clear"
    return raw_label


# =========================
# PUBLIC API
# =========================

def predict_weather(image_path: str) -> dict:
    """
    Predict a weather condition for one image.

    Returns:
        {
            "label": canonical weather label,
            "raw_label": original classifier output,
            "confidence": softmax confidence
        }
    """
    model = _get_model()

    with Image.open(image_path) as image:
        img = image.convert("RGB")

    x = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    conf, idx = probs.max(dim=1)
    raw_label = RAW_CLASSES[idx.item()]
    final_label = _canonicalize_label(raw_label)

    return {
        "label": final_label,
        "raw_label": raw_label,
        "confidence": float(conf.item()),
    }
