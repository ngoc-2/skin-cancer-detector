"""
model.py

EfficientNet-B0 classifier for binary skin lesion classification.
Imported by both train.ipynb and app.py.

Architecture rationale:
    CNNs outperform standard ANNs on image tasks because convolutional filters
    preserve spatial relationships between pixels, learning hierarchical features
    (edges -> textures -> shapes) rather than treating each pixel independently.
    EfficientNet-B0 uses compound scaling across depth, width, and resolution,
    giving strong accuracy with a small parameter count suitable for real-time inference.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES   = ["Benign", "Malignant"]
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# Training: augmentation + normalization
# Augmentation simulates real-world variation: different camera angles,
# lighting conditions, and image orientations.
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation / inference: resize and normalize only — no randomness
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

inference_transform = val_transform


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model(freeze_backbone: bool = True) -> nn.Module:
    """
    Build an EfficientNet-B0 model with a custom binary classification head.

    The pretrained backbone provides feature extraction learned from 1.2M
    ImageNet images. The custom head is designed for binary medical classification
    with dropout regularization to reduce overfitting on the small ISIC dataset.

    Classifier head architecture:
        Dropout(0.4)
        Linear(1280 -> 512) + ReLU
        Dropout(0.2)
        Linear(512 -> 128) + ReLU
        Linear(128 -> 2)

    Args:
        freeze_backbone: True for phase 1 training (head only).
                         False for phase 2 full fine-tuning.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, len(CLASS_NAMES)),
    )

    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """
    Unfreeze all backbone parameters for full fine-tuning (phase 2).
    Use a lower learning rate after calling this to avoid destroying
    pretrained feature representations.
    """
    for param in model.features.parameters():
        param.requires_grad = True
    return model


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a saved model checkpoint for inference.

    Args:
        checkpoint_path: Path to .pth file saved during training.
        device:          'cuda', 'mps', or 'cpu'.

    Returns:
        Model in eval mode on the specified device.
    """
    model = build_model(freeze_backbone=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict(model: nn.Module, image: Image.Image, device: str = "cpu") -> dict:
    """
    Run inference on a single PIL image.

    The model outputs raw logits which are passed through softmax to produce
    calibrated probabilities. The probability for each class sums to 1.0.

    Args:
        model:  Model in eval mode.
        image:  PIL Image (any mode — converted to RGB internally).
        device: Device string.

    Returns:
        {
            "label":       "Malignant",
            "probability": 0.83,
            "scores": {"Benign": 0.17, "Malignant": 0.83}
        }
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    scores        = {name: round(p, 4) for name, p in zip(CLASS_NAMES, probs)}
    predicted_idx = probs.index(max(probs))

    return {
        "label":       CLASS_NAMES[predicted_idx],
        "probability": round(max(probs), 4),
        "scores":      scores,
    }
