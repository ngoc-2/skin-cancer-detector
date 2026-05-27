"""
model.py — Model definition and inference logic for skin cancer classifier.
Used by both the training notebook and the Gradio web app.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ── Constants ────────────────────────────────────────────────────────────────

LABELS = ["Benign", "Malignant"]
IMAGE_SIZE = 224

# ImageNet normalization stats (required for pretrained EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transforms ───────────────────────────────────────────────────────────────

# Used during inference (no augmentation)
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Used during training (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Used during validation (no augmentation, but same resize)
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Build an EfficientNet-B0 model with a custom classification head.

    Args:
        num_classes:      Number of output classes (2 for binary: benign/malignant)
        freeze_backbone:  If True, freeze all layers except the classifier head.
                          Set to False for full fine-tuning after initial training.

    Returns:
        A PyTorch nn.Module ready for training or inference.
    """
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze backbone weights so we only train the head initially
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the default classifier head
    # EfficientNet-B0's final feature map has 1280 channels
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )

    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """
    Unfreeze the backbone for full fine-tuning (phase 2 of training).
    Call this after the classifier head has converged.
    """
    for param in model.features.parameters():
        param.requires_grad = True
    return model


# ── Inference ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a saved model checkpoint for inference.

    Args:
        checkpoint_path: Path to the .pth file saved during training.
        device:          'cuda', 'mps', or 'cpu'

    Returns:
        Model in eval mode, moved to the specified device.
    """
    model = build_model(num_classes=2, freeze_backbone=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model: nn.Module, image: Image.Image, device: str = "cpu") -> dict:
    """
    Run inference on a single PIL image.

    Args:
        model:  Loaded model in eval mode.
        image:  PIL Image (RGB).
        device: Device string.

    Returns:
        Dictionary with predicted label and confidence scores.
        Example: {"label": "Malignant", "Benign": 0.12, "Malignant": 0.88}
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    predicted_idx = probs.index(max(probs))
    return {
        "label": LABELS[predicted_idx],
        "confidence": {label: round(prob, 4) for label, prob in zip(LABELS, probs)},
    }
