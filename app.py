"""
app.py

Gradio web interface for skin cancer probability detection.
Supports file upload and live camera capture (phone or webcam).

Run locally:
    python app/app.py

Then open http://localhost:7860 in a browser, or on a phone via your
local network IP (the terminal will print the address).

Deploy to Hugging Face Spaces:
    Push app.py, model.py, best_model.pth, requirements.txt to a
    Gradio-type Space repo. The interface will be publicly accessible.
"""

import os
import sys

import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    from model import load_model, predict
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Device and model
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")

DEVICE = "cpu"
if TORCH_AVAILABLE:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"

model = None

def _load_model():
    global model
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch not installed. Run: pip install -r requirements.txt")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Checkpoint not found: {MODEL_PATH}")
        print("        Run notebook/train.ipynb to generate best_model.pth")
        return
    print(f"Loading model on {DEVICE}...")
    model = load_model(MODEL_PATH, device=DEVICE)
    print("Model ready.")

_load_model()

# ---------------------------------------------------------------------------
# Prediction callback
# ---------------------------------------------------------------------------

def classify(image: Image.Image):
    """
    Gradio callback. Returns a dict of {class_name: probability} which
    Gradio's Label component renders as a confidence bar chart.
    """
    if image is None:
        return {}
    if model is None:
        return {"Run train.ipynb to generate best_model.pth": 1.0}

    result = predict(model, image, device=DEVICE)
    return result["scores"]

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

INTRO = """
## Skin Cancer Detection

Upload a photo of a skin lesion or use your camera to take one.
The model will return the probability of the lesion being **benign** or **malignant**.

The model was trained on the [ISIC 2017](https://challenge.isic-archive.com/data/#2017)
dermoscopy dataset using transfer learning with EfficientNet-B0.
It learns the same visual features dermatologists use: asymmetry, border irregularity,
color variation, and texture — the clinical ABCD criteria for melanoma assessment.
"""

INTERPRETATION = """
**Reading the result:**

- **Benign** — the lesion does not show strong features of malignancy
- **Malignant** — the lesion shows features associated with skin cancer

A score near 50% means the model is uncertain. Scores above 70% in either
direction indicate higher confidence. This output is a probability, not a diagnosis.
"""

DISCLAIMER = """
---
**Disclaimer:** This tool is for educational and research purposes only.
It is not a medical device. Do not use it for clinical diagnosis.
Consult a qualified dermatologist for any skin concerns.
"""

with gr.Blocks(title="Skin Cancer Detection") as demo:

    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                sources=["upload", "webcam"],
                type="pil",
                label="Skin Lesion Image",
                height=320,
                mirror_webcam=False,
            )
            analyze_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(
                label="Probability",
                num_top_classes=2,
            )
            gr.Markdown(INTERPRETATION)

    gr.Markdown(DISCLAIMER)

    analyze_btn.click(fn=classify, inputs=image_input, outputs=label_output)
    image_input.change(fn=classify, inputs=image_input, outputs=label_output)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible from phone on same network
        server_port=7860,
        share=False,
    )
