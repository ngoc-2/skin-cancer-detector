"""
app.py — Gradio web demo for the skin cancer classifier.

Run locally:
    python app/app.py

Deploy to Hugging Face Spaces:
    Push this file + model.py + best_model.pth + requirements.txt
    to a Gradio-type Space repo.
"""

import os
import gradio as gr
from PIL import Image

# ── Try to import model; give a clear error if torch isn't installed ──────────
try:
    import torch
    from model import load_model, predict, LABELS
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
DEVICE = (
    "cuda" if TORCH_AVAILABLE and torch.cuda.is_available()
    else "mps" if TORCH_AVAILABLE and torch.backends.mps.is_available()
    else "cpu"
)


# ── Load model once at startup ────────────────────────────────────────────────

model = None

def load():
    global model
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install requirements first.")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"Model checkpoint not found at {MODEL_PATH}.")
        print("Run the training notebook first to generate best_model.pth.")
        return
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    model = load_model(MODEL_PATH, device=DEVICE)
    print("Model loaded.")

load()


# ── Prediction function ────────────────────────────────────────────────────────

def classify(image: Image.Image):
    """
    Called by Gradio when the user submits an image.
    Returns a label dict for the confidence bar chart.
    """
    if model is None:
        return {"Error: model not loaded. Run train.ipynb first.": 1.0}

    result = predict(model, image, device=DEVICE)
    return result["confidence"]


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Skin Cancer Detector",
    theme=gr.themes.Soft(),
    css="""
        .container { max-width: 860px; margin: auto; }
        .disclaimer { font-size: 0.8em; color: #888; border-top: 1px solid #ddd; padding-top: 8px; margin-top: 16px; }
    """,
) as demo:

    gr.Markdown(
        """
        # 🔬 Skin Cancer Detector
        Upload a **dermoscopic image** of a skin lesion to classify it as **Benign** or **Malignant**.

        > Model: EfficientNet-B0 fine-tuned on [ISIC 2017](https://challenge.isic-archive.com/data/#2017)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Skin Lesion Image",
                height=300,
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(
                label="Prediction",
                num_top_classes=2,
            )

    # Example images (add real dermoscopy images to assets/ folder)
    example_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "examples")
    if os.path.isdir(example_dir):
        examples = [[os.path.join(example_dir, f)] for f in os.listdir(example_dir) if f.endswith(".jpg")]
        if examples:
            gr.Examples(examples=examples, inputs=image_input)

    gr.Markdown(
        """
        ---
        **How to interpret results:**
        - **Benign** — lesion does not appear malignant
        - **Malignant** — lesion shows signs of skin cancer (melanoma or other)

        The confidence score shows how certain the model is. Scores near 50/50 are uncertain.
        """,
        elem_classes="disclaimer"
    )

    gr.Markdown(
        """
        <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational and research purposes only.
        It is <strong>not</strong> a medical device and should never replace professional dermatological evaluation.
        </div>
        """,
    )

    submit_btn.click(fn=classify, inputs=image_input, outputs=label_output)
    image_input.submit(fn=classify, inputs=image_input, outputs=label_output)


if __name__ == "__main__":
    demo.launch(share=False)
