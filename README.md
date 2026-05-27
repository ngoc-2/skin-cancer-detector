# 🔬 Skin Cancer Detector

A deep learning model that classifies dermoscopic skin lesion images as **benign** or **malignant**, trained on the [ISIC 2017 dataset](https://challenge.isic-archive.com/data/#2017) using transfer learning with EfficientNet-B0.

> **Live Demo:** [🚀 Try it on Hugging Face Spaces](#) *(update link after deploying)*

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | — |
| AUC-ROC | — |
| Sensitivity | — |
| Specificity | — |

*Fill in after training. These will appear in the notebook.*

---

## 🧠 Model Architecture

- **Backbone:** EfficientNet-B0 (pretrained on ImageNet)
- **Strategy:** Transfer learning — frozen backbone, fine-tuned classifier head
- **Output:** Binary classification (benign / malignant)
- **Loss:** Binary Cross-Entropy with class weighting (handles class imbalance)
- **Optimizer:** Adam with learning rate scheduling

---

## 📁 Project Structure

```
skin-cancer-detector/
├── README.md
├── requirements.txt
├── notebook/
│   └── train.ipynb          ← Full pipeline: data → preprocess → train → evaluate
├── app/
│   ├── app.py               ← Gradio web demo
│   └── model.py             ← Model definition + inference logic
└── assets/
    └── demo.png             ← Screenshot of the web app (add after running)
```

---

## 🗂️ Dataset

This project uses the **ISIC 2017 Skin Lesion Classification** dataset.

1. Go to https://challenge.isic-archive.com/data/#2017
2. Download **Task 3: Lesion Classification** training data + ground truth labels
3. Place files in this structure:

```
data/
├── ISIC-2017_Training_Data/        ← .jpg dermoscopy images
└── ISIC-2017_Training_Part3_GroundTruth.csv
```

The notebook handles all preprocessing from there.

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/skin-cancer-detector.git
cd skin-cancer-detector
pip install -r requirements.txt
```

### 2. Train the model

Open and run `notebook/train.ipynb` top to bottom. It will:
- Load and preprocess the ISIC dataset
- Apply data augmentation
- Fine-tune EfficientNet-B0
- Evaluate and save the best model to `app/best_model.pth`
- Generate a confusion matrix and ROC curve

### 3. Run the demo locally

```bash
python app/app.py
```

Then open http://localhost:7860 in your browser.

### 4. Deploy to Hugging Face Spaces (free)

```bash
# Create a new Space at huggingface.co/new-space (Gradio SDK)
# Push app/ contents + requirements.txt to the Space repo
```

---

## 🔬 Techniques Used

- **Transfer Learning** — leverages ImageNet pretrained weights, drastically reduces training data needed
- **Data Augmentation** — random flips, rotations, color jitter to prevent overfitting
- **Class Imbalance Handling** — weighted loss function (benign samples outnumber malignant ~4:1 in ISIC)
- **Learning Rate Scheduling** — ReduceLROnPlateau for stable convergence
- **Early Stopping** — saves best checkpoint by validation AUC

---

## 📚 References

- Akinrinade, O. & Du, C. (2025). *Skin cancer detection using deep machine learning techniques.* Intelligence-Based Medicine, 11, 100191.
- ISIC Archive: https://www.isic-archive.com
- EfficientNet: Tan & Le, 2019 — https://arxiv.org/abs/1905.11946

---

## ⚠️ Disclaimer

This tool is for **educational purposes only** and is not a medical device. Do not use for clinical diagnosis.
