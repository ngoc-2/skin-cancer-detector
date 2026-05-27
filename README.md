# Skin Cancer Detection

A deep learning application that analyzes a photo of a skin lesion and returns the probability of it being malignant. Submit an image from your phone camera or upload a file — the model outputs a benign/malignant classification with confidence scores for each class.

Live demo: [Hugging Face Spaces](#) — update after deploying

---

## Overview

Melanoma accounts for less than 4% of skin cancer cases but is responsible for approximately 75% of skin cancer deaths. It is only curable when caught early. Traditional diagnosis relies on dermoscopy and biopsy — procedures that are painful, slow, and inaccessible in rural or underserved areas.

This project demonstrates how a convolutional neural network can assist in early detection by learning the same visual features that dermatologists use: asymmetry, border irregularity, color variation, and diameter (the clinical ABCD criteria). The model is trained on dermoscopic images from the ISIC 2017 benchmark dataset and returns a probability score rather than a hard binary decision, giving the user a sense of confidence in the result.

---

## How It Works

1. You submit a photo of a skin lesion via file upload or phone camera
2. The image is resized and normalized to match the format the model was trained on
3. The image passes through an EfficientNet-B0 convolutional neural network
4. The final layer outputs a probability for each class: Benign and Malignant
5. The interface displays both probabilities as a confidence bar

---

## Model

**Architecture:** EfficientNet-B0 with a custom classification head, pretrained on ImageNet and fine-tuned on ISIC 2017 dermoscopic images.

**Why CNN over ANN for this task:**
Standard ANNs treat each pixel as an independent input, which loses all spatial information — the relationships between neighboring pixels that define shapes, textures, and edges. CNNs use convolutional filters that slide across the image, preserving spatial structure and learning hierarchical features: edges at early layers, textures at middle layers, and high-level patterns like irregular borders or uneven pigmentation at deeper layers. Shah et al. (2023) found that CNN consistently outperforms ANN and other classifiers for skin cancer image classification tasks.

**Why EfficientNet-B0:**
EfficientNet scales network depth, width, and input resolution together using a compound coefficient. This gives it better accuracy per parameter than older architectures like VGG or ResNet. The B0 variant is small enough to run on CPU while maintaining competitive AUC on medical imaging benchmarks.

**Classification head:**
The pretrained backbone outputs a 1280-dimensional feature vector. A custom head maps this to the binary output:

```
Dropout(0.4)
Linear(1280 -> 512) + ReLU
Dropout(0.2)
Linear(512 -> 128) + ReLU
Linear(128 -> 2)
Softmax
```

**Training strategy — two phases:**

Phase 1 freezes the pretrained backbone and trains only the classification head for 10 epochs at lr=1e-3. This lets the head converge on the pretrained features without corrupting them. Phase 2 unfreezes all layers and fine-tunes end-to-end at lr=1e-4 for 10 more epochs using cosine annealing. The lower learning rate in phase 2 preserves the pretrained weights while allowing the backbone to adapt to dermoscopic image characteristics.

**Class imbalance handling:**
The ISIC 2017 dataset is roughly 4:1 benign to malignant. A WeightedRandomSampler is used during training so each batch contains a balanced representation. This prevents the model from defaulting to always predicting benign.

**Loss:** Cross-entropy  
**Optimizer:** Adam  
**Scheduler:** ReduceLROnPlateau (phase 1), CosineAnnealingLR (phase 2)

---

## Results

| Metric      | Score |
|-------------|-------|
| Accuracy    | 80.5% |
| AUC-ROC     | 0.8213 |
| Sensitivity | 63% |
| Specificity | 85% |

Fill in after running the training notebook. Benchmarks from literature for comparison: CNN-based models on ISIC data consistently achieve accuracy in the 85–99% range depending on architecture and dataset size (Shah et al., 2023; Akinrinade & Du, 2025). A multilayer perceptron trained with the ABCD rule achieved 96.9% accuracy (Kanimozhi & Murthi, 2016). The target for this project is AUC > 0.85 on the ISIC 2017 validation set.

---

## Dataset

This project uses the ISIC 2017 Skin Lesion Classification dataset, the established benchmark for automated melanoma detection research.

Download from: https://challenge.isic-archive.com/data/#2017

Download Task 3 — Lesion Classification. You need the training images folder and the ground truth CSV. Place them here:

```
data/
├── ISIC-2017_Training_Data/
│   └── *.jpg
└── ISIC-2017_Training_Part3_GroundTruth.csv
```

The CSV contains columns: `image_id`, `melanoma`, `seborrheic_keratosis`. This project uses `melanoma` as the binary label — 1 = Malignant, 0 = Benign.

---

## Project Structure

```
skin-cancer-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── train.ipynb          full pipeline: data -> preprocess -> train -> evaluate
├── app/
│   ├── app.py               Gradio web interface (file upload + camera)
│   └── model.py             model definition and inference logic
└── assets/
    └── evaluation.png       confusion matrix and ROC curve generated by notebook
```

---

## Quickstart

### Install

```bash
git clone https://github.com/ngoc-2/skin-cancer-detector.git
cd skin-cancer-detector
pip install -r requirements.txt
```

### Train

Open `notebook/train.ipynb` and run all cells top to bottom. The notebook will:

- Load and split the dataset 80/20, stratified by class
- Visualize class distribution and sample images
- Apply data augmentation to the training set
- Run phase 1 training (frozen backbone, 10 epochs)
- Run phase 2 fine-tuning (full model, 10 epochs)
- Generate a confusion matrix and ROC curve saved to `assets/evaluation.png`
- Save the best checkpoint to `app/best_model.pth`

Training on CPU will take several hours. Google Colab with a free GPU is recommended.

### Run the app

```bash
python app/app.py
```

Open http://localhost:7860. Upload a dermoscopic image or use your webcam. The app also runs on your local network so you can open it from a phone on the same WiFi.

### Deploy

1. Create a free Space at https://huggingface.co/new-space (select Gradio SDK)
2. Push `app/app.py`, `app/model.py`, `app/best_model.pth`, and `requirements.txt` to the Space repo
3. Update the live demo link at the top of this README

---

## References

- Shah, A., Shah, M., et al. (2023). A comprehensive study on skin cancer detection using artificial neural network (ANN) and convolutional neural network (CNN). Clinical eHealth, 6, 76–84.
- Akinrinade, O. & Du, C. (2025). Skin cancer detection using deep machine learning techniques. Intelligence-Based Medicine, 11, 100191.
- Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
- ISIC Archive: https://www.isic-archive.com

---

## Disclaimer

This tool is for educational and research purposes only. It is not a medical device and should not be used for clinical diagnosis. Always consult a qualified dermatologist.
