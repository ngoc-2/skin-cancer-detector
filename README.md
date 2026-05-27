# Skin Cancer Detection

A deep learning web application that analyzes dermoscopic images of skin lesions and returns the probability of malignancy. Upload a photo or use your camera — the model classifies the lesion as benign or malignant with a confidence score for each class.

**Live demo:** [huggingface.co/spaces/ngoc2/skin-cancer-detector](https://huggingface.co/spaces/ngoc2/skin-cancer-detector)  

---

## Summary

Melanoma accounts for less than 4% of skin cancer cases but roughly 75% of skin cancer deaths. It is only curable when caught early, but traditional diagnosis through biopsy is slow, painful, and inaccessible in many areas.

This project applies **deep learning** — a subfield of machine learning where artificial neural networks with many layers automatically learn patterns from data — to the problem of skin cancer detection. Rather than manually engineering features like color histograms or texture descriptors, the neural network learns directly from raw pixel data which visual patterns distinguish malignant from benign lesions.

The model is built on a **Convolutional Neural Network (CNN)**, an architecture specifically designed for image analysis. CNNs are organized into layers of filters that scan across an image, each layer learning increasingly abstract features: the first layers detect low-level edges and color gradients, middle layers recognize textures and shapes, and the deepest layers identify high-level patterns like irregular borders, asymmetric growth, and uneven pigmentation — the same visual criteria dermatologists use clinically (the ABCD rule: Asymmetry, Border, Color, Diameter).

Rather than training a CNN from scratch — which would require millions of images — this project uses **transfer learning**: starting from EfficientNet-B0, a CNN pretrained on 1.2 million ImageNet photos. The network has already learned general visual features; we repurpose those features for dermoscopy by replacing the final classification layer and fine-tuning the entire network on the ISIC 2017 dataset of 2000 labeled skin lesion images.

The trained model is served through a **Gradio** web interface deployed on Hugging Face Spaces, making it accessible from any browser or phone camera without any local setup.

---

## Results

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 80.5%  |
| AUC-ROC     | 0.8213 |
| Sensitivity | 63%    |
| Specificity | 85%    |

Trained on 2000 dermoscopic images from the ISIC 2017 dataset (1600 train, 400 validation). AUC of 0.82 means the model correctly ranks a randomly chosen malignant lesion above a randomly chosen benign lesion 82% of the time — a standard metric for medical classification tasks where class imbalance makes raw accuracy misleading.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Model | EfficientNet-B0 (PyTorch) |
| Training | Transfer learning, two-phase fine-tuning |
| Data | ISIC 2017 dermoscopy dataset |
| Web app | Gradio |
| Deployment | Hugging Face Spaces |
| Language | Python 3.9 |

---

## Machine Learning

### Neural Networks

A neural network is a computational model loosely inspired by the brain. It consists of layers of interconnected nodes (neurons), where each connection has a learned weight. During training, the network sees labeled examples (images + correct diagnoses), makes predictions, measures how wrong it was using a loss function, and adjusts every weight slightly via **backpropagation** — repeatedly propagating the error signal backwards through the network and using **gradient descent** to update weights in the direction that reduces error. After thousands of iterations over the training data, the network converges to weights that generalize well to new images it has never seen.

### Why CNN over Standard ANN

A standard **Artificial Neural Network (ANN)** flattens an image into a 1D vector and connects every pixel to every neuron. This destroys spatial relationships — a pixel's meaning depends entirely on its neighbors, and an ANN has no mechanism to exploit that structure. It also requires an enormous number of parameters for even a small image.

A **Convolutional Neural Network (CNN)** solves this with two key operations:

**Convolution** — small learnable filters (e.g. 3×3) slide across the image, computing a dot product at each position. Each filter learns to detect a specific local pattern regardless of where it appears in the image (translation invariance). Stacking many filters in a layer produces a feature map capturing different aspects of the image.

**Pooling** — downsamples feature maps by taking the maximum or average value in small regions, reducing spatial dimensions while retaining the most important activations and providing robustness to small shifts and distortions.

By stacking convolution and pooling layers, CNNs build a hierarchy of features from simple to complex, which is exactly what is needed to learn the ABCD criteria from raw dermoscopic pixels. Shah et al. (2023) found CNN consistently outperforms ANN and other classical classifiers for this task.

### Architecture

**Backbone:** EfficientNet-B0 pretrained on ImageNet. EfficientNet uses compound scaling to jointly increase network depth, width, and input resolution by a fixed ratio, achieving better accuracy per parameter than older architectures like VGG-16 or ResNet-50.

**Custom classification head** (replaces EfficientNet's original 1000-class ImageNet head):
```
Dropout(0.4)          <- regularization to prevent overfitting
Linear(1280 -> 512) + ReLU
Dropout(0.2)
Linear(512 -> 128) + ReLU
Linear(128 -> 2)      <- binary output: Benign / Malignant
Softmax               <- converts logits to probabilities that sum to 1
```

### Training Strategy

**Phase 1 — Frozen backbone (10 epochs, lr=1e-3):**
The EfficientNet backbone weights are frozen. Only the new classification head is trained. This prevents random gradients from an untrained head from destroying the pretrained feature representations on the first pass.

**Phase 2 — Full fine-tuning (10 epochs, lr=1e-4):**
All layers are unfrozen. The entire network is fine-tuned end-to-end using cosine annealing learning rate scheduling. The lower learning rate preserves the pretrained backbone while allowing it to adapt its feature representations to dermoscopic image characteristics.

**Optimizer:** Adam — an adaptive learning rate optimizer that maintains per-parameter learning rates, making it well suited for fine-tuning where different layers may need to update at different speeds.

### Handling Class Imbalance

The ISIC dataset is roughly 4:1 benign to malignant — matching real-world prevalence. Without correction, a naive model can achieve ~80% accuracy by always predicting benign, which is clinically useless. This project addresses imbalance with a **WeightedRandomSampler**: malignant samples are oversampled during training so each batch sees a roughly equal representation of both classes, forcing the model to learn discriminative features for the minority class.

---

## Project Structure

```
skin-cancer-detector/
├── README.md
├── requirements.txt
├── notebook/
│   └── train.ipynb       full pipeline: data -> preprocess -> train -> evaluate
└── app/
    ├── app.py            Gradio web interface (file upload + camera)
    └── model.py          model architecture and inference logic
```

---

## Quickstart

```bash
git clone https://github.com/ngoc-2/skin-cancer-detector.git
cd skin-cancer-detector
pip install -r requirements.txt
```

Download the ISIC 2017 dataset from https://challenge.isic-archive.com/data/#2017 (Task 1 images + Task 3 ground truth CSV) and place them in `data/`.

Open `notebook/train.ipynb` and run all cells to train the model. Training on CPU takes several hours — Google Colab with a free GPU is recommended.

Once `app/best_model.pth` is generated:

```bash
python app/app.py
```

Open http://localhost:7860 to use the app locally.

---

## References

- Shah et al. (2023). A comprehensive study on skin cancer detection using ANN and CNN. *Clinical eHealth*, 6, 76-84.
- Akinrinade & Du (2025). Skin cancer detection using deep machine learning techniques. *Intelligence-Based Medicine*, 11, 100191.
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*.
- ISIC Archive: https://www.isic-archive.com

---

*This tool is for educational and research purposes only. It is not a medical device. Always consult a qualified dermatologist.*
