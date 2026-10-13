# Image Classification — CIFAR-10 CNN

> Classify 10 object categories from 32×32 colour images using a Convolutional Neural Network trained on the CIFAR-10 benchmark dataset.

---

## Overview

This project trains a **CNN** on the **CIFAR-10** dataset — one of the most widely used benchmarks in computer vision. The model learns to distinguish between 10 everyday object classes from small colour images.

The dataset is loaded directly from Keras, so no manual download is required. The architecture uses L2 regularisation and Dropout to reduce overfitting, with a `ReduceLROnPlateau` callback to adaptively lower the learning rate during training.

---

## Dataset

**Source:** CIFAR-10 — built into `tensorflow.keras.datasets`, downloaded automatically on first run.

| Detail | Value |
|---|---|
| Training images | 50,000 |
| Test images | 10,000 |
| Image size | 32×32 RGB |
| Classes | 10 |

**Classes:** airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck

---

## Algorithm & Architecture

### CNN Architecture

| Layer | Details |
|---|---|
| Conv2D | 128 filters, 2×2, ReLU |
| MaxPooling2D | 2×2, stride 2 |
| Conv2D | 64 filters, 2×2, ReLU |
| MaxPooling2D | 2×2, stride 1 |
| Flatten | — |
| Dense | 128 units, ReLU, L2 regularisation (λ=0.01) |
| Dropout | 20% |
| Dense | 64 units, ReLU |
| Dropout | 20% |
| Output Dense | 10 units, Softmax |

- **Optimizer:** Adam (lr=0.0001)
- **Loss:** Sparse Categorical Cross-Entropy
- **Epochs:** 30 · **Batch size:** 32
- **LR Scheduler:** ReduceLROnPlateau (factor=0.1, patience=3, min_lr=1e-6)

---

## How to Run

### Prerequisites

```
Python 3.8+
tensorflow, numpy, pandas, matplotlib
```

### Install dependencies

```bash
pip install tensorflow numpy pandas matplotlib
```

### Run

```bash
cd 05_Image_Classification_CIFAR_CNN
python CNN_Image_Classification.py
```

CIFAR-10 downloads automatically on first run (~170 MB). Training runs for 30 epochs and the final test accuracy is printed.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~75–80% (varies by run) |
| Classes | 10 |
| Input image size | 32×32 RGB |

CIFAR-10 is a challenging dataset — human-level accuracy is ~94%. The model achieves a solid baseline without any pre-trained transfer learning.

---

## Tech Stack

`Python` · `TensorFlow/Keras` · `NumPy` · `Pandas` · `Matplotlib`
