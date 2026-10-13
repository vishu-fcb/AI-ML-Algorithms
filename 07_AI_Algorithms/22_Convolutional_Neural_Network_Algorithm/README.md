# Convolutional Neural Network (CNN)

**Category:** Deep Learning — Computer Vision

---

## What It Does

This notebook implements a **CNN for binary image classification** — distinguishing between two image categories (e.g. cats vs dogs). CNNs use convolutional filters to automatically learn spatial features from images (edges, textures, shapes) without manual feature engineering.

---

## Network Architecture

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Output Dense | 1 unit, Sigmoid |

- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Data Augmentation:** rotation, shear, zoom, horizontal flip

---

## When To Use It

- Any image classification task
- CNNs are the standard for computer vision — they outperform fully-connected networks by exploiting spatial structure
- Transfer learning (VGG, ResNet, EfficientNet) is recommended for real-world tasks — this notebook teaches the fundamentals from scratch

---

## Dataset

**Image dataset required** — not included in the repository.

**Recommended:** Download the Cats vs Dogs dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats).

**Expected folder structure:**
```
22_Convolutional_Neural_Network_Algorithm/
└── dataset/
    ├── training_set/
    │   ├── cats/      (images)
    │   └── dogs/      (images)
    └── test_set/
        ├── cats/      (images)
        └── dogs/      (images)
```

---

## How to Run

```bash
pip install tensorflow numpy
jupyter notebook convolutional_neural_network.ipynb
```

Run all cells: **Kernel → Restart & Run All**

---

## Key Output

- Training and validation accuracy curves
- Single image prediction: upload any cat or dog photo and get a classification
