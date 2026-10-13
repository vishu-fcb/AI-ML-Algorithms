# Traffic Sign Detection — CNN

> Classify 43 types of German traffic signs from images using a Convolutional Neural Network, achieving ~95% accuracy.

---

## Overview

This project trains a **Convolutional Neural Network (CNN)** to recognise and classify traffic signs from the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

The model identifies 43 distinct sign categories — from speed limits to stop signs to pedestrian warnings — making it a practical computer vision project with real-world automotive safety applications.

---

## Dataset

**Source:** [GTSRB — German Traffic Sign Recognition Benchmark (Kaggle)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

**Structure required:**
```
04_Traffic_Sign_Detection_CNN/
└── dataset/
    ├── Train/        # 43 subfolders (one per class), each containing training images
    └── Test.csv      # Test image paths and class labels
```

**Classes (43 total):** Speed limits (20–120 km/h), No passing, Priority road, Yield, Stop, No entry, Warning signs, Mandatory direction signs, and more.

---

## Algorithm & Architecture

### CNN Architecture

| Layer | Details |
|---|---|
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2, stride 2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2, stride 2 |
| Flatten | — |
| Dense | 512 units, ReLU, He initialisation |
| Dropout | 50% |
| Output Dense | 43 units, Softmax |

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical Cross-Entropy
- **Epochs:** 30 · **Batch size:** 32
- **Image size:** 64×64 RGB

### Data Augmentation

Applied during training: rotation (±10°), zoom (15%), width/height shift (10%), shear (15%).

---

## How to Run

### Prerequisites

```
Python 3.8+
tensorflow, keras, numpy, pandas, opencv-python, Pillow, scikit-learn
```

### Install dependencies

```bash
pip install tensorflow numpy pandas opencv-python Pillow scikit-learn
```

### Download dataset

Download from [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and place in `dataset/` as described above.

### Run

```bash
cd 04_Traffic_Sign_Detection_CNN
python TrafficSignDetector_CNN.py
```

Training runs for 30 epochs. Final test accuracy is printed at the end.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~95% |
| Classes | 43 traffic sign types |
| Input image size | 64×64 RGB |

---

## Tech Stack

`Python` · `TensorFlow/Keras` · `OpenCV` · `Pillow` · `NumPy` · `Pandas` · `scikit-learn`
