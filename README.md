# Apple Color Classification with YOLOv8

This repository contains a Jupyter Notebook for training a YOLOv8 model to classify apple color stages (green, red, yellow) using object detection. The model achieves **99.5% mAP50** and is optimized for CUDA-enabled GPUs.

Data set [Apple Maturity](https://universe.roboflow.com/nn-2ju5u/apple_maturity-1ayzw/dataset/1)

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Inference](#inference)
- [Directory Structure](#directory-structure)
- [References](#references)

---

## Project Overview

The notebook performs the following tasks:

1. **Environment Setup**: Installs Ultralytics YOLOv8 and CUDA dependencies.
2. **Hardware Verification**: Checks NVIDIA GPU availability and CUDA compatibility.
3. **Data Preparation**: Uses a custom dataset in YOLO format with `data.yaml`.
4. **Model Training**: Trains YOLOv8m with hyperparameter tuning and early stopping.
5. **Evaluation**: Validates model performance on test data.

---

## Requirements

- **OS**: Windows/Linux/macOS
- **GPU**: NVIDIA GPU with CUDA 12.8+ (Tested on RTX 4060 Laptop GPU)
- **Python**: 3.12.8
- **Key Packages**:
  ```text
  ultralytics==8.3.70
  torch==2.6.0+cu126
  opencv-python==4.11.0.86
  cuda-python==12.8.0
  ```

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/apple-maturity-detection.git
cd apple-maturity-detection
```

Install dependencies:

```bash
pip install ultralytics cuda-python opencv-python
```

## Dataset Structure

Raw dataset from Roboflow. Organized as:

```bash
data/classify/
├── data.yaml
├── train/
├── valid/
├── test/
├── raw/        # Original images
└── processed/  # Preprocessed outputs
```

## Training Configuration

**Model**: yolov8m.pt\
**Hyperparameters**:

```python
{
  "epochs": 300,
  "imgsz": 224,
  "batch": 8,
  "optimizer": "AdamW",
  "lr0": 0.001,
  "lrf": 0.01,
  "weight_decay": 0.0001,
  "dropout": 0.2,
  "patience": 20  # Early stopping
}
```

## Results

**Validation Metrics**:

| Class  | Precision | Recall | mAP50 | mAP50-95 |
| ------ | --------- | ------ | ----- | -------- |
| All    | 0.999     | 1.0    | 0.995 | 0.951    |
| Green  | 0.997     | 1.0    | 0.995 | 0.973    |
| Red    | 1.0       | 1.0    | 0.995 | 0.942    |
| Yellow | 1.0       | 1.0    | 0.995 | 0.944    |

**Training Logs**: Saved to `runs/detect/train37/`

## Inference

To use the trained model:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train37/weights/best.pt")
results = model.predict("path/to/image.jpg", conf=0.2)
```

## Directory Structure

```bash
/0_Apple Color Classification/
├── data/
│   └── classify/
├── runs/
│   └── detect/
│       └── train37/  # Training artifacts
└── classify.ipynb    # Main notebook
```

## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [Roboflow Dataset](https://universe.roboflow.com/)
- [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)

