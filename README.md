# CIFAR-10 Image Classification with a Lightweight Residual CNN (PyTorch)

This repository contains an implementation of a lightweight convolutional neural network for image classification on the CIFAR-10 dataset.  

The goal is to design and train a CNN model with **fewer than 400,000 trainable parameters**, while achieving good generalization performance using standard deep learning practices.

---

## 📌 Project Overview

- **Framework**: PyTorch
- **Dataset**: CIFAR-10
- **Task**: Image classification (10 classes)
- **Model Constraint**: < 400k trainable parameters
- **Metrics**: Cross-entropy loss, classification accuracy

---

## 📂 Project Structure
```text
project/
│
├── train.py                      # Main training script (checkpointing per epoch)
├── test.py                       # Test evaluation using a selected epoch checkpoint
│
├── dataset.py                    # CIFAR-10 dataset loading and preprocessing
│
├── models/
│   ├── __init__.py
│   └── cnn.py                    # CNN architecture (ResNet-inspired, <400k params)
│
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py             # Save/load checkpoints (all epochs)
│   └── plot.py                   # Training/validation loss & accuracy curves
│
├── checkpoints/                  # Saved model weights (per epoch)
│   ├── epoch_01.pth
│   ├── epoch_02.pth
│   └── ...
│
├── data/                         # dataset
│
├── loss_curve.png                # Loss & accuracy plots
│
└── README.md                     # Project documentation
```


## 🧠 Model Architecture

The model is a compact residual CNN inspired by ResNet-style architectures.

Main components:
- Initial convolutional stem (3×3 Conv + BatchNorm + ReLU)
- Residual blocks with skip connections
- Channel expansion and spatial downsampling via strided convolutions
- Dropout inside selected residual blocks
- Global average pooling
- Single fully connected layer for classification

Residual connections improve optimization stability while keeping the parameter count low.

**Total trainable parameters**: ~401K

---

## 🔧 Training Configuration

- **Optimizer**: SGD with momentum
  - Learning rate: 0.1
  - Momentum: 0.9
  - Weight decay: 5e-4
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate Scheduler**: StepLR
  - Step size: 30 epochs
  - Gamma: 0.1

---

## 📊 Dataset & Validation Strategy

- CIFAR-10 training set: 50,000 images
- CIFAR-10 test set: 10,000 images
- Validation split:
  - Last 5,000 samples of the training set
  - Remaining 45,000 samples used for training

Data augmentation is applied **only to the training set**:
- Random crop with padding
- Random horizontal flip

Validation and test sets use deterministic preprocessing.

---

## 💾 Checkpointing & Model Selection

- Model weights are saved **at every epoch**
- Each checkpoint contains:
  - Model weights
  - Optimizer state
  - Training & validation loss history

After training:
1. Training and validation loss curves are plotted
2. The user manually selects the desired epoch
3. The selected checkpoint is evaluated on the test set

This approach provides transparent and interpretable model selection.

---

## ▶️ How to Run

### Train and Test the model
```bash
python train.py
python test.py

