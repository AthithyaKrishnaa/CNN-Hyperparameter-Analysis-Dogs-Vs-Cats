# CNN-Hyperparameter-Analysis-Dogs-Vs-Cats

A comprehensive Convolutional Neural Network implementation for binary image classification of cats and dogs, with extensive hyperparameter experimentation to determine optimal training configurations.

**Author:** Athithya Krishnaa M  
**Date:** 29th Dec 2025  
**Colab Link:** [View Notebook](https://colab.research.google.com/drive/1NLYmXrofGD9TDMw6ZVNkJJjJBZwrIjvo?usp=sharing)

## 📊 Project Overview

This project explores the impact of different hyperparameter configurations on CNN performance for image classification. Three distinct training configurations (Low, Medium, High) were tested to identify the optimal balance between training speed and model generalization.

## 🎯 Best Results

**Medium Configuration achieved the best performance:**
- **Validation Accuracy:** 88.07%
- **Validation Loss:** 0.3193
- **Training Time:** 22 epochs (with Early Stopping)

## 🔧 Training Configurations

### Low Settings
```python
batch_size = 16
learning_rate = 1e-5
epochs = 5
```
**Results:**
- Final Val Accuracy: 66.15%
- **Observation:** Underfitted - Learning rate too slow for only 5 epochs

### Medium Settings (Optimal) ⭐
```python
batch_size = 32
learning_rate = 2e-4
epochs = 30 (with EarlyStopping)
```
**Results:**
- Best Val Accuracy: 88.07%
- Val Loss: 0.3193
- Stopped at Epoch 16
- **Observation:** Best balance of speed and generalization

### High Settings
```python
batch_size = 64
learning_rate = 1e-3
epochs = 100 (with EarlyStopping)
```
**Results:**
- Final Val Accuracy: 88.01%
- Val Loss: 0.2951
- Stopped at Epoch 18
- **Observation:** High accuracy but showed validation loss instability

## 🏗️ Model Architecture

The CNN architecture consists of 5 convolutional blocks with progressive feature extraction:

```python
# Architecture Overview
Input Layer: (180, 180, 3)

CNN Blocks:
├── Block 1: 32 filters, Dropout 0.10
├── Block 2: 64 filters, Dropout 0.15
├── Block 3: 128 filters, Dropout 0.20
├── Block 4: 256 filters, Dropout 0.25
└── Block 5: 512 filters, Dropout 0.30

Classifier:
├── GlobalAveragePooling2D
├── Dense(256, activation='relu')
├── Dropout(0.40)
└── Dense(2, activation='softmax')
```

Each block contains:
- 2x Conv2D layers (3x3 kernel, same padding, ReLU activation)
- BatchNormalization
- MaxPooling2D
- Dropout

## 📦 Dataset

- **Source:** [Kaggle - Cat and Dog Dataset](https://www.kaggle.com/tongpython/cat-and-dog)
- **Total Images:** 8,005
- **Training Set:** 6,404 images (80%)
- **Validation Set:** 1,601 images (20%)
- **Classes:** 2 (Cats, Dogs)
- **Image Size:** 180x180 pixels

## 🚀 Getting Started

### Prerequisites

```python
tensorflow>=2.19.0
numpy
matplotlib
kagglehub
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CNN-dogs_vs_cats
```

2. Install dependencies:
```bash
pip install tensorflow numpy matplotlib kagglehub
```

### Running the Code

#### In Google Colab:
1. Open the [Colab Notebook](https://colab.research.google.com/drive/1NLYmXrofGD9TDMw6ZVNkJJjJBZwrIjvo?usp=sharing)
2. Select GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells

## 🔬 Hyperparameter Experimentation

To test different configurations, modify the settings in the first cell:

```python
# HYPERPARAMETER SETTINGS
SETTINGS = {
    "batch_size": 32,      # Options: 16, 32, 64
    "learning_rate": 2e-4, # Options: 1e-5, 2e-4, 1e-3
    "epochs": 30,          # Options: 5, 30, 100
    "img_height": 180,
    "img_width": 180
}
```

## 📈 Training Features

### Data Augmentation
- Random horizontal flip
- Random rotation (±10%)
- Random zoom (±10%)

### Callbacks
- **ModelCheckpoint:** Saves best model based on validation loss
- **ReduceLROnPlateau:** Reduces learning rate when validation loss plateaus
  - Factor: 0.5
  - Patience: 3 epochs
  - Min LR: 1e-7
- **EarlyStopping:** Stops training when no improvement
  - Patience: 6 epochs
  - Restores best weights

## 📊 Results Summary

| Configuration | Batch Size | Learning Rate | Epochs | Final Val Accuracy | Key Observation |
|--------------|------------|---------------|--------|-------------------|-----------------|
| **Low** | 16 | 1×10⁻⁵ | 5 | 66.15% | Underfitted |
| **Medium** ⭐ | 32 | 2×10⁻⁴ | 19* | 88.26% | Optimal balance |
| **High** | 64 | 1×10⁻³ | 18* | 88.01% | Validation loss instability |

*Stopped early due to EarlyStopping callback

## 🎓 Key Learnings

1. **Learning Rate Impact:** Too low (1e-5) leads to underfitting, too high (1e-3) causes instability
2. **Batch Size:** Medium batch size (32) provided best balance between training speed and generalization
3. **Early Stopping:** Essential for preventing overfitting and reducing training time
4. **Data Augmentation:** Crucial for improving model generalization on limited dataset

## 📝 License

This project is open source and available for educational purposes.

---

**Note:** This project was developed as part of a deep learning experimentation study to understand the impact of hyperparameters on CNN performance.
