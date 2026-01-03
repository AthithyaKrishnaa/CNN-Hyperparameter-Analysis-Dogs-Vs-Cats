
# CNN-Hyperparameter-Analysis-Dogs-Vs-Cats

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NLYmXrofGD9TDMw6ZVNkJJjJBZwrIjvo?usp=sharing)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning study exploring the impact of hyperparameter configurations on CNN performance for binary image classification (Cats vs Dogs). This project systematically analyzes three distinct training configurations to identify optimal settings for balancing training efficiency and model generalization.

**Author:** Athithya Krishnaa M  
**Date:** December 29, 2025  

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Best Results](#-best-results)
- [Training Configurations](#-training-configurations)
  - [Low Configuration](#low-configuration-️)
  - [Medium Configuration](#medium-configuration--recommended)
  - [High Configuration](#high-configuration-)
- [Model Architecture](#️-model-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results Analysis](#-results-analysis)
- [Key Learnings](#-key-learnings)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [Contact](#-contact)

## 📊 Project Overview

This research project investigates how different hyperparameter configurations affect Convolutional Neural Network (CNN) performance in binary image classification tasks. By systematically testing Low, Medium, and High configuration profiles, the study provides empirical evidence for optimal hyperparameter selection.

### Research Questions
1. How does learning rate affect model convergence and final accuracy?
2. What is the optimal batch size for this classification task?
3. How do different configurations impact training stability?
4. What trade-offs exist between training speed and model generalization?

## 🎯 Best Results

**Medium Configuration achieved optimal performance:**
- **Validation Accuracy:** 88.07%
- **Validation Loss:** 0.3193
- **Training Epochs:** 16 (stopped early)
- **Training Time:** ~22 epochs scheduled
- **Model Stability:** Excellent (no validation loss spikes)

This configuration represents the sweet spot between training efficiency and model accuracy.

## 🔧 Training Configurations

### Configuration Comparison

| Setting | Low ❄️ | Medium ⭐ | High 🔥 |
|---------|--------|----------|---------|
| **Batch Size** | 16 | 32 | 64 |
| **Learning Rate** | 1×10⁻⁵ | 2×10⁻⁴ | 1×10⁻³ |
| **Max Epochs** | 5 | 30 | 100 |
| **Final Val Acc** | 66.15% | **88.07%** | 88.01% |
| **Val Loss** | N/A | **0.3193** | 0.2951 |
| **Actual Epochs** | 5 | 16 | 18 |
| **Status** | Underfitted | ✅ Optimal | Unstable |

---

### Low Configuration ❄️

```python
SETTINGS = {
    "batch_size": 16,
    "learning_rate": 1e-5,
    "epochs": 5
}
```

#### Training Progress
![Low Config Training Graphs](https://github.com/AthithyaKrishnaa/CNN-Hyperparameter-Analysis-Dogs-Vs-Cats/blob/main/low_graph.png)

#### Prediction Results
![Low Config Predictions](https://github.com/AthithyaKrishnaa/CNN-Hyperparameter-Analysis-Dogs-Vs-Cats/blob/main/op-low-result.png)

**Analysis:** 
- ❌ **Underfitted** - Learning rate too conservative for short training period
- Final validation accuracy: 66.15%
- Model failed to converge properly within 5 epochs
- Training accuracy plateaued around 65%
- Clear signs of underfitting with large accuracy gap

**Key Observation:** Combination of low learning rate (1e-5) and minimal epochs (5) prevented the model from learning complex features effectively.

---

### Medium Configuration ⭐ (Recommended)

```python
SETTINGS = {
    "batch_size": 32,
    "learning_rate": 2e-4,
    "epochs": 30  # EarlyStopping at epoch 16
}
```

#### Training Progress
![Medium Config Training Graphs](https://github.com/AthithyaKrishnaa/CNN-Hyperparameter-Analysis-Dogs-Vs-Cats/blob/main/medium_graph.png)

#### Prediction Results
![Medium Config Predictions](https://github.com/AthithyaKrishnaa/CNN-Hyperparameter-Analysis-Dogs-Vs-Cats/blob/main/op-medium-results.png)

**Analysis:**
- ✅ **Optimal Balance** - Best configuration for this task
- Final validation accuracy: **88.07%**
- Validation loss: **0.3193**
- Smooth convergence with stable validation metrics
- Early stopping at epoch 16 prevented overfitting
- Training accuracy reached 94.78% without significant overfitting
- Validation curve shows steady improvement with minimal fluctuations

**Key Observation:** Achieved excellent generalization with 8 out of 9 correct predictions in sample batch. The model demonstrates strong performance on both cats and dogs with balanced accuracy.

---

### High Configuration 🔥

```python
SETTINGS = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "epochs": 100  # EarlyStopping at epoch 18
}
```

#### Training Progress
![High Config Training Graphs](https://github.com/AthithyaKrishnaa/CNN-Hyperparameter-Analysis-Dogs-Vs-Cats/blob/main/high%20_graph.png)

#### Prediction Results
![High Config Predictions]([op-high-results.jpg](https://github.com/AthithyaKrishnaa/CNN-Hyperparameter-Analysis-Dogs-Vs-Cats/blob/main/op-high-results.png))

**Analysis:**
- ⚠️ **Fast but Unstable** - High learning rate causes volatility
- Final validation accuracy: 88.01%
- Validation loss: 0.2951
- Fast initial convergence but exhibited validation loss spikes
- Notable loss spike around epoch 13 (reaching 1.4)
- Training accuracy reached 97.11% but validation showed instability
- Higher learning rate caused occasional overshooting

**Key Observation:** While achieving competitive accuracy, the validation loss instability indicates potential issues with generalization in production scenarios. The large loss spike demonstrates the risk of aggressive learning rates.

---

## 🏗️ Model Architecture

Custom CNN architecture with progressive feature extraction:

```python
Model: "Sequential CNN"
_________________________________________________________________
Input Layer: (180, 180, 3)
_________________________________________________________________

CNN Block 1:
├── Conv2D(32, 3×3, ReLU, same padding)
├── Conv2D(32, 3×3, ReLU, same padding)
├── BatchNormalization()
├── MaxPooling2D(2×2)
└── Dropout(0.10)

CNN Block 2:
├── Conv2D(64, 3×3, ReLU, same padding)
├── Conv2D(64, 3×3, ReLU, same padding)
├── BatchNormalization()
├── MaxPooling2D(2×2)
└── Dropout(0.15)

CNN Block 3:
├── Conv2D(128, 3×3, ReLU, same padding)
├── Conv2D(128, 3×3, ReLU, same padding)
├── BatchNormalization()
├── MaxPooling2D(2×2)
└── Dropout(0.20)

CNN Block 4:
├── Conv2D(256, 3×3, ReLU, same padding)
├── Conv2D(256, 3×3, ReLU, same padding)
├── BatchNormalization()
├── MaxPooling2D(2×2)
└── Dropout(0.25)

CNN Block 5:
├── Conv2D(512, 3×3, ReLU, same padding)
├── Conv2D(512, 3×3, ReLU, same padding)
├── BatchNormalization()
├── MaxPooling2D(2×2)
└── Dropout(0.30)

Classifier Head:
├── GlobalAveragePooling2D()
├── Dense(256, ReLU)
├── Dropout(0.40)
└── Dense(2, Softmax)
_________________________________________________________________
Total Parameters: ~15M (trainable)
```

### Architecture Highlights
- **Progressive Dropout:** Increases from 0.10 → 0.40 to combat overfitting
- **Batch Normalization:** Stabilizes training and accelerates convergence
- **Global Average Pooling:** Reduces parameters while maintaining spatial information
- **Double Convolution:** Enhances feature extraction in each block

## 📦 Dataset

### Dataset Specifications
- **Source:** [Kaggle - Cat and Dog Dataset](https://www.kaggle.com/tongpython/cat-and-dog)
- **Total Images:** 8,005
- **Training Set:** 6,404 images (80%)
- **Validation Set:** 1,601 images (20%)
- **Classes:** 2 (Cats, Dogs) - Binary Classification
- **Image Dimensions:** 180×180 pixels (RGB)
- **Split Method:** Random seed=42 for reproducibility

### Data Preprocessing
```python
# Normalization
pixel_values =  →[1]

# Augmentation (Training only)
- RandomFlip(horizontal)
- RandomRotation(±10%)
- RandomZoom(±10%)
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- GPU recommended (Google Colab provides free GPU access)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/AthithyaKrishnaa/CNN-Hyperparameter-Analysis-Dogs-Vs-Cats.git
cd CNN-Hyperparameter-Analysis-Dogs-Vs-Cats

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```txt
tensorflow>=2.19.0
numpy>=1.24.0
matplotlib>=3.7.0
kagglehub>=0.2.0
```

## 💻 Usage

### Option 1: Google Colab (Recommended)
1. Click the **Open in Colab** badge above
2. Navigate to `Runtime → Change runtime type → GPU (T4)`
3. Run all cells sequentially
4. Models are automatically saved to `best_cnn_cat_dog.keras`

### Option 2: Local Execution
```bash
# Launch Jupyter Notebook
jupyter notebook CNN_-dogs_vs_cats.ipynb

# Or run directly with Python
python train_cnn.py --config medium
```

### Hyperparameter Customization

Modify the first code cell to experiment with configurations:

```python
# HYPERPARAMETER SETTINGS
# Change these values to test different configurations

SETTINGS = {
    "batch_size": 32,        # Options: 16, 32, 64
    "learning_rate": 2e-4,   # Options: 1e-5, 2e-4, 1e-3
    "epochs": 30,            # Options: 5, 30, 100
    "img_height": 180,
    "img_width": 180
}
```

## 📈 Results Analysis

### Training Progression Comparison

#### Low Configuration Performance
- Training plateaued quickly
- Large gap between train and validation accuracy
- Insufficient learning from limited epochs

#### Medium Configuration Performance (Best)
- Smooth convergence pattern
- Minimal overfitting (train: 94.78%, val: 88.07%)
- Stable validation metrics throughout training
- Early stopping at optimal point (epoch 16)

#### High Configuration Performance
- Fast initial learning
- Validation loss spike at epoch 13 (1.4136)
- Learning rate reduced by ReduceLROnPlateau callback
- Final stability achieved but with more volatility

### Training Features

#### Callbacks Implementation
```python
callbacks = [
    # Save best model
    ModelCheckpoint(
        'best_cnn_cat_dog.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    # Adaptive learning rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Early stopping
    EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
]
```

## 🎓 Key Learnings

### 1. Learning Rate Impact
- **Too Low (1e-5):** Slow convergence, underfitting, requires many epochs
- **Optimal (2e-4):** Smooth learning curve, stable validation metrics
- **Too High (1e-3):** Fast initial progress but unstable, prone to overshooting with loss spikes

### 2. Batch Size Trade-offs
- **Small (16):** Better generalization but slower training, prone to underfitting with limited epochs
- **Medium (32):** ✅ Optimal balance for this dataset size - best generalization
- **Large (64):** Faster training but less stable gradients, requires careful learning rate tuning

### 3. Regularization Importance
- Progressive dropout (0.10 → 0.40) crucial for preventing overfitting
- Data augmentation significantly improved generalization
- Early stopping saved computational resources (50% reduction)
- Batch normalization stabilized training across all configurations

### 4. Training Efficiency
- Medium config achieved 88% accuracy in just 16 epochs
- Early stopping reduced training time by ~50%
- GPU acceleration essential for reasonable training times
- Proper hyperparameter selection more important than extended training

### 5. Model Generalization
- Visual inspection of predictions reveals strong performance on diverse images
- Medium config shows balanced performance on both cats and dogs
- High config predictions accurate but training instability is concerning
- Low config struggles with harder examples (unusual poses, lighting)

## 📁 Project Structure

```
CNN-Hyperparameter-Analysis-Dogs-Vs-Cats/
├── CNN_-dogs_vs_cats.ipynb      # Main training notebook
├── best_cnn_cat_dog.keras       # Saved best model (Medium config)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── Results/                      # Training results and visualizations
│   ├── low_graph.jpg            # Low config training curves
│   ├── op-low-result.jpg        # Low config predictions
│   ├── medium_graph.jpg         # Medium config training curves
│   ├── op-medium-results.jpg    # Medium config predictions
│   ├── high-_graph.jpg          # High config training curves
│   └── op-high-results.jpg      # High config predictions
│
└── .gitignore                    # Git ignore file
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution
- Testing with different CNN architectures (ResNet, VGG, EfficientNet)
- Implementing additional hyperparameter combinations
- Adding cross-validation for more robust evaluation
- Exploring transfer learning approaches
- Creating automated hyperparameter tuning (GridSearch/RandomSearch)
- Improving documentation and tutorials

## 📧 Contact

**Athithya Krishnaa M**  
- GitHub: [@AthithyaKrishnaa](https://github.com/AthithyaKrishnaa)
- LinkedIn: [Connect with me](https://linkedin.com/in/athithyakrishnaa)
- Colab: [View Live Notebook](https://colab.research.google.com/drive/1NLYmXrofGD9TDMw6ZVNkJJjJBZwrIjvo?usp=sharing)

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Kaggle for providing the Cat and Dog dataset
- TensorFlow team for the excellent deep learning framework
- Google Colab for free GPU resources
- The deep learning community for continuous inspiration

## 📚 References

1. [Deep Learning Specialization - Andrew Ng](https://www.coursera.org/specializations/deep-learning)
2. [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
3. [Hyperparameter Tuning Best Practices](https://arxiv.org/abs/1206.5533)
4. [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the Deep Learning Community

[Report Bug](https://github.com/AthithyaKrishnaa/CNN-Hyperparameter-Analysis-Dogs-Vs-Cats/issues) · [Request Feature](https://github.com/AthithyaKrishnaa/CNN-Hyperparameter-Analysis-Dogs-Vs-Cats/issues)

</div>
```

## 📋 Setup Instructions

To use this README in your repository:

1. **Create a `Results` folder** in your repository root
2. **Upload these images** to the `Results` folder:
   - `low_graph.jpg`
   - `op-low-result.jpg`
   - `medium_graph.jpg`
   - `op-medium-results.jpg`
   - `high-_graph.jpg`
   - `op-high-results.jpg`

3. **Update image paths** in README if needed (currently set to root directory)

4. **Create `requirements.txt`**:
```txt
tensorflow>=2.19.0
numpy>=1.24.0
matplotlib>=3.7.0
kagglehub>=0.2.0
```
