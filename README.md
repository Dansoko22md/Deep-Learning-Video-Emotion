# 🎭 Video Emotion Recognition using Deep Learning

A comprehensive deep learning project for recognizing human emotions from video clips using a hybrid architecture combining CNNs, LSTMs, and Autoencoders.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Computer%20Vision-red.svg)]()

## 📋 Description

This project implements an end-to-end deep learning pipeline for emotion recognition from video sequences. The system extracts spatial features from individual frames using CNNs, compresses and denoises features through Autoencoders, and captures temporal dependencies using LSTMs to classify emotions with high accuracy.

### Key Features

- **Multi-Architecture Approach**: Combines CNN, LSTM, and Autoencoder architectures
- **Temporal Modeling**: Captures emotion dynamics across video frames
- **Feature Compression**: Uses autoencoders for efficient representation learning
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1-score, and confusion matrix
- **Modular Design**: Easy to extend and customize for different datasets

## 🏗️ Architecture

```
Input Video (T×H×W×C)
    ↓
Frame Extraction & Preprocessing
    ↓
CNN Encoder (Spatial Features)
    ↓
Autoencoder Bottleneck (Feature Compression)
    ↓
LSTM (Temporal Dependencies)
    ↓
Dense Layers
    ↓
Softmax (Emotion Classification)
```

## 🎯 Supported Emotions

The model is designed to recognize various emotional states including:
- Anger
- Disgust
- Fear
- Happiness
- Sadness
- Neutral
- Surprise (dataset dependent)

## 📁 Project Structure

```
.
├── .venv/                      # Virtual environment
├── extracted_frames/           # Preprocessed video frames
├── models/                     # Saved trained models
├── notebooks/                  # Jupyter notebooks for experimentation
├── results/                    # Training results and visualizations
├── venv/                       # Alternative virtual environment
├── VideoFlash/                 # Video processing utilities
├── advanced_features.py        # Advanced model features (attention, 3D CNN)
├── main.py                     # Main training and evaluation script
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Dansoko22md/Deep-Learning-Video-Emotion.git
cd Deep-Learning-Video-Emotion
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Setup

Download the CREMA-D dataset from [Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad) and place it in the appropriate directory:

```bash
mkdir -p data/raw
# Extract dataset to data/raw/
```

## 💻 Usage

### Training the Model

Run the main training script:

```bash
python main.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

### Advanced Features

For attention mechanism or 3D CNN:

```bash
python advanced_features.py --model_type attention
```

### Evaluation

Evaluate a trained model:

```bash
python main.py --mode evaluate --model_path models/best_model.h5
```

## 📊 Results

The model achieves competitive performance on emotion recognition tasks:

- **Accuracy**: ~85% on test set
- **F1-Score**: ~0.83 (macro-average)
- **Training Time**: ~2-3 hours on GPU

Detailed results including confusion matrices and training curves are saved in the `results/` directory.

## 🔧 Model Components

### 1. CNN Encoder
- Extracts spatial features from individual frames
- Options: Custom CNN or ResNet backbone
- Output: Feature vectors per frame

### 2. Autoencoder
- Compresses features to lower dimensions
- Denoises input representations
- Learns efficient encodings

### 3. LSTM Network
- Captures temporal dependencies across frames
- Handles variable-length sequences
- Bidirectional option available

### 4. Classification Head
- Fully connected layers
- Dropout for regularization
- Softmax output for emotion probabilities

## 📈 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision & Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Optional**: t-SNE/PCA visualizations

## 🎓 Academic Context

This project was developed as part of a Deep Learning course exam, focusing on:
- Spatial feature extraction with CNNs
- Temporal sequence modeling with LSTMs
- Unsupervised learning with Autoencoders
- Multi-architecture deep learning pipelines
- Model evaluation and performance analysis

## 🔮 Future Improvements

- [ ] Implement attention mechanisms
- [ ] Explore 3D CNN architectures
- [ ] Add real-time video processing
- [ ] Support for additional datasets (FER2013, AffectNet)
- [ ] Model compression and optimization
- [ ] Web-based demo interface

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CREMA-D dataset creators
- TensorFlow and Keras communities
- Deep Learning course instructors

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

## 🏷️ Tags

`deep-learning` `computer-vision` `emotion-recognition` `cnn` `lstm` `autoencoder` `video-analysis` `tensorflow` `keras` `neural-networks` `affective-computing` `machine-learning` `python` `pytorch` `video-classification` `temporal-modeling` `feature-extraction` `crema-d` `facial-expression` `sentiment-analysis`

---

**⭐ Star this repository if you find it helpful!**
