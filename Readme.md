# Emotion Detection from Tweets

A deep learning project for detecting emotions from Twitter text data using recurrent neural networks.

##  Dataset

- **Size**: ~410,000 tweets
- **Classes**: 6 emotion categories
- **Task**: Multi-class emotion classification
- **Location**: Available in this repository (`/data` folder)
- **Format**: CSV

##  Model Architecture

This project explores four different recurrent neural network architectures for emotion detection:

### Model 1: GRU
- Single GRU layer (40 units)
- Text Vectorization layer
- Embedding layer

### Model 2: LSTM
- Single LSTM layer (40 units)
- Text Vectorization layer
- Embedding layer

### Model 3: Bidirectional GRU
- Single Bidirectional GRU layer (32 units)
- Text Vectorization layer
- Embedding layer

### Model 4: Bidirectional LSTM
- Single Bidirectional LSTM layer (32 units)
- Text Vectorization layer
- Embedding layer

## Results

| Model | Loss | Accuracy | Precision | Recall |
|-------|------|----------|-----------|--------|
| GRU (40 units) | 0.4252 | **85.17%** | - | - |
| LSTM (40 units) | 0.4195 | **85.54%** | **87.17%** | **84.75%** |
| Bi-GRU (32 units) | 0.4861 | 85.08% | 86.37% | 84.53% |
| Bi-LSTM (32 units) | 0.4878 | 84.53% | 85.80% | 84.05% |

### Key Findings

- **Best Overall Performance**: LSTM (40 units) achieved the highest accuracy (85.54%) with the lowest loss (0.4195)
- **Best Precision**: LSTM (40 units) at 87.17%
- **Best Recall**: LSTM (40 units) at 84.75%
- Unidirectional architectures (GRU and LSTM) outperformed bidirectional variants in this task
- Simple LSTM architecture proved most effective despite having fewer parameters than bidirectional models

## Tech Stack

- **Framework**: TensorFlow/Keras
- **Language**: Python
- **Key Libraries**: NumPy, Pandas, Matplotlib, Seaborn

## Getting Started

### Prerequisites
```bash
pip install tensorflow numpy pandas matplotlib seaborn
```

### Usage
```python
# Load and preprocess data
# Train model
# Evaluate performance
```

## Model Training

Each model includes:
- Text vectorization preprocessing
- Embedding layer for word representations
- Recurrent layers (GRU/LSTM/Bi-GRU/Bi-LSTM)
- Dense output layer with softmax activation

##  Future Improvements

- Experiment with attention mechanisms
- Try transformer-based architectures (BERT, RoBERTa)
- Implement ensemble methods
- Add real-time emotion detection API
- Deploy as web application

