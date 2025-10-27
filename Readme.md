# Emotion Detection with Deep Learning

A text-based emotion classification system that detects six different emotions using LSTM and GRU neural networks. This project demonstrates the superiority of deep learning approaches for large-scale, complex NLP tasks.

## Overview

This project implements deep learning models to classify text into six emotion categories:
-  Sadness (0)
-  Joy (1)
-  Love (2)
-  Anger (3)
-  Fear (4)
-  Surprise (5)

## Why Deep Learning?

Traditional machine learning models (Logistic Regression, Random Forest, SVM) are **not suitable** for this task due to:
- **Dataset size**: 416,809 samples - too large for traditional models to process efficiently
- **Text complexity**: Capturing semantic relationships and context requires sequential processing
- **Feature engineering burden**: Traditional ML requires manual feature extraction (TF-IDF, n-grams), which loses contextual information
- **Long-range dependencies**: Understanding emotion requires capturing word relationships across sentences

Deep learning models (LSTM/GRU) automatically learn hierarchical features and temporal patterns, making them ideal for this scale and complexity.

## Dataset

- **Total samples**: 416,809 text entries
- **Distribution** (imbalanced):
  - Joy: 141,067 samples (33.8%)
  - Sadness: 121,187 samples (29.1%)
  - Anger: 57,317 samples (13.8%)
  - Fear: 47,712 samples (11.4%)
  - Love: 34,554 samples (8.3%)
  - Surprise: 14,972 samples (3.6%)
- **Data versioning**: Dataset (`text.csv`) is tracked using **DVC** for reproducibility

## Data Preprocessing

- **Text Vectorization**: TensorFlow's `TextVectorization` layer
  - Standardization: Lowercase + punctuation stripping
  - Vocabulary size: 10,000 most frequent tokens
  - Sequence length: 35 words (90th percentile)
  - Out-of-vocabulary handling: UNK token
- **Embeddings**: 128-dimensional learned embeddings
- **Class balancing**: Computed class weights to handle imbalanced distribution
- **Train/Val split**: 80/20 stratified split (333,447 train / 83,362 validation)

## Models

All models use the same preprocessing pipeline with TensorFlow's functional API for end-to-end text processing.

### 1. GRU Model v1
- **Architecture**: 
  - Input → TextVectorization → Embedding(128d)
  - GRU(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
  - GRU(32, dropout=0.3, recurrent_dropout=0.2)
  - Dropout(0.3) → Dense(6, softmax)
- **Parameters**: ~1.5M
- **Performance**:
  - Test Accuracy: 92.75%
  - Test F1-Score: 93.00%
  - Training time: ~6 minutes (4 epochs)
- **Tracked with**: DVC

### 2. GRU Model v2
- **Architecture**: Same as v1 but with 64 units per GRU layer
- **Parameters**: ~5M
- **Performance**:
  - Test Accuracy: 92.78%
  - Test F1-Score: 92.97%
  - Training time: ~10 minutes (5 epochs)

### 3. LSTM Model (Best Performance) ⭐
- **Architecture**:
  - Input → TextVectorization → Embedding(128d)
  - LSTM(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
  - LSTM(32, dropout=0.3, recurrent_dropout=0.2)
  - Dropout(0.3) → Dense(6, softmax)
- **Parameters**: ~1.8M
- **Performance**:
  - Test Accuracy: 92.72%
  - Test F1-Score: 92.92%
  - Test Precision: 93.82%
  - Test Recall: 92.72%
  - Training time: ~7 minutes (6 epochs)
- **Tracked with**: DVC
- **Saved as**: `Models/lstm.keras`

## Evaluation Metrics

Comprehensive evaluation using multiple metrics to assess model performance:

- **Accuracy**: Overall correct predictions
- **Precision**: Quality of positive predictions (reduces false positives)
- **Recall**: Coverage of actual positives (reduces false negatives)
- **F1-Score**: Harmonic mean of precision and recall (balanced metric)
- **Per-class metrics**: Evaluated for each emotion to identify class-specific strengths/weaknesses

All metrics are computed using `sklearn.metrics.classification_report` and logged to MLflow for experiment tracking.

## Training Configuration

- **Optimizer**: Adam (initial LR: 0.001)
- **Loss function**: Sparse Categorical Crossentropy
- **Batch size**: 64
- **Max epochs**: 25
- **Callbacks**:
  - Early Stopping (patience=2, restore_best_weights=True)
  - ReduceLROnPlateau (patience=2, factor=0.5, min_lr=1e-6)
- **Class weights**: Applied to handle imbalanced dataset

## Experiment Tracking & Version Control

### MLflow
All experiments tracked with MLflow including:
- Hyperparameters (layers, units, dropout rates, learning rate)
- Training & validation metrics per epoch
- Final evaluation metrics (accuracy, precision, recall, F1)
- Model artifacts and metadata
- Custom tags for model type identification

### DVC (Data Version Control)
Version-controlled assets:
- **Dataset**: `data/text.csv` (large file tracking)
  - Pointer file: `data.dvc`
- **Models**: 
  - `Models/gru.keras` (GRU v1)
  - `Models/lstm.keras` (LSTM)
  - Pointer file: `models.dvc`
- DVC pointer files (`.dvc`) are committed to Git for version tracking
- Actual large files stored in remote DVC storage
- Ensures reproducibility and efficient storage
- Enables model and data lineage tracking

## Requirements

```bash
tensorflow>=2.10.0
pandas
numpy
scikit-learn
mlflow
dvc
```

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd emotion-detection

# Install dependencies
pip install -r requirements.txt

# Pull DVC-tracked files
dvc pull
```

## Usage

```python
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('Models/lstm.keras')

# Predict emotion from text
texts = [
    "i feel really happy today",
    "i am so scared right now",
    "this makes me very angry"
]

predictions = model.predict(texts)
emotions = np.argmax(predictions, axis=1)

emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 
               3: 'anger', 4: 'fear', 5: 'surprise'}

for text, emotion_id in zip(texts, emotions):
    print(f"Text: {text}")
    print(f"Emotion: {emotion_map[emotion_id]}\n")
```

## Results & Insights

- All models achieved **>92% accuracy** with balanced precision and recall
- LSTM slightly outperforms GRU in precision (93.82% vs 93.39%)
- Models handle imbalanced classes well due to class weighting
- Joy and Sadness (majority classes) show best performance (>94% F1)
- Surprise (minority class) is most challenging (~83% F1)
- Early stopping prevented overfitting (models converged in 4-6 epochs)

## Project Structure

```
emotion-detection/
│
├── data/
│   └── text.csv              # DVC-tracked dataset
│
├── Models/
│   ├── lstm.keras            # DVC-tracked LSTM model
│   └── gru.keras             # DVC-tracked GRU v1 model
│
├── data.dvc                  # DVC pointer file for dataset
├── models.dvc                # DVC pointer file for models
├── mlruns/                   # MLflow experiment logs
├── Emotion Detection.ipynb   # Main notebook
├── .dvc/                     # DVC configuration
├── .dvcignore
├── requirements.txt
└── README.md
```

## Future Improvements

- [ ] Implement attention mechanisms (Bahdanau/Luong attention)
- [ ] Experiment with Bidirectional LSTM/GRU layers
- [ ] Try Transformer-based models (BERT, RoBERTa)
- [ ] Use pre-trained embeddings (GloVe, Word2Vec, FastText)
- [ ] Apply data augmentation (backtranslation, synonym replacement)
- [ ] Implement ensemble methods (voting, stacking)
- [ ] Deploy as REST API using FastAPI/Flask
- [ ] Create web interface with Streamlit/Gradio
- [ ] Add multilingual support
- [ ] Perform error analysis on misclassified samples

## License

MIT License

## Author

[Your Name]

## Acknowledgments

- Dataset: [Source/Citation if applicable]
- MLflow for experiment tracking
- DVC for data and model versioning
