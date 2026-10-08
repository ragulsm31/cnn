# COVID-19 Chest X-Ray Classification Using CNN

A TensorFlow/Keras deep-learning project that classifies chest X-ray images into three classes:

- Covid
- Normal
- Viral Pneumonia

This repository is based on the uploaded Kaggle notebook and has been reorganized into a reproducible GitHub project. The original notebook used Kaggle-specific paths; this version uses a local `data/` directory and a reproducible KaggleHub download script.

> **Medical disclaimer:** This is an educational/research project, not a medical diagnostic system. Predictions must not be used for clinical decisions.

## Project workflow

```text
Kaggle dataset
      ↓
KaggleHub download
      ↓
Image preprocessing + augmentation
      ↓
CNN training
      ↓
Validation / test evaluation
      ↓
Saved Keras model
      ↓
Single-image prediction
```

## CNN architecture

- Input: 224 × 224 × 3
- Conv2D: 32 filters + MaxPooling
- Conv2D: 64 filters + MaxPooling
- Conv2D: 128 filters + MaxPooling
- Conv2D: 64 filters + MaxPooling
- Flatten
- Dense: 128
- Dense: 64
- Softmax output: 3 classes
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Training epochs: 3 (matching the original notebook baseline)

## Dataset

Dataset: `pranavraikokte/covid19-image-dataset` on Kaggle.

The dataset contains the `train/` and `test/` directory structure with the three classes used by this project. **The dataset is intentionally not committed to this repository.** Download it using the provided script.

Kaggle dataset page:
https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

## Repository structure

```text
covid19-cnn-classification/
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── covid19_cnn_classification.ipynb
├── outputs/
│   └── .gitkeep
├── src/
│   ├── download_data.py
│   ├── train.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/covid19-cnn-classification.git
cd covid19-cnn-classification
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python src/download_data.py
```

Kaggle authentication may be required depending on your KaggleHub setup.

### 5. Train the CNN

```bash
python src/train.py
```

The trained model will be saved as:

```text
models/covid_cnn.keras
```

### 6. Predict a single image

Example:

```bash
python src/predict.py --image "data/Covid19-dataset/test/Viral Pneumonia/0112.jpeg"
```

Example output:

```text
Predicted class: Viral Pneumonia
Confidence: XX.XX%
```

## Jupyter Notebook

Open:

```text
notebooks/covid19_cnn_classification.ipynb
```

The notebook contains:

1. Dataset setup
2. Sample image visualization
3. Image preprocessing
4. Data augmentation
5. CNN architecture
6. Model training
7. Evaluation
8. Model saving
9. Single-image prediction

## Important implementation notes

The original notebook applied augmentation to the test generator. This repository keeps augmentation for the training data and uses only rescaling for the test data, which gives a cleaner evaluation pipeline.

The original notebook also predicted an image without applying the same `1/255` scaling used during training. The GitHub version applies the same preprocessing during inference.

## Results

Do not hard-code an accuracy value in this README. Run the training script and report the actual test accuracy from your own run, because results depend on the environment and training configuration.

## Technologies

- Python
- TensorFlow / Keras
- CNN
- NumPy
- OpenCV
- Matplotlib
- KaggleHub
- Jupyter Notebook

## Resume-ready project description

**COVID-19 Chest X-Ray Classification using CNN** — Built a TensorFlow/Keras convolutional neural network to classify chest X-ray images into Covid, Normal, and Viral Pneumonia classes. Implemented image preprocessing, augmentation, CNN feature extraction, model training, evaluation, model serialization, and single-image inference using a reproducible KaggleHub-based data pipeline.

## Disclaimer

This project demonstrates computer-vision and deep-learning techniques for educational purposes. It is not validated for clinical use and should not be treated as a diagnostic tool.
