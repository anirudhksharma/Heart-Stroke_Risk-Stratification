# 🫀 Heart Stroke Risk Stratification

A deep learning project for predicting and stratifying heart stroke risk using PyTorch neural networks. The project explores multiple strategies for handling class imbalance in medical data, including weighted loss functions and hybrid resampling techniques (SMOTE + ENN).

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

Stroke is one of the leading causes of death and disability worldwide. Early identification of individuals at high risk can enable timely interventions and save lives. This project builds a binary classification model using PyTorch to predict whether a patient is at risk of stroke based on clinical and demographic features.

A key challenge in stroke prediction is **severe class imbalance** — in the dataset, only ~4.9% of patients experienced a stroke (249 out of 5,110 records). The project addresses this through two distinct approaches:

1. **Weighted Class Loss** — Using `BCEWithLogitsLoss` with `pos_weight` to penalize misclassifications of the minority class more heavily.
2. **SMOTE + ENN Resampling (SMOTEENN)** — Combining Synthetic Minority Oversampling Technique (SMOTE) with Edited Nearest Neighbors (ENN) to rebalance the dataset before training.

## Dataset

The project uses the [Healthcare Dataset Stroke Data](healthcare-dataset-stroke-data.csv), which contains **5,110 patient records** with the following **12 features**:

| Feature | Description | Type |
|---------|-------------|------|
| `id` | Unique patient identifier | Integer |
| `gender` | Patient gender (Male/Female/Other) | Categorical |
| `age` | Patient age | Continuous |
| `hypertension` | Whether patient has hypertension (0/1) | Binary |
| `heart_disease` | Whether patient has heart disease (0/1) | Binary |
| `ever_married` | Marital status (Yes/No) | Categorical |
| `work_type` | Type of employment (Private, Self-employed, Govt_job, children, Never_worked) | Categorical |
| `Residence_type` | Living area (Urban/Rural) | Categorical |
| `avg_glucose_level` | Average blood glucose level | Continuous |
| `bmi` | Body Mass Index | Continuous |
| `smoking_status` | Smoking status (formerly smoked, never smoked, smokes, Unknown) | Categorical |
| `stroke` | Target variable — whether the patient had a stroke (0/1) | Binary |

### Class Distribution

- **No Stroke (0):** 4,861 samples (95.1%)
- **Stroke (1):** 249 samples (4.9%)

## Project Structure

```
Heart-Stroke_Risk-Stratification/
├── Heart-Stroke-Risk-Prediction.ipynb    # Main prediction notebook (2 approaches)
├── healthcare-dataset-stroke-data.csv     # Dataset (5,110 records)
├── LICENSE                                # MIT License
└── README.md                              # This file
```

### Notebook

- **`Heart-Stroke-Risk-Prediction.ipynb`** — Comprehensive notebook that implements and compares two approaches for handling class imbalance:
  - **Approach 1:** Weighted classes using `BCEWithLogitsLoss(pos_weight=15)` with 10,000 training epochs
  - **Approach 2:** SMOTEENN resampling to balance the dataset, then training without weighted classes for 500 epochs

## Methodology

### Data Preprocessing

1. **Missing Value Handling:** BMI column contains missing values (`N/A`), imputed using **median** strategy via `SimpleImputer`
2. **Categorical Encoding:** All categorical features (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`) are encoded using `LabelEncoder`
3. **Feature Selection:** The `id` column is dropped as it's not a predictive feature
4. **Train-Test Split:** 80/20 split with stratification to preserve class distribution
5. **Feature Scaling:** All features are standardized using `StandardScaler`

### Class Imbalance Strategies

#### Approach 1: Weighted Loss Function
- Uses `BCEWithLogitsLoss` with a `pos_weight` parameter to assign higher penalty for false negatives
- `pos_weight` values explored: 10, 15, and computed ratio of negatives to positives
- Optimizer: SGD with learning rate 0.0001 and momentum 0.09

#### Approach 2: SMOTEENN Resampling
- Applies `SMOTEENN` from `imbalanced-learn` to generate synthetic minority samples and clean noisy majority samples
- After resampling: Training set grows from ~4,088 to ~6,734 samples with a more balanced distribution (3,010 negatives, 3,724 positives)
- Trains without weighted classes using standard `BCEWithLogitsLoss`
- Optimizer: SGD with learning rate 0.01 and momentum 0.09

## Model Architecture

Both notebooks use the same `BaseLineModel`, a fully connected neural network built with PyTorch:

```
BaseLineModel
├── LinearBlock1 (Sequential)
│   ├── Linear(11 → 32)
│   ├── Linear(32 → 32)
│   └── ReLU
├── LinearBlock2 (Sequential)
│   ├── Linear(32 → 16)
│   ├── Linear(16 → 1)
│   └── ReLU
```

- **Total Parameters:** 1,985 (all trainable)
- **Input:** 11 clinical features (standardized)
- **Output:** Single logit (passed through sigmoid for binary classification)
- **Loss:** `BCEWithLogitsLoss` (with or without `pos_weight`)
- **GPU Support:** CUDA acceleration when available

## Results

Model performance is evaluated using **accuracy** and **confusion matrices** visualized with Seaborn heatmaps.

### Approach 1 — Weighted Classes (Prediction Notebook)
Training with `pos_weight=15` for 10,000 epochs:
- Final Train Accuracy: ~61.33%
- Final Test Accuracy: ~61.35%
- Final Test Loss: ~0.8061

### Approach 2 — SMOTEENN (Prediction Notebook)
Training on resampled data for 500 epochs:
- Final Train Accuracy: ~42.81%
- Final Test Accuracy: ~42.52%
- Final Test Loss: ~0.3220

> **Note:** Accuracy alone may be misleading for imbalanced datasets. The confusion matrices in the notebooks provide a more nuanced view of model performance across both classes.

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, for accelerated training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anirudhksharma/Heart-Stroke_Risk-Stratification.git
   cd Heart-Stroke_Risk-Stratification
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn torch torchinfo imbalanced-learn matplotlib seaborn
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

4. **Open and run** the notebook:
   - `Heart-Stroke-Risk-Prediction.ipynb` for the full comparison of both approaches

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Preprocessing, train-test split, metrics |
| `torch` (PyTorch) | Neural network model building and training |
| `torchinfo` | Model architecture summary |
| `imbalanced-learn` | SMOTEENN resampling |
| `matplotlib` | Plotting |
| `seaborn` | Confusion matrix visualization |

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.