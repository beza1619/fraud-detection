# Fraud Detection System for E-commerce and Banking

## Project Overview
This project implements fraud detection models for Adey Innovations Inc. to identify fraudulent transactions in both e-commerce and banking domains. The system handles class imbalance, includes geolocation analysis, and provides model explainability.

## Repository Structure
fraud-detection/
├── data/ # Data files (.gitignored)
├── notebooks/ # Jupyter notebooks
│ ├── eda-fraud-data.ipynb # EDA for fraud data
│ ├── eda-creditcard.ipynb # EDA for credit card data
│ ├── feature-engineering.ipynb # Feature engineering
│ ├── modeling.ipynb # Model training
│ └── fraud_detection_complete.ipynb # Complete analysis
├── src/ # Source code modules
├── models/ # Saved model artifacts
├── scripts/ # Execution scripts
├── tests/ # Unit tests
├── requirements.txt # Python dependencies
└── README.md # This file

## Tasks Completed

### Task 1: Data Analysis and Preprocessing
- Data cleaning (missing values, duplicates, data types)
- Exploratory Data Analysis (EDA) with visualizations
- Geolocation integration (IP to country mapping)
- Feature engineering (time_since_signup, transaction frequency)
- Handling class imbalance with SMOTE

### Task 2: Model Building and Training
- Stratified train-test split preserving class distribution
- Baseline model: Logistic Regression
- Ensemble model: Random Forest with hyperparameter tuning
- 5-fold stratified cross-validation
- Model evaluation using AUC-PR, F1-Score, Confusion Matrix
- Model comparison and selection

### Task 3: Model Explainability
- Feature importance analysis from Random Forest
- SHAP framework for model interpretability
- Business recommendations based on insights
- Individual prediction analysis framework

## Results Summary
| Model | AUC-ROC (Fraud) | AUC-PR (Fraud) | AUC-ROC (Credit) | AUC-PR (Credit) |
|-------|----------------|----------------|------------------|-----------------|
| Logistic Regression | 0.7784 | 0.6418 | 0.9730 | 0.7284 |
| Random Forest | 0.85 | 0.72 | 0.985 | 0.80 |

**Selected Model:** Random Forest for production deployment

## Setup Instructions
1. Clone repository:
   ```bash
   git clone https://github.com/beza1619/fraud-detection.git
   cd fraud-detection