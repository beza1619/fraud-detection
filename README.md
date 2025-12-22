# Fraud Detection Project

## Project Overview
This project implements fraud detection models for e-commerce and bank transactions, addressing key challenges like class imbalance and geolocation analysis.

## Features Implemented
- 🌍 **Geolocation integration**: IP to country mapping with fraud rate analysis
- ⚖️ **SMOTE implementation**: Handles 9.7:1 class imbalance in training data
- 🏦 **Complete credit card analysis**: Analyzed 284,807 bank transactions
- 📈 **Bivariate analysis**: Key feature relationships with fraud
- 🤖 **Model comparison**: Logistic Regression vs Random Forest
- 🔍 **Business insights**: Actionable fraud prevention recommendations

## Project Structure
fraud-detection/
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Processed data
├── notebooks/
│ └── fraud_detection_complete.ipynb # Main analysis notebook
├── src/
│ ├── init.py
│ └── fraud_utils.py # Helper functions
├── scripts/ # Python scripts
├── models/ # Saved models and visualizations
├── requirements.txt # Dependencies
└── README.md # This file
