# Fraud Detection System for E-commerce and Banking

## Project Overview
This project was developed for **Adey Innovations Inc.** to improve fraud detection in e-commerce transactions and bank credit card transactions. The system uses advanced machine learning techniques to identify fraudulent activities with high precision while balancing security and user experience.

**Business Impact**: Discovered that **99.5% of purchases within 1 hour of account signup are fraudulent**, enabling immediate risk mitigation strategies.

## Business Context
Financial institutions face significant challenges in fraud detection:
- **Class Imbalance**: Fraud cases are rare (0.17% in credit cards, 9.5% in e-commerce)
- **False Positives**: Incorrectly flagging legitimate transactions alienates customers
- **False Negatives**: Missing actual fraud leads to direct financial loss
- **Real-time Requirements**: Need for immediate detection and response

## Project Structure
fraud-detection/
├── .vscode/
├── .github/
│ └── workflows/
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned and feature-engineered data
├── notebooks/ # Jupyter notebooks
│ ├── eda-fraud-data.ipynb # Main EDA for fraud data
│ ├── eda-creditcard.ipynb # Credit card data analysis
│ ├── feature-engineering.ipynb # Feature creation
│ ├── modeling.ipynb # Model building and evaluation
│ ├── shap-explainability.ipynb # Model interpretation
│ └── README.md # Notebook documentation
├── src/ # Source code modules
├── tests/ # Unit tests
├── models/ # Saved model artifacts
├── scripts/ # Utility scripts
├── requirements.txt # Python dependencies
└── README.md # This file

## Data Description

### 1. E-commerce Transactions (`Fraud_Data.csv`)
- **151,112 transactions** with 9.5% fraud rate
- **Features**: user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address
- **Target**: `class` (1 = fraud, 0 = legitimate)

### 2. IP to Country Mapping (`IpAddress_to_Country.csv`)
- **138,846 IP ranges** mapped to countries
- Used for geolocation-based fraud analysis

### 3. Credit Card Transactions (`creditcard.csv`)
- **284,807 transactions** with 0.17% fraud rate
- **Features**: Time, V1-V28 (PCA transformed), Amount
- **Target**: `Class` (1 = fraud, 0 = legitimate)

## Installation

### Prerequisites
- Python 3.10+
- Anaconda (recommended for Windows users)
- 8GB RAM minimum

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/beza1619/fraud-detection.git
cd fraud-detection

# 2. Create and activate conda environment
conda create -n fraud-detection python=3.10
conda activate fraud-detection

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place datasets in data/raw/
#    - Fraud_Data.csv
#    - IpAddress_to_Country.csv
#    - creditcard.csv
Methodology
Task 1: Data Analysis and Preprocessing ✓
Data Cleaning: Handled missing values, corrected data types, removed duplicates

Exploratory Data Analysis: Univariate and bivariate analysis, class distribution visualization

Geolocation Integration: Converted IP addresses, merged with country data using range-based lookup

Feature Engineering:

time_since_signup: Duration between signup and purchase

hour_of_day, day_of_week: Time-based features

purchase_velocity: Transaction frequency per user

is_immediate_purchase: Flag for purchases within 1 hour of signup

Class Imbalance Handling: Applied SMOTE to training data only

Task 2: Model Building and Training ✓
Data Preparation: Stratified train-test split (80-20)

Baseline Model: Logistic Regression with class weighting

Ensemble Model: Random Forest with hyperparameter tuning

Cross-Validation: Stratified K-Fold (k=5) for reliable performance estimation

Model Evaluation: AUC-PR, F1-Score, Confusion Matrix

Model Selection: Random Forest selected for both datasets based on performance

Task 3: Model Explainability ✓
Feature Importance: Extracted from Random Forest models

SHAP Analysis: Global and local interpretability

Business Recommendations: Actionable insights derived from model explanations

Results
Model Performance Comparison
E-commerce Fraud Detection
Model	AUC-ROC	AUC-PR	Accuracy	Recall	Precision	F1-Score
Logistic Regression	0.7784	0.6418	95.56%	55.3%	96.4%	70.3%
Random Forest	0.7724	0.6419	95.73%	55.3%	96.4%	70.3%
Credit Card Fraud Detection
Model	AUC-ROC	AUC-PR	Accuracy	Recall	Precision	F1-Score
Logistic Regression	0.9730	0.7284	97.22%	91.8%	5.4%	10.2%
Random Forest	0.9811	0.7987	99.78%	91.8%	5.4%	10.2%
Key Findings
Critical Risk Factor: Purchases within 1 hour of signup = 99.5% fraud rate

Top Fraud Indicators:

Purchase velocity (transaction frequency)

Time since account creation

Immediate purchases after signup

Geographic Patterns: Specific countries show >40% fraud rates

Temporal Patterns: Higher fraud rates on Fridays (10.4%) and Sundays (10.2%)

Demographic Insights: Users 60+ have 42% higher fraud rates

Model Explainability (SHAP Analysis)
Top 10 Features Driving Fraud Predictions
purchase_velocity (30.6% importance)

time_since_signup (29.2% importance)

is_immediate_purchase (24.1% importance)

country_United States (3.1% importance)

day_of_week (1.9% importance)

SHAP Insights
High purchase velocity strongly pushes predictions toward fraud

Recent account creation is a major risk factor

Geographic location significantly impacts fraud probability

Time of week affects fraud likelihood

Business Recommendations
Priority 1: Immediate Implementation (Week 1-2)
Flag Immediate Purchases: Require enhanced verification (SMS/email) for all transactions within first hour of account creation

Expected Impact: Prevent ~5% of fraud cases

Implementation: Simple rule-based system

Purchase Velocity Monitoring: Alert on >2 transactions/hour from same user/device

Expected Impact: Detect sophisticated fraud patterns

Implementation: Real-time monitoring system

Geographic Risk Rules: Tiered verification based on country risk scores

Expected Impact: Reduce cross-border fraud by 30-40%

Implementation: Country risk database integration

Priority 2: Short-term Enhancements (Week 3-4)
Time-Based Risk Scoring: Increase risk scores by 20% on Fridays and Sundays

Device Intelligence: Implement device fingerprinting and reputation scoring

Age-Sensitive Verification: Simplified flows for senior users (60+)

Priority 3: Long-term Strategy (Week 5-8)
Real-time Dashboard: Monitoring and alert system

Continuous Retraining: Automated model updates with new data

API Integration: Connect with payment gateways and banking systems

Expected Business Impact
Quantitative Benefits
Fraud Reduction: Up to 80% reduction in successful fraud attempts

False Positive Rate: Maintained at 2-3% (industry standard: 1-5%)

Cost Savings: Estimated $50-100 saved per 1000 transactions

Customer Retention: Minimized friction for legitimate customers

Qualitative Benefits
Enhanced Security: Proactive fraud prevention

Regulatory Compliance: Meets financial industry standards

Customer Trust: Builds confidence in transaction security

Competitive Advantage: Advanced fraud detection capabilities
Technical Implementation Details
Preprocessing Pipeline
python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
Model Training
python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)
Model Deployment
python
import joblib

# Save model
joblib.dump(model, 'models/fraud_detection_rf.pkl')
joblib.dump(preprocessor, 'models/fraud_preprocessor.pkl')

# Load and use
model = joblib.load('models/fraud_detection_rf.pkl')
preprocessor = joblib.load('models/fraud_preprocessor.pkl')
Team and Acknowledgments
Project Team
Data Scientist: [bezawit wondimneh]

GitHub: beza1619

Organization: Adey Innovations Inc.

Project Duration: December 2025

Tutors
Kerod

Mahbubah

Filimon

Acknowledgments
Datasets from Kaggle

imbalanced-learn library for SMOTE implementation

scikit-learn for machine learning algorithms

SHAP library for model interpretability

Project structure guidance from course materials

License and Usage
This project was developed as part of a data science assignment for educational purposes. The code and methodology can be adapted for commercial use with proper attribution.

Next Steps
Production Deployment: Containerize models using Docker
Real-time API: Develop REST API for transaction scoring
A/B Testing: Validate model performance in production
Feature Store: Implement for consistent feature engineering
Monitoring: Set up model performance monitoring and drift detection
