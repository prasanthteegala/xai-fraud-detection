# XAI Fraud Detection System

Explainable AI system for detecting credit card fraud using ensemble machine learning models and SHAP interpretability.

## Results
- 92% Recall on fraudulent transactions
- F1-Score: 0.89
- Dataset: Real credit card transactions (Kaggle)

## Tools & Technologies
- Python, XGBoost, LightGBM, Random Forest
- SMOTE for handling imbalanced data
- SHAP for model explainability
- Scikit-learn, Pandas, NumPy

## How It Works
1. Loads real credit card transaction data
2. Handles class imbalance using SMOTE
3. Trains 3 ensemble classifiers
4. Evaluates using recall and F1-score
5. Applies SHAP for global and local explainability

## Dataset
Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Author
Prasanth Goud Teegala — AI Data Engineer
