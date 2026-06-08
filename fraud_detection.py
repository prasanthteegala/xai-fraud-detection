# XAI Fraud Detection System
# Tools: XGBoost, LightGBM, SHAP, SMOTE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Load dataset
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df = pd.read_csv('creditcard.csv')

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- XGBoost Model ---
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_res, y_train_res)
xgb_preds = xgb_model.predict(X_test)
print("XGBoost Results:")
print(classification_report(y_test, xgb_preds))

# --- LightGBM Model ---
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
lgb_model.fit(X_train_res, y_train_res)
lgb_preds = lgb_model.predict(X_test)
print("LightGBM Results:")
print(classification_report(y_test, lgb_preds))

# --- Random Forest Model ---
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train_res, y_train_res)
rf_preds = rf_model.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, rf_preds))

# --- SHAP Explainability ---
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Local explanation for first prediction
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0]
)
