"""
Churn Prediction Model Module (XGBoost)

WHAT IT DOES:
- Trains a machine learning model to predict WHO will churn.
- Preprocesses features (one-hot encoding for categoricals, passthrough for numerics).
- Evaluates model performance with multiple metrics (ROC-AUC, PR-AUC, precision, recall, F1).
- Selects an optimal decision threshold to balance precision and recall.
- Scores customers to produce churn risk rankings.

WHY WE DO THIS:
- Proactively identify high-risk customers before they churn.
- Enable targeted retention interventions (e.g., offers, support outreach).
- Quantify the predictability of churn (model performance = confidence in predictions).

WHAT IT TELLS ABOUT THE DATA:
- Which customers are most likely to churn (risk scores).
- Model performance: how well we can predict churn (ROC-AUC, PR-AUC).
- Precision/recall tradeoff: how many false positives vs false negatives at a given threshold.

HOW IT'S DONE:
- Split data into train/test (stratified by churn to preserve class balance).
- Preprocess:
  - Categorical columns → one-hot encoded.
  - Numeric columns (tenure, MonthlyCharges, TotalCharges, SeniorCitizen) → passthrough.
- Train XGBoost classifier (gradient boosted trees).
- Evaluate on test set with multiple metrics.
- Choose decision threshold by maximizing F1 score on precision-recall curve.
- Return ModelReport with pipeline, metrics, and test predictions.

NOTE: XGBoost is chosen for its strong performance on tabular data and built-in handling of missing values.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .config import RANDOM_SEED
from .report_types import ModelReport


def _split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).
    
    Drops customerID (not a predictive feature) and Churn (target).
    """
    X = df.drop(columns=["Churn"], errors="ignore").copy()
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])
    y = df["Churn"].astype(int)
    return X, y


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline for features.
    
    - Categorical columns: one-hot encode (handle_unknown="ignore" for robustness).
    - Numeric columns: passthrough (no scaling needed for tree-based models).
    """
    numeric_cols: List[str] = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"] if c in X.columns]
    categorical_cols: List[str] = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor


def _choose_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Choose optimal decision threshold by maximizing F1 score.
    
    F1 balances precision and recall, which is appropriate for churn prediction
    where both false positives (wasted retention spend) and false negatives
    (missed at-risk customers) are costly.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    precision = precision[:-1]
    recall = recall[:-1]

    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    if len(thresholds) == 0:
        return 0.5

    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def train_model(df: pd.DataFrame) -> ModelReport:
    """
    Train XGBoost churn prediction model with full preprocessing pipeline.
    
    Steps:
    1. Split features/target and train/test sets (stratified).
    2. Build preprocessing pipeline (one-hot encoding + passthrough).
    3. Train XGBoost classifier.
    4. Evaluate on test set with multiple metrics.
    5. Choose optimal threshold (max F1).
    6. Return ModelReport with pipeline, metrics, and test predictions.
    """
    X, y = _split_features_target(df)

    # Stratified split to preserve class balance in train/test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    preprocessor = _build_preprocessor(X_train)

    # XGBoost hyperparameters tuned for churn prediction on Telco data.
    # Conservative settings to avoid overfitting (low learning rate, regularization).
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss",
    )

    # Build and fit the full pipeline (preprocessing + model).
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    # Predict probabilities on test set (probability of churn = class 1).
    y_proba = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=X_test.index)
    
    # Choose optimal threshold by maximizing F1 score.
    threshold = _choose_threshold(y_test.to_numpy(), y_proba.to_numpy())
    y_pred = (y_proba >= threshold).astype(int)

    # Compute evaluation metrics.
    # ROC-AUC: overall discriminative power (higher is better, 0.5 = random).
    # PR-AUC: precision-recall area (better for imbalanced classes).
    # Accuracy: overall correctness (can be misleading if classes are imbalanced).
    # Precision: % of predicted churners who actually churn (low false positives).
    # Recall: % of actual churners we correctly identify (low false negatives).
    # F1: harmonic mean of precision and recall.
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Extract feature names after one-hot encoding (needed for SHAP).
    try:
        feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        feature_names = []

    return ModelReport(
        pipeline=pipeline,
        preprocessor=pipeline.named_steps["preprocessor"],
        model=pipeline.named_steps["model"],
        feature_names=feature_names,
        metrics=metrics,
        threshold=float(threshold),
        X_test=X_test,
        y_test=y_test,
        y_proba_test=y_proba,
    )


def score_customers(model_report: ModelReport, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all customers in df with churn probabilities and predictions.
    
    Returns:
    - DataFrame with customerID, churn_proba, churn_pred, sorted by risk (descending).
    """
    X = df.drop(columns=["Churn"], errors="ignore").copy()
    ids = df["customerID"] if "customerID" in df.columns else pd.Series(range(len(df)), index=df.index)

    X_no_id = X.drop(columns=["customerID"], errors="ignore")
    proba = pd.Series(model_report.pipeline.predict_proba(X_no_id)[:, 1], index=df.index)
    pred = (proba >= model_report.threshold).astype(int)

    out = pd.DataFrame({"customerID": ids.astype(str), "churn_proba": proba, "churn_pred": pred})
    out = out.sort_values("churn_proba", ascending=False).reset_index(drop=True)
    return out
