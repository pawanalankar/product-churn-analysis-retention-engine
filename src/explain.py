"""
SHAP Explainability Module

WHAT IT DOES:
- Computes SHAP (SHapley Additive exPlanations) values for the churn prediction model.
- Identifies which features drive churn predictions (global feature importance).
- Extracts top churn drivers for the CPO memo and dashboard.

WHY WE DO THIS:
- Answer "WHY do customers churn?" with model-based evidence.
- Translate black-box ML predictions into actionable business insights.
- Prioritize which factors to address (e.g., "high MonthlyCharges is the #1 churn driver").

WHAT IT TELLS ABOUT THE DATA:
- Global feature importance: which features matter most for churn predictions.
- Directionality: whether high/low values of a feature increase/decrease churn risk.
- Segment-specific drivers: e.g., "for Month-to-month contracts, PaymentMethod matters more".

HOW IT'S DONE:
- Use SHAP TreeExplainer (optimized for tree-based models like XGBoost).
- Compute SHAP values on a sample of test data (for speed).
- SHAP values represent each feature's contribution to the prediction for each customer.
- Aggregate: mean absolute SHAP = global feature importance.
- Sort features by importance and extract top k drivers.

NOTE: SHAP is model-agnostic and provides consistent, theoretically grounded explanations.
Unlike simple feature importance, SHAP accounts for feature interactions.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from .report_types import ModelReport, ShapReport


def compute_shap(model_report: ModelReport, X_sample: pd.DataFrame) -> ShapReport:
    """
    Compute SHAP values for a sample of customers.
    
    Steps:
    1. Transform X_sample using the fitted preprocessor (one-hot encoding, etc.).
    2. Convert to dense array (SHAP TreeExplainer requires dense input).
    3. Compute SHAP values using TreeExplainer.
    4. Aggregate to global feature importance (mean absolute SHAP).
    
    Returns:
    - ShapReport with SHAP values, feature names, and mean_abs_shap DataFrame.
    """
    # Transform features using the fitted preprocessor.
    X_trans = model_report.preprocessor.transform(X_sample)

    # Convert sparse matrix to dense if needed (SHAP requires dense arrays).
    if hasattr(X_trans, "toarray"):
        X_trans_dense = X_trans.toarray()
    else:
        X_trans_dense = np.asarray(X_trans)

    # Compute SHAP values using TreeExplainer (fast for tree-based models).
    explainer = shap.TreeExplainer(model_report.model)
    shap_values = explainer.shap_values(X_trans_dense)

    # Use feature names from the preprocessor (post one-hot encoding).
    feature_names = model_report.feature_names
    if not feature_names:
        feature_names = [f"f{i}" for i in range(X_trans_dense.shape[1])]

    # Handle multi-class output (XGBoost sometimes returns 3D array).
    # For binary classification, we want SHAP values for class 1 (churn).
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    # Compute global feature importance: mean absolute SHAP across all samples.
    # Higher mean |SHAP| = more important feature globally.
    mean_abs = np.mean(np.abs(sv), axis=0)
    mean_abs_shap = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )

    return ShapReport(shap_values=shap_values, feature_names=feature_names, mean_abs_shap=mean_abs_shap)


def top_shap_drivers(shap_report: ShapReport, k: int = 10) -> pd.DataFrame:
    """
    Extract top k churn drivers (features with highest mean |SHAP|).
    
    Used by the CPO memo and dashboard to highlight key insights.
    """
    return shap_report.mean_abs_shap.head(k).reset_index(drop=True)
