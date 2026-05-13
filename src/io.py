"""
Data Loading & Cleaning Module

WHAT IT DOES:
- Loads the Telco customer churn CSV into a pandas DataFrame.
- Cleans and standardizes data types for downstream analysis.
- Handles missing values in TotalCharges (a known data quality issue).
- Converts categorical churn labels to numeric (Yes/No → 1/0).

WHY WE DO THIS:
- Raw CSV data often has type inconsistencies (e.g., TotalCharges stored as string).
- Missing TotalCharges typically occurs for brand-new customers (tenure=0).
- Clean, typed data is essential for modeling and analysis pipelines.

WHAT IT TELLS ABOUT THE DATA:
- Overall churn rate (% of customers who churned).
- Early churn rate (churn among customers with tenure ≤ 3 months).
- Data quality issues (missing values, rows dropped).

HOW IT'S DONE:
- Load CSV with pandas.
- Coerce numeric columns (tenure, MonthlyCharges, TotalCharges) to proper types.
- Apply imputation policy for missing TotalCharges (default: impute for tenure=0).
- Convert Churn Yes/No to binary 1/0.
- Return cleaned DataFrame + DataQualityReport with metrics.

IMPORTANT: The original CSV file is never modified. All transformations happen on DataFrame copies.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd

from .report_types import DataQualityReport


TotalChargesPolicy = Literal["impute", "drop"]


def load_telco(path: str) -> pd.DataFrame:
    """Load the Telco CSV into a DataFrame. CSV file remains unchanged."""
    return pd.read_csv(path)


def clean_telco(df: pd.DataFrame, total_charges_policy: TotalChargesPolicy = "impute") -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Clean and standardize the Telco DataFrame.
    
    Cleaning steps:
    1. Convert tenure, MonthlyCharges, TotalCharges to numeric types.
    2. Handle missing TotalCharges using the specified policy.
    3. Convert Churn from Yes/No to 1/0 for modeling.
    4. Compute data quality metrics (churn rate, early churn rate).
    
    Returns:
    - Cleaned DataFrame (copy, original untouched).
    - DataQualityReport with metrics and notes.
    """
    df = df.copy()
    n_loaded = len(df)

    # Convert tenure to integer (months since signup).
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    
    # Convert MonthlyCharges to float (current monthly bill).
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    # TotalCharges is often stored as string with blanks for new customers.
    # Strip whitespace and convert blanks to NaN, then coerce to numeric.
    total_raw = df["TotalCharges"].astype(str).str.strip().replace({"": np.nan})
    df["TotalCharges"] = pd.to_numeric(total_raw, errors="coerce")

    missing_total_charges = int(df["TotalCharges"].isna().sum())
    rows_dropped = 0

    # Policy: how to handle missing TotalCharges.
    if total_charges_policy == "impute":
        # Impute missing TotalCharges for tenure=0 customers (brand new, no billing yet).
        # For these, TotalCharges = MonthlyCharges * tenure = 0.
        impute_mask = df["TotalCharges"].isna() & (df["tenure"] == 0)
        if impute_mask.any():
            df.loc[impute_mask, "TotalCharges"] = df.loc[impute_mask, "MonthlyCharges"] * df.loc[impute_mask, "tenure"]

        # Drop any remaining rows with missing TotalCharges (unexpected cases).
        still_missing = df["TotalCharges"].isna()
        rows_dropped = int(still_missing.sum())
        if rows_dropped:
            df = df.loc[~still_missing].copy()
    elif total_charges_policy == "drop":
        # Drop all rows with missing TotalCharges.
        rows_dropped = int(df["TotalCharges"].isna().sum())
        df = df.loc[df["TotalCharges"].notna()].copy()
    else:
        raise ValueError("total_charges_policy must be 'impute' or 'drop'")

    # Convert Churn from categorical (Yes/No) to binary (1/0) for modeling.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # Compute overall churn rate (what % of customers churned).
    churn_rate = float(df["Churn"].mean()) if len(df) else float("nan")

    # Compute early churn rate (churn among customers with tenure ≤ 3 months).
    # This is critical for understanding activation/onboarding issues.
    early_mask = df["tenure"] <= 3
    early_churn_rate = float(df.loc[early_mask, "Churn"].mean()) if early_mask.any() else float("nan")

    notes = f"Loaded {n_loaded} rows; dropped {rows_dropped} rows after TotalCharges handling. CSV remains unchanged."

    report = DataQualityReport(
        n_rows=int(len(df)),
        churn_rate=churn_rate,
        early_churn_rate=early_churn_rate,
        missing_total_charges=missing_total_charges,
        rows_dropped=rows_dropped,
        notes=notes,
    )

    return df, report
