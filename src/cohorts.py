"""
Cohort Retention Analysis Module

WHAT IT DOES:
- Groups customers into tenure-based cohorts (e.g., 0-6 months, 7-12 months).
- Computes retention rates at different time horizons (e.g., 3, 6, 12 months).
- Produces a retention matrix (cohorts × horizons) for heatmap visualization.

WHY WE DO THIS:
- Understand WHEN customers churn (timing patterns).
- Identify which cohorts have better/worse retention.
- Spot critical drop-off periods (e.g., "most churn happens between months 6-12").

WHAT IT TELLS ABOUT THE DATA:
- Retention curves by customer lifecycle stage.
- Whether early-stage customers (low tenure) churn faster than mature customers.
- Which cohorts need intervention (e.g., new users vs long-term users).

HOW IT'S DONE:
- Use pd.cut() to bin customers by tenure into cohorts.
- For each horizon (e.g., 12 months), compute % of cohort with tenure >= horizon.
- Aggregate into a matrix: rows = cohorts, columns = horizons, values = retention %.
- This matrix is visualized as a heatmap in the Streamlit dashboard.

NOTE: Since we don't have actual signup dates, we use tenure as a proxy for cohort assignment.
"""
from __future__ import annotations

import pandas as pd

from .config import COHORT_BINS, RETENTION_HORIZONS
from .report_types import RetentionReport


def build_cohort_labels(df: pd.DataFrame) -> pd.Series:
    """
    Assign each customer to a tenure-based cohort.
    
    Example: if COHORT_BINS = [0, 6, 12, 24, 48, 72], cohorts are:
    - "0-6" (brand new)
    - "6-12" (early stage)
    - "12-24" (established)
    - etc.
    """
    bins = COHORT_BINS
    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    return pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True, right=True)


def compute_retention_matrix(df: pd.DataFrame) -> RetentionReport:
    """
    Compute retention matrix: cohorts × horizons.
    
    For each cohort and horizon, compute:
    retention_rate = % of customers in cohort with tenure >= horizon.
    
    Returns:
    - RetentionReport with the retention matrix (DataFrame) and cohort definition.
    """
    cohort = build_cohort_labels(df)
    horizons = [h for h in RETENTION_HORIZONS if h <= int(df["tenure"].max())]
    if not horizons:
        horizons = [1]

    # For each horizon, create a binary flag: 1 if customer tenure >= horizon, 0 otherwise.
    matrix = {}
    for h in horizons:
        matrix[h] = (df["tenure"] >= h).astype(int)

    tmp = pd.DataFrame(matrix)
    tmp["cohort"] = cohort.astype(str)

    # Group by cohort and compute mean (= retention rate) for each horizon.
    # Result: rows = cohorts, columns = horizons, values = retention %.
    retention = tmp.groupby("cohort")[horizons].mean().sort_index()

    return RetentionReport(
        retention_matrix=retention,
        cohort_definition=f"tenure bands using bins={COHORT_BINS}",
    )
