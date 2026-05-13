"""
RFM Segmentation Module

WHAT IT DOES:
- Segments customers using RFM-style scoring (Recency, Frequency, Monetary).
- Assigns customers to actionable segments: High Value, At Risk, Lost, Core.
- Produces segment-level summary statistics (size, churn rate, avg spend).

WHY WE DO THIS:
- Identify WHO is at risk vs who is valuable vs who already churned.
- Enable targeted retention strategies (e.g., save high-value at-risk customers).
- Prioritize resources on segments with highest impact potential.

WHAT IT TELLS ABOUT THE DATA:
- Distribution of customers across value/risk segments.
- Which segments have highest churn rates.
- Average spend and tenure patterns by segment.

HOW IT'S DONE:
- Recency = tenure (how long they've been a customer).
- Frequency = MonthlyCharges (proxy for usage intensity).
- Monetary = TotalCharges (lifetime value).
- Score each dimension (1-3) using quantile-based ranking.
- Apply business rules to assign segments:
  - Lost: already churned.
  - High Value: high monetary + high frequency, good recency.
  - At Risk (High Value): high monetary + high frequency, but low recency (newer).
  - Core: everyone else.

NOTE: This is a simplified RFM adapted for the Telco dataset (no event-level data).
"""
from __future__ import annotations

import pandas as pd

from .report_types import RfmReport


def compute_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM proxy features from Telco data.
    
    RFM mapping:
    - Recency = tenure (how long customer has been with us).
    - Frequency = MonthlyCharges (proxy for usage/engagement).
    - Monetary = TotalCharges (lifetime value).
    """
    out = df[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]].copy()
    out["recency"] = out["tenure"]
    out["frequency"] = out["MonthlyCharges"]
    out["monetary"] = out["TotalCharges"]
    return out


def segment_rfm(df: pd.DataFrame) -> RfmReport:
    """
    Segment customers using RFM scores and business rules.
    
    Returns:
    - RfmReport with segment summary table and per-customer segment assignments.
    """
    feats = compute_rfm_features(df)

    # Score each RFM dimension (1=low, 2=medium, 3=high) using quantile-based ranking.
    feats["recency_score"] = pd.qcut(feats["recency"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
    feats["frequency_score"] = pd.qcut(feats["frequency"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
    feats["monetary_score"] = pd.qcut(feats["monetary"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)

    def assign_segment(row: pd.Series) -> str:
        """
        Assign customer to a segment based on RFM scores and churn status.
        
        Segment logic:
        - Lost: already churned (Churn=1).
        - High Value: high monetary + high frequency, good recency (established customers).
        - At Risk (High Value): high monetary + high frequency, but low recency (newer, risky).
        - Core: everyone else.
        """
        if int(row["Churn"]) == 1:
            return "Lost"
        if row["monetary_score"] == 3 and row["frequency_score"] >= 2:
            if row["recency_score"] <= 2:
                return "At Risk (High Value)"
            return "High Value"
        return "Core"

    feats["segment"] = feats.apply(assign_segment, axis=1)

    # Aggregate segment-level statistics for business review.
    seg_table = (
        feats.groupby("segment")
        .agg(
            customers=("customerID", "count"),
            churn_rate=("Churn", "mean"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
            avg_total_charges=("TotalCharges", "mean"),
            avg_tenure=("tenure", "mean"),
        )
        .reset_index()
        .sort_values(["customers"], ascending=False)
    )

    # Per-customer segment assignments for targeting/filtering.
    customer_segments = feats[["customerID", "segment", "Churn", "tenure", "MonthlyCharges", "TotalCharges"]].copy()

    return RfmReport(segment_table=seg_table, customer_segments=customer_segments)
