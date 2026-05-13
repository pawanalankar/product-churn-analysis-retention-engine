"""
Survival Analysis Module (Kaplan-Meier)

WHAT IT DOES:
- Estimates survival curves: probability that a customer survives (doesn't churn) over time.
- Compares survival across groups (e.g., Contract types, InternetService types).
- Extracts survival probabilities at key horizons (3, 6, 12 months) for the CPO memo.

WHY WE DO THIS:
- Understand HOW LONG customers typically stay before churning.
- Identify which product/plan groups have better/worse retention over time.
- Quantify the impact of contract types, services, or payment methods on customer lifetime.

WHAT IT TELLS ABOUT THE DATA:
- Median survival time (if defined): "50% of customers churn by month X".
- Survival probability at key milestones (e.g., "80% survive past 6 months").
- Comparative retention: "Two-year contracts have 90% survival at 12 months vs 60% for month-to-month".

HOW IT'S DONE:
- Use the Kaplan-Meier estimator from the lifelines library.
- Duration = tenure (months since signup).
- Event = Churn (1 if churned, 0 if still active/censored).
- Fit overall survival curve and group-specific curves.
- Extract survival probabilities at predefined horizons for reporting.

NOTE: Kaplan-Meier handles censoring (customers still active) correctly, unlike naive churn rate calculations.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from lifelines import KaplanMeierFitter

from .config import SURVIVAL_HORIZONS
from .report_types import SurvivalReport


def fit_km(df: pd.DataFrame) -> SurvivalReport:
    """
    Fit overall Kaplan-Meier survival curve.
    
    Returns:
    - SurvivalReport with survival_function DataFrame (timeline, survival_prob).
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["tenure"], event_observed=df["Churn"])

    sf = kmf.survival_function_.reset_index()
    sf.columns = ["timeline", "survival_prob"]

    return SurvivalReport(overall_survival=sf, by_group=None, group_col=None)


def fit_km_by_group(df: pd.DataFrame, group_col: str) -> SurvivalReport:
    """
    Fit Kaplan-Meier survival curves by group (e.g., Contract type).
    
    Returns:
    - SurvivalReport with overall curve + dict of group-specific curves.
    """
    overall = fit_km(df).overall_survival

    by_group: Dict[str, pd.DataFrame] = {}
    for group_value, g in df.groupby(group_col, dropna=False):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=g["tenure"], event_observed=g["Churn"], label=str(group_value))
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["timeline", "survival_prob"]
        by_group[str(group_value)] = sf

    return SurvivalReport(overall_survival=overall, by_group=by_group, group_col=group_col)


def survival_at_horizons(survival_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract survival probabilities at key horizons (e.g., 3, 6, 12 months).
    
    Used by the CPO memo to report concrete survival metrics.
    """
    horizons = SURVIVAL_HORIZONS
    out = []
    for h in horizons:
        eligible = survival_df.loc[survival_df["timeline"] <= h]
        if len(eligible) == 0:
            prob = float("nan")
        else:
            prob = float(eligible.iloc[-1]["survival_prob"])
        out.append({"horizon_months": int(h), "survival_prob": prob})
    return pd.DataFrame(out)
