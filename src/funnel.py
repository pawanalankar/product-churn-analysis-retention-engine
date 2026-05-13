from __future__ import annotations

from typing import Optional

import pandas as pd

from .config import FUNNEL_RULES
from .report_types import FunnelReport


def build_funnel_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    activated_min_tenure = int(FUNNEL_RULES["activated_min_tenure"])
    engaged_min_monthly_charges = float(FUNNEL_RULES["engaged_min_monthly_charges"])
    retained_min_tenure = int(FUNNEL_RULES["retained_min_tenure"])

    # Lifecycle stage: Activation
    # Business meaning: the customer made it past initial setup and stayed long enough
    # to be considered "activated".
    # Implementation: tenure >= activated_min_tenure (months).
    df["activated"] = df["tenure"] >= activated_min_tenure

    has_internet = df.get("InternetService")
    if has_internet is not None:
        internet_ok = df["InternetService"].astype(str).str.lower() != "no"
    else:
        internet_ok = False

    # Lifecycle stage: Engagement
    # Business meaning: the activated customer shows signs of meaningful usage/value.
    # This dataset has no event logs, so we use a proxy definition.
    # Implementation (proxy):
    # - MonthlyCharges >= engaged_min_monthly_charges OR
    # - has an InternetService that is not "No".
    # Funnel behavior: engagement is cumulative, so only activated customers can be engaged.
    df["engaged"] = df["activated"] & ((df["MonthlyCharges"] >= engaged_min_monthly_charges) | internet_ok)

    # Lifecycle stage: Retention
    # Business meaning: the customer has persisted for a meaningful duration.
    # Implementation: tenure >= retained_min_tenure (months).
    # Funnel behavior: retention is cumulative, so only engaged customers can be retained.
    df["retained"] = df["engaged"] & (df["tenure"] >= retained_min_tenure)

    return df


def _funnel_table_from_flags(df_flags: pd.DataFrame) -> pd.DataFrame:
    # Funnel counts are computed from boolean masks.
    # Assumption: stage flags are cumulative (each stage is a subset of the previous stage).
    stages = [
        ("signup", pd.Series([True] * len(df_flags), index=df_flags.index)),
        ("activated", df_flags["activated"]),
        ("engaged", df_flags["engaged"]),
        ("retained", df_flags["retained"]),
    ]

    rows = []
    prev_count = None
    signup_count = len(df_flags)

    for stage_name, mask in stages:
        count = int(mask.sum())
        pct_of_signup = (count / signup_count) if signup_count else 0.0
        if prev_count is None:
            step_conversion = 1.0
        else:
            step_conversion = (count / prev_count) if prev_count else 0.0
        prev_count = count

        rows.append(
            {
                "stage": stage_name,
                "count": count,
                "pct_of_signup": pct_of_signup,
                "step_conversion": step_conversion,
            }
        )

    return pd.DataFrame(rows)


def compute_funnel(df: pd.DataFrame, segment_by: Optional[str] = None) -> FunnelReport:
    df_flags = build_funnel_flags(df)
    funnel_table = _funnel_table_from_flags(df_flags)

    segment_table = None
    if segment_by:
        if segment_by not in df_flags.columns:
            raise ValueError(f"segment_by='{segment_by}' not found in df")

        parts = []
        for seg_value, seg_df in df_flags.groupby(segment_by, dropna=False):
            seg_table = _funnel_table_from_flags(seg_df)
            seg_table.insert(0, "segment", str(seg_value))
            parts.append(seg_table)

        segment_table = pd.concat(parts, ignore_index=True) if parts else None

    return FunnelReport(funnel_table=funnel_table, segment_table=segment_table)
