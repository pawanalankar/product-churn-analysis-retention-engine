"""
Report Types Module

WHAT IT DOES:
- Defines strongly-typed result objects (dataclasses) returned by each analysis module.
- Ensures consistent data structures across the pipeline.
- Makes the Streamlit app and memo generator easier to maintain.

WHY WE DO THIS:
- Type safety: catch errors at development time, not runtime.
- Self-documenting: each report's fields clearly show what the analysis produces.
- Consistency: all modules return structured objects, not ad-hoc tuples/dicts.

WHAT IT TELLS ABOUT THE DATA:
- Each dataclass represents the output of one analysis:
  - DataQualityReport: churn rates, missing data, cleaning notes.
  - FunnelReport: funnel table + optional segment breakdowns.
  - RetentionReport: retention matrix + cohort definition.
  - SurvivalReport: survival curves (overall + by group).
  - RfmReport: segment summary + per-customer assignments.
  - ModelReport: trained pipeline, metrics, test predictions.
  - ShapReport: SHAP values + feature importance.
  - MemoReport: final CPO memo as markdown.

HOW IT'S DONE:
- Use Python dataclasses with frozen=True (immutable).
- Each analysis module imports the relevant report type.
- The Streamlit app and memo generator consume these reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    n_rows: int
    churn_rate: float
    early_churn_rate: float
    missing_total_charges: int
    rows_dropped: int
    notes: str


@dataclass(frozen=True)
class FunnelReport:
    funnel_table: pd.DataFrame
    segment_table: Optional[pd.DataFrame]


@dataclass(frozen=True)
class RetentionReport:
    retention_matrix: pd.DataFrame
    cohort_definition: str


@dataclass(frozen=True)
class SurvivalReport:
    overall_survival: pd.DataFrame
    by_group: Optional[dict[str, pd.DataFrame]]
    group_col: Optional[str]


@dataclass(frozen=True)
class RfmReport:
    segment_table: pd.DataFrame
    customer_segments: pd.DataFrame


@dataclass(frozen=True)
class ModelReport:
    pipeline: Any
    preprocessor: Any
    model: Any
    feature_names: list[str]
    metrics: dict[str, float]
    threshold: float
    X_test: pd.DataFrame
    y_test: pd.Series
    y_proba_test: pd.Series


@dataclass(frozen=True)
class ShapReport:
    shap_values: Any
    feature_names: list[str]
    mean_abs_shap: pd.DataFrame


@dataclass(frozen=True)
class MemoReport:
    memo_markdown: str
