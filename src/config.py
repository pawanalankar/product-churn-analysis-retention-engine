"""
Configuration Module

WHAT IT DOES:
- Centralizes all project constants and configuration parameters.
- Defines file paths, random seeds, and analysis thresholds.
- Ensures consistency across all modules (no magic numbers scattered in code).

WHY WE DO THIS:
- Single source of truth for tunable parameters.
- Easy to adjust thresholds/bins without editing multiple files.
- Reproducibility: fixed random seed ensures consistent train/test splits.

WHAT IT TELLS ABOUT THE DATA:
- Dataset location (CSV path).
- Lifecycle stage definitions (funnel rules).
- Cohort and retention bucketing strategy.
- Key survival horizons for reporting.

HOW IT'S DONE:
- Simple Python module with constants.
- Imported by other modules as needed.
- To change analysis parameters, edit this file and re-run.
"""
from __future__ import annotations


DATASET_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_SEED = 42

FUNNEL_RULES = {
    "activated_min_tenure": 2,
    "engaged_min_monthly_charges": 50.0,
    "retained_min_tenure": 12,
}
COHORT_BINS = [0, 6, 12, 24, 48, 72]
RETENTION_HORIZONS = [1, 3, 6, 12, 24, 36, 48, 60, 72]
SURVIVAL_HORIZONS = [3, 6, 12]
