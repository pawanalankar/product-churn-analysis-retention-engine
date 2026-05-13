"""
CPO Memo Generator Module

WHAT IT DOES:
- Synthesizes all analysis outputs into a business decision memo.
- Auto-populates metrics from computed results (no hard-coded numbers).
- Structures insights into: Problem → Insights → Drivers → Recommendations → Measurement.

WHY WE DO THIS:
- Translate technical analysis into executive-ready business recommendations.
- Provide a clear, actionable narrative for product/retention strategy decisions.
- Ensure recommendations are data-driven and tied to measurable outcomes.

WHAT IT TELLS ABOUT THE DATA:
- Overall churn problem magnitude (churn rate, early churn).
- Where/when customers drop off (funnel, retention cohorts, survival).
- Why customers churn (SHAP drivers).
- What to do about it (targeted interventions, experiments, KPIs).

HOW IT'S DONE:
- Extract key metrics from each analysis module:
  - Data quality: churn rate, early churn rate.
  - Funnel: largest drop-off stage.
  - Retention: worst-performing cohort at 12 months.
  - Survival: survival probabilities at 3/6/12 months.
  - SHAP: top 5 churn drivers.
  - Model: ROC-AUC, PR-AUC, operating threshold.
- Format as markdown memo with sections:
  - Problem statement (quantified).
  - Key insights (where/when churn happens).
  - Churn drivers (why it happens).
  - Recommendations (what to do: interventions, experiments).
  - Measurement plan (KPIs, A/B test design).
  - Model readiness (confidence in predictions).

NOTE: This memo is the MOST IMPORTANT deliverable. It turns analysis into business action.
"""
from __future__ import annotations

import pandas as pd

from .report_types import (
    DataQualityReport,
    FunnelReport,
    MemoReport,
    ModelReport,
    RetentionReport,
    ShapReport,
    SurvivalReport,
)
from .survival import survival_at_horizons


def _largest_funnel_drop(funnel_table: pd.DataFrame) -> tuple[str, str, int]:
    """
    Identify the largest drop-off between consecutive funnel stages.
    
    Returns:
    - (from_stage, to_stage, drop_count)
    """
    ft = funnel_table.copy()
    ft["prev_count"] = ft["count"].shift(1)
    ft["drop"] = (ft["prev_count"] - ft["count"]).fillna(0).astype(int)

    if len(ft) <= 1:
        return ("signup", "signup", 0)

    idx = int(ft["drop"].idxmax())
    if idx == 0:
        return ("signup", "signup", 0)

    from_stage = str(ft.loc[idx - 1, "stage"])
    to_stage = str(ft.loc[idx, "stage"])
    drop = int(ft.loc[idx, "drop"])
    return from_stage, to_stage, drop


def generate_cpo_memo(
    *,
    data_quality: DataQualityReport,
    funnel: FunnelReport,
    retention: RetentionReport,
    survival: SurvivalReport,
    model: ModelReport,
    shap: ShapReport,
) -> MemoReport:
    """
    Generate CPO recommendation memo from all analysis outputs.
    
    The memo is structured as:
    1. Problem (churn rates).
    2. Key insights (where/when churn happens).
    3. Survival metrics (how long customers last).
    4. Churn drivers (why churn happens, from SHAP).
    5. Recommendations (what to do: interventions).
    6. Measurement plan (KPIs, experiments).
    7. Model readiness (confidence in predictions).
    
    Returns:
    - MemoReport with memo_markdown (ready to display/export).
    """
    churn_rate = data_quality.churn_rate
    early_churn = data_quality.early_churn_rate

    from_stage, to_stage, drop = _largest_funnel_drop(funnel.funnel_table)

    retention_snapshot = retention.retention_matrix.copy()
    worst_cohort = None
    worst_12m = None
    if 12 in retention_snapshot.columns:
        worst_row = retention_snapshot[12].idxmin()
        worst_cohort = str(worst_row)
        worst_12m = float(retention_snapshot.loc[worst_row, 12])

    survival_overall_h = survival_at_horizons(survival.overall_survival)

    top_drivers = shap.mean_abs_shap.head(5)
    drivers_text = "\n".join([f"- {r.feature}: {r.mean_abs_shap:.4f}" for r in top_drivers.itertuples(index=False)])

    memo = f"""# CPO Memo — Telco Churn & Retention

## 1) Problem
- Current churn rate is **{churn_rate:.1%}**.
- Early churn (tenure ≤ 3 months) churn rate is **{early_churn:.1%}**.

## 2) Key insights
- Largest lifecycle drop-off occurs from **{from_stage} → {to_stage}** (drop of **{drop:,}** users).
"""

    if worst_cohort is not None and worst_12m is not None:
        memo += f"- The weakest retention cohort is **{worst_cohort}** with **{worst_12m:.1%}** retained at 12 months (tenure-based proxy).\n"

    memo += "\n## 3) Survival (how long users survive)\n"
    for r in survival_overall_h.itertuples(index=False):
        memo += f"- Survival probability at **{int(r.horizon_months)} months**: **{float(r.survival_prob):.1%}**\n"

    memo += f"""

## 4) Churn drivers (why churn happens)
Top model drivers (mean |SHAP|):
{drivers_text}

## 5) Recommendations
- Create a targeted retention playbook for the highest-risk users (based on model score and contract/service segments).
- Add an early-tenure intervention (first 30–90 days): onboarding reinforcement and proactive support nudges.
- Test price/plan packaging for high MonthlyCharges segments with low tenure.

## 6) Measurement plan
- Primary KPI: churn rate (overall and tenure ≤ 3 months).
- Guardrails: ARPU / MonthlyCharges distribution, support contact rates.
- Experiment: A/B test retention offers and onboarding interventions; measure lift vs control.

## 7) Model readiness
- ROC-AUC: **{model.metrics.get('roc_auc', float('nan')):.3f}**
- PR-AUC: **{model.metrics.get('pr_auc', float('nan')):.3f}**
- Operating threshold: **{model.threshold:.2f}**
"""

    return MemoReport(memo_markdown=memo)
