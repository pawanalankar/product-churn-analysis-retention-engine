# Telco Product Analytics + Churn System — End-to-End Plan

This plan delivers an end-to-end, reproducible product analytics system (drop-offs, churn timing, churn drivers, churn prediction, actions) culminating in a CPO-ready memo generated from the computed results.

## 0) Scope, dataset, and success criteria

### Dataset
- **Input CSV**: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (already in repo root)
- **Expected columns (verified header)**:
  - `customerID`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`, service add-ons, `Churn` (Yes/No), etc.

### Dataset handling policy (important)
- The original CSV file will remain unchanged.
- All data engineering, cleaning, feature creation, and transformations will be performed on DataFrames created in code after loading the CSV.

### Primary deliverables (must ship)
- **Funnel analysis (SQL-thinking)**
  - A funnel drop-off table + chart.
  - A “where users drop off” narrative with segment breakdowns.
- **Cohort retention heatmap**
  - Tenure-based cohorts and retention by month buckets.
- **Survival analysis (Kaplan–Meier)**
  - Overall survival curve.
  - Group comparisons (at minimum: `Contract`, optionally `InternetService`, `PaymentMethod`).
- **Churn prediction model (XGBoost + SHAP)**
  - Robust preprocessing, train/test split, evaluation, calibration/thresholding.
  - SHAP summary plot + top drivers + segment-level driver story.
- **CPO Recommendation Memo (most important)**
  - A crisp 1–2 page memo with:
    - Problem statement
    - Key insights (quantified)
    - Root cause hypothesis
    - Recommendations + experiments
    - Expected impact + measurement plan

### Definition of done (acceptance checks)
- **Reproducible run**: one command runs the dashboard locally.
- **All charts appear** in Streamlit:
  - Funnel bar
  - Retention heatmap
  - Survival curve(s)
  - SHAP summary (or fallback bar plot if SHAP rendering is limited)
- **Memo is generated** from the current run’s computed metrics (no hard-coded numbers).

## 1) Repository structure (what will be created)

- `requirements.txt`
- `data/` (optional) 
  - (Either move CSV here or keep in root; code will support a configurable path.)
- `src/` (pure-python analysis library)
  - `config.py` (paths, constants like bins)
  - `io.py` (load + clean + type coercions)
  - `funnel.py` (flag creation + funnel aggregation + plots)
  - `cohorts.py` (cohort bucketing + retention matrix + heatmap)
  - `rfm.py` (simple RFM-style segmentation + tables)
  - `survival.py` (KM fitting + group comparisons)
  - `model.py` (preprocess + XGBoost train/eval + threshold selection)
  - `explain.py` (SHAP computation + driver extraction)
  - `memo.py` (assemble memo text from outputs)
  - `report_types.py` (dataclasses for passing structured results)
- `app.py` (Streamlit dashboard entry point)
- `notebooks/01_end_to_end.ipynb` (optional, mirrors the app’s pipeline)

## 1.1) Detailed scaffold (exact files + minimal public APIs)

### Root
- `requirements.txt`
  - Pins runtime dependencies (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `lifelines`, `matplotlib`, `seaborn`, `streamlit`, optional `duckdb`).
- `README.md`
  - Run instructions:
    - create venv
    - `pip install -r requirements.txt`
    - `streamlit run app.py`
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`
  - Source dataset (kept in repo root by default; optionally moved to `data/`).

### Optional data folder
- `data/`
  - If you prefer, we’ll support moving the CSV to `data/telco_churn.csv` via config.

### `src/` package
- `src/__init__.py`
  - Marks `src` as an importable package.
- `src/config.py`
  - **Responsibilities**: centralize file paths and analysis constants.
  - **Exports** (minimal):
    - `DATASET_PATH: str`
    - `FUNNEL_RULES: dict[str, object]` (thresholds, tenure cutoffs)
    - `COHORT_BINS: list[int]`
    - `RETENTION_HORIZONS: list[int]` (e.g., 3/6/12/24 months)
    - `RANDOM_SEED: int`
- `src/report_types.py`
  - **Responsibilities**: strongly-typed return objects so the app can render everything consistently.
  - **Exports** (dataclasses):
    - `DataQualityReport`
    - `FunnelReport`
    - `RetentionReport`
    - `SurvivalReport`
    - `RfmReport`
    - `ModelReport`
    - `ShapReport`
    - `MemoReport`
- `src/io.py`
  - **Responsibilities**: load + clean + encode core fields.
  - **Exports**:
    - `load_telco(path: str) -> pd.DataFrame`
    - `clean_telco(df: pd.DataFrame, total_charges_policy: str) -> tuple[pd.DataFrame, DataQualityReport]`
- `src/funnel.py`
  - **Responsibilities**: simulate lifecycle funnel + segment breakdown.
  - **Exports**:
    - `build_funnel_flags(df: pd.DataFrame) -> pd.DataFrame`
    - `compute_funnel(df: pd.DataFrame, segment_by: str | None = None) -> FunnelReport`
- `src/cohorts.py`
  - **Responsibilities**: tenure-based cohorts + retention matrix.
  - **Exports**:
    - `build_cohort_labels(df: pd.DataFrame) -> pd.Series`
    - `compute_retention_matrix(df: pd.DataFrame) -> RetentionReport`
- `src/rfm.py`
  - **Responsibilities**: simple, explainable segmentation tables.
  - **Exports**:
    - `compute_rfm_features(df: pd.DataFrame) -> pd.DataFrame`
    - `segment_rfm(df: pd.DataFrame) -> RfmReport`
- `src/survival.py`
  - **Responsibilities**: Kaplan–Meier curves + group comparisons.
  - **Exports**:
    - `fit_km(df: pd.DataFrame) -> SurvivalReport`
    - `fit_km_by_group(df: pd.DataFrame, group_col: str) -> SurvivalReport`
- `src/model.py`
  - **Responsibilities**: preprocessing + XGBoost training + evaluation + threshold selection.
  - **Exports**:
    - `train_model(df: pd.DataFrame) -> ModelReport`
    - `score_customers(model_report: ModelReport, df: pd.DataFrame) -> pd.DataFrame`
- `src/explain.py`
  - **Responsibilities**: SHAP computation + driver extraction.
  - **Exports**:
    - `compute_shap(model_report: ModelReport, X_sample: pd.DataFrame) -> ShapReport`
    - `top_shap_drivers(shap_report: ShapReport, k: int = 10) -> pd.DataFrame`
- `src/memo.py`
  - **Responsibilities**: generate the CPO memo using computed metrics (no hard-coded numbers).
  - **Exports**:
    - `generate_cpo_memo(*, data_quality: DataQualityReport, funnel: FunnelReport, retention: RetentionReport, survival: SurvivalReport, model: ModelReport, shap: ShapReport) -> MemoReport`

### Streamlit app
- `app.py`
  - **Responsibilities**: orchestrate pipeline, provide filters, render plots, render memo.
  - Tabs:
    - Overview / Funnel / Retention / Survival / Predict & Explain / CPO Memo

### Notebook (optional)
- `notebooks/01_end_to_end.ipynb`
  - Mirrors `app.py` pipeline for development/QA.

## 2) Environment setup

### Python dependencies
- Core:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `xgboost`
  - `shap`
  - `lifelines`
  - `streamlit`
- Optional but helpful:
  - `duckdb` (for “SQL-thinking” funnel and segment queries)

### Run targets
- **Dashboard**: `streamlit run app.py`
- Optional: notebook run for development/QA.

## 3) Data loading + cleaning (critical foundation)

### Known Telco quirks to handle
- `TotalCharges` is often read as string and may contain blanks.
  - Coerce with `pd.to_numeric(..., errors="coerce")` and decide a policy:
    - Drop rows with missing `TotalCharges`, or
    - Impute with `MonthlyCharges * tenure` (documented).
- `Churn` is `Yes/No`.
  - Convert to `1/0` for modeling.
- Ensure `tenure` is numeric (months) and non-negative.

### Decisions for this implementation
- `TotalCharges` handling: use **imputation** when `TotalCharges` becomes missing after coercion.
  - Default imputation: `TotalCharges = MonthlyCharges * tenure`.
  - Safety rule: apply imputation primarily to very low-tenure rows (e.g., `tenure == 0`); for any unexpected missingness outside that, prefer dropping those rows and reporting how many were removed.

### Outputs
- A `df_clean` DataFrame and a data quality summary:
  - row count
  - missingness
  - churn rate overall

## 4) Funnel analysis (SQL-thinking)

### Goal
Answer: **Where users drop off?** Even though Telco isn’t event-level, we simulate a lifecycle funnel using rules.

### Funnel definition (configurable)
- `signup`: all customers
- `activated`: `tenure >= 2`
- `engaged`: `MonthlyCharges >= X` (default X=50) OR “has internet service”
- `retained`: `tenure >= 12`
- (Optional) `high_value`: `TotalCharges` in top quantile

### Computation (two interchangeable implementations)
- **Pandas** aggregation.
- **DuckDB SQL** (preferred for “SQL-thinking”):
  - Create a view on the dataframe and compute counts by stage, and by segments (e.g., `Contract`).

### Outputs
- Funnel table with counts and conversion rates stage-to-stage.
- Bar chart of drop-offs.
- Segment drill-downs:
  - funnel by `Contract`
  - funnel by `InternetService`

## 5) Cohort retention (heatmap)

### Goal
Answer: **When do users churn?** (timing + patterns)

### Cohort logic
- Use tenure bands to simulate cohorts (since we lack start dates):
  - Example bins: `0–6`, `7–12`, `13–24`, `25–48`, `49–72` (months)
- Create time buckets (columns) as tenure month ranges (or fixed 6-month periods) and compute:
  - `% retained` ≈ share of users in cohort with tenure above bucket threshold

### Outputs
- Retention matrix DataFrame.
- Seaborn heatmap with annotated percentages.
- Key narrative callouts:
  - earliest bucket with steepest decay
  - cohorts that retain better/worse

## 6) RFM-style segmentation (actionability layer)

### Goal
Answer: **Who is at risk / high value / lost?**

### Features (simple, explainable)
- `recency` = `tenure`
- `frequency` proxy = `MonthlyCharges` (or number of add-on services)
- `monetary` = `TotalCharges`

### Segment rules (explicit and tunable)
- **High value**: high `monetary` and high `frequency`, churn=0 (or low predicted risk later)
- **At risk**: high value but high predicted churn risk OR known risky contract types
- **Lost**: churn=1

### Outputs
- Segment sizes, churn rates, and top descriptive stats.

## 7) Survival analysis (Kaplan–Meier)

### Goal
Answer: **How long do users survive?** + compare survival across product/plan groups.

### Implementation
- Use `lifelines.KaplanMeierFitter` with:
  - `duration` = `tenure`
  - `event_observed` = churn flag (1 if churned)

### Group comparisons
- At minimum:
  - `Contract` (Month-to-month vs One year vs Two year)
- Optional:
  - `InternetService` (DSL vs Fiber vs None)
  - `PaymentMethod`

### Outputs
- Survival curve plot overall.
- Small multiples / overlay plots by group.
- Table:
  - median survival time by group (if defined)
  - survival probability at key horizons (3, 6, 12 months)

## 8) Churn prediction model (XGBoost)

### Goal
Answer: **Who will churn?** and provide a risk score for targeting.

### Data prep
- Target: `y = Churn (1/0)`
- Features:
  - Drop `customerID`
  - Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`
  - Categorical: all remaining object columns
- Preprocessing:
  - `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")`
  - Optional scaling for numeric (not required for trees, but fine if used consistently)

### Training
- Split: stratified train/test.
- Model: `xgboost.XGBClassifier` with reasonable defaults + basic tuning.
- Evaluation metrics (tell the product story):
  - ROC-AUC
  - PR-AUC (often more useful if churn is imbalanced)
  - Precision/recall at an operational threshold
  - Confusion matrix

### Thresholding (business decision)
- Add a simple threshold selection approach:
  - maximize F1 OR
  - choose recall target (e.g., 70% recall) and report precision
- Output: “top N high-risk users” table.

## 9) SHAP explainability (why churn happens)

### Goal
Answer: **Why they churn?** with credible model explanations.

### Outputs
- SHAP summary plot (global)
- Top features list with directionality
- Segment-specific insights:
  - drivers for Month-to-month
  - drivers for Fiber optic
  - drivers for high MonthlyCharges

### Story constraints
- Translate technical outputs into product language:
  - “Month-to-month contracts are associated with higher churn risk”
  - “High MonthlyCharges combined with low tenure is a churn hotspot”

## 10) Streamlit dashboard (make it look real)

### Pages / sections
- **Overview**
  - KPI tiles: churn rate, estimated early churn (<=3 months), median survival, model ROC-AUC
- **Funnel**
  - funnel chart + segment filters
- **Retention**
  - heatmap + cohort insights
- **Survival**
  - KM curve overall + group comparison selector
- **Predict & Explain**
  - model metrics
  - SHAP drivers
  - top-risk customer table (anonymized IDs)
- **CPO Memo**
  - memo rendered as markdown with “copy/paste ready” formatting

### UX requirements
- Fast enough for local runs (cache data + model artifacts with Streamlit caching).
- Controls: select segment, threshold, horizon.

## 11) CPO Recommendation Memo (auto-generated)

### Inputs (computed each run)
- Overall churn rate
- Early churn rate (<=3 months tenure)
- Biggest funnel drop-off (stage-to-stage delta)
- Worst cohorts and timing of decay
- Survival by contract type (survival at 3/6/12 months)
- Top 3 churn drivers from SHAP

### Memo structure (output)
1. **Problem**
   - Quantify churn and the business risk.
2. **Key insights**
   - Where drop-off occurs + when churn spikes.
3. **Root cause hypotheses**
   - Pricing sensitivity, contract structure, support/services, payment friction.
4. **Recommendations** (actionable, testable)
   - Example initiatives:
     - targeted offers for high-risk Month-to-month users
     - onboarding reinforcement for first 60–90 days
     - “save” playbook triggered by risk score
5. **Expected impact + measurement plan**
   - Success metrics, experiment design (A/B), and monitoring.

## 12) Execution milestones (practical timeline)

### Milestone A — Foundation + Analytics (Day 1)
- Data cleaning + QA
- Funnel + cohort retention heatmap
- Initial dashboard scaffolding

### Milestone B — Survival + Modeling (Day 2)
- KM survival overall + by groups
- XGBoost pipeline + baseline metrics

### Milestone C — Explainability + Decision (Day 3)
- SHAP insights + segment drill-down
- Memo generator connected to computed metrics
- Polish dashboard and ensure reproducibility

## 13) Risks / open questions (to resolve before implementation)
- Where you want the CSV to live long-term:
  - keep in repo root vs move to `data/` (plan supports either via config)
- Whether to impute or drop missing `TotalCharges` after coercion.
- Whether to treat the model as:
  - general churn propensity, or
  - “early churn” propensity (<=3 months) for activation/onboarding focus.

## 14) What you review/approve
- Confirm:
  - keep CSV in root or move to `data/`
  - missing `TotalCharges` policy (drop vs impute)
  - primary focus: overall churn vs early churn

