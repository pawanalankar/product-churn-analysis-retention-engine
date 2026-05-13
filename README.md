# Telco Product Analytics + Churn Retention Engine

## What this project does

This repository is an end-to-end **product analytics system** built on the IBM Telco Customer Churn dataset. It is designed to answer the full set of product-retention questions:

- **Where** users drop off (lifecycle funnel)
- **When** users churn (cohort retention)
- **How long** users survive (Kaplan–Meier survival analysis)
- **Who** will churn (XGBoost churn prediction)
- **Why** they churn (SHAP explainability)
- **What to do about it** (auto-generated CPO recommendation memo)

The primary interface is a **Streamlit dashboard** (`app.py`) that orchestrates the full pipeline and renders each analysis as a dedicated tab.

## Run

1. Create and activate a virtual environment (recommended).
2. Install dependencies (use the same interpreter you will run Streamlit with):

```bash
pip install -r requirements.txt
```

3. Start the dashboard:

```bash
streamlit run app.py
```

### Running with a specific Python (pyenv / Python 3.12)

If you are using `pyenv` (or have multiple Python installations), run Streamlit via the interpreter explicitly:

```bash
/path/to/python -m pip install -r requirements.txt
/path/to/python -m streamlit run app.py
```

This avoids the common issue where packages are installed into one Python environment but Streamlit is executed by another.

## Dataset

This project reads the dataset from the repo root:

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The CSV file is not modified; all transformations happen on DataFrames in code.

## How it works (end-to-end)

The pipeline is intentionally modular. Each module in `src/` is responsible for one analytical “question”, and `app.py` stitches them together.

### 1) Data loading + cleaning (`src/io.py`)

- Loads the raw CSV into a DataFrame.
- Standardizes key numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`).
- Handles missing `TotalCharges` (commonly blank for brand-new customers) using an imputation/drop policy.
- Converts `Churn` from `Yes/No` to `1/0`.

Output:

- Cleaned DataFrame
- `DataQualityReport` (row counts, churn rate, early churn rate, missingness, notes)

### 2) Lifecycle funnel (where users drop off) (`src/funnel.py`)

The funnel is a product-analytics abstraction over a static dataset. Customers are marked into stages using simple business rules:

- `signup`: all customers
- `activated`: tenure ≥ 2 months
- `engaged`: activated AND (MonthlyCharges ≥ threshold OR has Internet)
- `retained`: engaged AND tenure ≥ 12 months

Stages are **cumulative** (each stage is a subset of the prior stage), so drop-offs represent true funnel attrition.

Output:

- Funnel table with counts and conversion rates

### 3) Cohort retention (when users churn) (`src/cohorts.py`)

Because we do not have explicit signup dates, we use **tenure as a proxy** to define cohorts (e.g., `0-6`, `6-12`, `12-24` months). For each cohort, the system computes retention rates at multiple horizons (1/3/6/12/… months).

Output:

- Retention matrix used for a retention heatmap

### 4) Survival analysis (how long users survive) (`src/survival.py`)

Kaplan–Meier survival curves estimate:

- `P(customer survives beyond month t)`

This correctly handles **censoring** (customers who have not churned yet), which makes it more appropriate than naive churn rates for “time-to-churn” questions.

Output:

- Overall survival curve
- Optional survival curves by group (e.g., by `Contract`)
- Survival probability snapshots at key horizons (used in the memo)

### 5) Churn prediction (who will churn) (`src/model.py`)

The predictive model is an XGBoost classifier trained on the cleaned dataset.

- Categorical features are one-hot encoded.
- Numeric features are passed through.
- The model is evaluated with ROC-AUC, PR-AUC, precision/recall/F1.
- A decision threshold is chosen using a simple, explicit rule (maximize F1).

Output:

- `ModelReport` containing the fitted pipeline, evaluation metrics, threshold, and test predictions
- Customer risk scores (probability of churn)

### 6) Explainability (why churn happens) (`src/explain.py`)

SHAP values are computed for a sample of customers to:

- Rank global churn drivers (mean absolute SHAP)
- Support a causal-seeming narrative with model-based evidence (while still being observational)

Output:

- Top churn driver list
- SHAP objects that can be plotted in the dashboard

### 7) CPO memo (what to do about it) (`src/memo.py`)

The memo generator pulls together metrics from all modules and formats an executive-ready markdown memo:

- Problem size (churn rate, early churn)
- Key insights (largest funnel drop, worst retention cohort, survival at horizons)
- Churn drivers (top SHAP features)
- Recommendations + measurement plan
- Model readiness metrics

Output:

- `MemoReport` containing the memo markdown

## Dashboard (`app.py`)

`app.py` is the single orchestration layer. It:

- Loads and cleans the dataset
- Runs each analysis module
- Renders tables/plots and the memo in Streamlit tabs

If a computation fails (e.g., SHAP due to missing dependencies), the dashboard surfaces the error so you can fix the environment.

## Repository structure

- `app.py`: Streamlit dashboard entry point
- `src/`: analysis modules
  - `io.py`: load/clean + data quality report
  - `funnel.py`: lifecycle funnel metrics
  - `cohorts.py`: cohort retention matrix
  - `survival.py`: Kaplan–Meier survival
  - `model.py`: churn prediction model
  - `explain.py`: SHAP explainability
  - `memo.py`: CPO recommendation memo
  - `config.py`: shared constants and thresholds
  - `report_types.py`: dataclasses for standardized outputs
- `requirements.txt`: Python dependencies
- `notebooks/01_end_to_end.ipynb`: development / QA notebook

## Interpreting results (product lens)

- **Funnel**: highlights lifecycle bottlenecks (activation/engagement/retention). A large drop suggests a product or packaging issue at that stage.
- **Retention heatmap**: shows which tenure bands retain better and when retention decays.
- **Survival curves**: compare expected lifetime across plans (e.g., Contract types). Separation indicates segments with structurally different churn dynamics.
- **Model scores**: identify a ranked list of “customers to save now”.
- **SHAP drivers**: prioritize what to fix (pricing, contract structure, service bundle, payment method friction).

## Troubleshooting

- If you see `ModuleNotFoundError` (e.g., `seaborn`), ensure you are installing and running using the same interpreter:

```bash
/path/to/python -m pip install -r requirements.txt
/path/to/python -m streamlit run app.py
```

