from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.config import DATASET_PATH
from src.cohorts import compute_retention_matrix
from src.explain import compute_shap, top_shap_drivers
from src.funnel import compute_funnel
from src.io import clean_telco, load_telco
from src.memo import generate_cpo_memo
from src.model import score_customers, train_model
from src.rfm import segment_rfm
from src.survival import fit_km_by_group


st.set_page_config(page_title="Telco Product Analytics", layout="wide")


@st.cache_data
def load_and_clean() -> tuple[pd.DataFrame, object]:
    df_raw = load_telco(DATASET_PATH)
    df_clean, dq = clean_telco(df_raw, total_charges_policy="impute")
    return df_clean, dq


@st.cache_resource
def build_model(df: pd.DataFrame):
    return train_model(df)


st.title("Telco Product Analytics + Retention Engine")

df, dq = load_and_clean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{dq.n_rows:,}")
col2.metric("Churn rate", f"{dq.churn_rate:.1%}")
col3.metric("Early churn (≤3m)", f"{dq.early_churn_rate:.1%}")
col4.metric("TotalCharges missing (raw)", f"{dq.missing_total_charges:,}")

st.caption(dq.notes)

tab_overview, tab_funnel, tab_retention, tab_survival, tab_rfm, tab_model, tab_memo = st.tabs(
    ["Overview", "Funnel", "Retention", "Survival", "RFM", "Predict & Explain", "CPO Memo"]
)

with tab_overview:
    st.subheader("Overview")

    st.markdown(
        """
This dashboard is built on a *static snapshot* of customers. The KPIs at the top summarize the cleaned dataset:

- **Customers**: number of rows after cleaning.
- **Churn rate**: share of customers with `Churn = 1`.
- **Early churn (≤3m)**: churn rate among customers with `tenure <= 3` months (a proxy for onboarding/activation risk).
- **TotalCharges missing (raw)**: count of missing/blank `TotalCharges` values in the *raw* CSV before applying the cleaning policy.
        """
    )

    st.subheader("Data preview")
    st.dataframe(df.head(25), width="stretch")

    st.subheader("Churn breakdown by Contract")
    if "Contract" in df.columns:
        by_contract = (
            df.groupby("Contract", dropna=False)
            .agg(customers=("Churn", "size"), churn_rate=("Churn", "mean"))
            .reset_index()
            .sort_values("customers", ascending=False)
        )
        st.dataframe(by_contract, width="stretch")

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(by_contract["Contract"].astype(str), by_contract["churn_rate"].astype(float))
        ax.set_ylabel("Churn rate")
        ax.set_xlabel("Contract")
        ax.set_title("Churn rate by Contract")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Column 'Contract' not found in dataset.")

    st.subheader("Churn breakdown by tenure bucket")
    tenure_bins = [0, 3, 6, 12, 24, 48, 72]
    tenure_labels = [
        "0-3",
        "3-6",
        "6-12",
        "12-24",
        "24-48",
        "48-72",
    ]
    tenure_bucket = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True, right=True)
    by_tenure = (
        df.assign(tenure_bucket=tenure_bucket.astype(str))
        .groupby("tenure_bucket", dropna=False)
        .agg(customers=("Churn", "size"), churn_rate=("Churn", "mean"))
        .reset_index()
    )
    st.dataframe(by_tenure, width="stretch")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(by_tenure["tenure_bucket"].astype(str), by_tenure["churn_rate"].astype(float))
    ax.set_ylabel("Churn rate")
    ax.set_xlabel("Tenure bucket (months)")
    ax.set_title("Churn rate by tenure bucket")
    plt.setp(ax.get_xticklabels(), rotation=0)
    st.pyplot(fig, clear_figure=True)

with tab_funnel:
    st.subheader("Lifecycle Funnel")
    segment_by = st.selectbox("Segment funnel by", ["(none)", "Contract", "InternetService", "PaymentMethod"], index=0)
    seg = None if segment_by == "(none)" else segment_by
    funnel = compute_funnel(df, segment_by=seg)

    st.dataframe(funnel.funnel_table, width="stretch")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(funnel.funnel_table["stage"], funnel.funnel_table["count"])
    ax.set_ylabel("Users")
    ax.set_xlabel("Stage")
    ax.set_title("Funnel Drop-Off")
    st.pyplot(fig, clear_figure=True)

    if funnel.segment_table is not None:
        st.subheader("Funnel by segment")
        st.dataframe(funnel.segment_table, width="stretch")

with tab_retention:
    st.subheader("Cohort Retention Heatmap (tenure-based proxy)")
    retention = compute_retention_matrix(df)
    st.dataframe(retention.retention_matrix, width="stretch")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(retention.retention_matrix, annot=False, cmap="Blues", ax=ax)
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Cohort (tenure band)")
    st.pyplot(fig, clear_figure=True)

with tab_survival:
    st.subheader("Survival Analysis (Kaplan–Meier)")
    group_col = st.selectbox("Compare survival by", ["Contract", "InternetService", "PaymentMethod"], index=0)
    survival = fit_km_by_group(df, group_col=group_col)

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, sf in (survival.by_group or {}).items():
        ax.plot(sf["timeline"], sf["survival_prob"], label=label)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"Survival by {group_col}")
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)

with tab_rfm:
    st.subheader("RFM-style Segmentation")
    rfm = segment_rfm(df)
    st.dataframe(rfm.segment_table, width="stretch")

with tab_model:
    st.subheader("Churn Prediction + Explainability")

    model_report = build_model(df)
    st.write(model_report.metrics)

    scores = score_customers(model_report, df)
    st.subheader("Highest-risk customers")
    st.dataframe(scores.head(50), width="stretch")

    st.subheader("Top churn drivers (SHAP)")
    X_sample = model_report.X_test.sample(n=min(500, len(model_report.X_test)), random_state=42)

    try:
        shap_report = compute_shap(model_report, X_sample)
        top = top_shap_drivers(shap_report, k=15)
        st.dataframe(top, width="stretch")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(list(reversed(top["feature"])), list(reversed(top["mean_abs_shap"])))
        ax.set_xlabel("mean |SHAP|")
        ax.set_ylabel("feature")
        ax.set_title("Top drivers")
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        shap_report = None

with tab_memo:
    st.subheader("CPO Recommendation Memo")

    funnel = compute_funnel(df, segment_by=None)
    retention = compute_retention_matrix(df)
    survival = fit_km_by_group(df, group_col="Contract")
    model_report = build_model(df)

    X_sample = model_report.X_test.sample(n=min(500, len(model_report.X_test)), random_state=42)
    shap_report = compute_shap(model_report, X_sample)

    memo = generate_cpo_memo(
        data_quality=dq,
        funnel=funnel,
        retention=retention,
        survival=survival,
        model=model_report,
        shap=shap_report,
    )

    st.markdown(memo.memo_markdown)
