import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

from feature_engineering import engineer_features
from predictor import prepare_inputs, predict_pd
from risk_engine import expected_loss, risk_bucket

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

if "models" not in st.session_state:
    st.error("⚠️ Models not loaded. Please launch via **app.py**.")
    st.stop()

models = st.session_state["models"]

st.title("📂 Batch Sequential Prediction")
st.markdown(
    "Upload a **CSV file** containing customer timelines. "
    "Each row is one month for one customer. "
    "The engine will score every customer's latest month."
)
st.divider()

# ── Expected columns ──────────────────────────────────────────────────────────
REQUIRED_COLS = [
    "customer_id", "month",
    "active_products_count", "credit_card_utilization",
    "total_monthly_obligation", "emi_amount", "days_to_emi",
    "emi_to_income_ratio", "salary_delay_days",
    "weekly_balance_change_pct", "atm_withdrawal_amount", "monthly_income",
]

with st.expander("📋 Expected CSV Format"):
    sample = pd.DataFrame([{
        "customer_id": "CUST_001", "month": 1,
        "active_products_count": 2, "credit_card_utilization": 0.45,
        "total_monthly_obligation": 15000, "emi_amount": 8000,
        "days_to_emi": 10, "emi_to_income_ratio": 0.30,
        "salary_delay_days": 2, "weekly_balance_change_pct": 0.02,
        "atm_withdrawal_amount": 3000, "monthly_income": 35000,
    }])
    st.dataframe(sample, use_container_width=True, hide_index=True)

    csv_bytes = sample.to_csv(index=False).encode()
    st.download_button("⬇️ Download Sample CSV", csv_bytes, "sample_input.csv", "text/csv")

st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📤 Upload your customer CSV", type=["csv"])

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)

    # Validate columns
    missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.stop()

    st.success(f"✅ Loaded {len(df_raw):,} rows across {df_raw['customer_id'].nunique():,} customers.")
    st.dataframe(df_raw.head(10), use_container_width=True, hide_index=True)

    st.divider()

    if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
        progress = st.progress(0, text="Running predictions…")
        results  = []
        customers = df_raw["customer_id"].unique()

        for idx, cid in enumerate(customers):
            df_cust = df_raw[df_raw["customer_id"] == cid].copy()
            df_eng  = engineer_features(df_cust)

            try:
                tree_input, lstm_input = prepare_inputs(models, df_eng)
                pd_score = predict_pd(models, tree_input, lstm_input)
            except Exception as e:
                pd_score = np.nan

            last = df_cust.iloc[-1]
            bucket = risk_bucket(pd_score) if not np.isnan(pd_score) else "ERROR"

            salary_flag = int(last["salary_delay_days"] > 5)
            util_flag   = int(last["credit_card_utilization"] > 0.75)

            try:
                el, lgd, ead = expected_loss(
                    models, pd_score,
                    last["emi_amount"], last["credit_card_utilization"],
                    last["monthly_income"], salary_flag, util_flag
                )
            except Exception:
                el = lgd = ead = np.nan

            results.append({
                "customer_id":           cid,
                "PD (%)":                round(pd_score * 100, 2) if not np.isnan(pd_score) else np.nan,
                "Risk Bucket":           bucket,
                "Expected Loss (₹)":     round(el, 2)  if not np.isnan(el)  else np.nan,
                "LGD":                   round(lgd, 4) if not np.isnan(lgd) else np.nan,
                "EAD (₹)":               round(ead, 2) if not np.isnan(ead) else np.nan,
                "salary_delay_flag":     salary_flag,
                "utilization_high_flag": util_flag,
            })

            progress.progress((idx + 1) / len(customers),
                              text=f"Scored {idx+1}/{len(customers)} customers…")

        progress.empty()

        results_df = pd.DataFrame(results)
        st.session_state["batch_results"] = results_df

        st.subheader("📊 Batch Results")

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers",   len(results_df))
        m2.metric("Avg PD",            f"{results_df['PD (%)'].mean():.1f}%")
        m3.metric("High Risk (≥40%)",  int((results_df["PD (%)"] >= 40).sum()))
        m4.metric("Total Exp. Loss",   f"₹{results_df['Expected Loss (₹)'].sum():,.0f}")

        st.divider()
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Risk bucket distribution
        bucket_counts = results_df["Risk Bucket"].value_counts().reset_index()
        bucket_counts.columns = ["Risk Bucket", "Count"]

        ORDER = ["VERY LOW", "LOW", "MEDIUM", "HIGH", "VERY HIGH"]
        COLORS = {"VERY LOW":"#2ecc71","LOW":"#27ae60","MEDIUM":"#f39c12",
                  "HIGH":"#e67e22","VERY HIGH":"#e74c3c"}

        bucket_counts["Risk Bucket"] = pd.Categorical(bucket_counts["Risk Bucket"],
                                                       categories=ORDER, ordered=True)
        bucket_counts = bucket_counts.sort_values("Risk Bucket")

        col_a, col_b = st.columns(2)

        with col_a:
            fig_bar = px.bar(bucket_counts, x="Risk Bucket", y="Count",
                             color="Risk Bucket",
                             color_discrete_map=COLORS,
                             title="Risk Bucket Distribution")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            fig_hist = px.histogram(results_df, x="PD (%)", nbins=30,
                                    title="PD Distribution Across Portfolio",
                                    color_discrete_sequence=["#636EFA"])
            st.plotly_chart(fig_hist, use_container_width=True)

        # Download
        st.divider()
        csv_out = results_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results CSV", csv_out,
                           "batch_predictions.csv", "text/csv",
                           use_container_width=True)