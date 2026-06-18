import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os

from feature_engineering import engineer_features
from predictor import prepare_inputs

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🧠",
    layout="wide"
)

if "models" not in st.session_state:
    st.error("⚠️ Models not loaded. Please launch via **app.py**.")
    st.stop()

models = st.session_state["models"]

st.title("🧠 SHAP Explainability")
st.markdown("""
This page visualises **SHAP (SHapley Additive exPlanations)** values for the tree ensemble
(XGBoost + LightGBM + CatBoost) to explain which features drive each customer's risk score.
""")

st.divider()

# ── Source selection ──────────────────────────────────────────────────────────
source = st.radio("📥 Data Source", ["Use Batch Results", "Upload New CSV"], horizontal=True)

df_raw = None

if source == "Use Batch Results":
    if "batch_results" not in st.session_state:
        st.warning("⚠️ No batch results found. Please run a batch prediction first, "
                   "or switch to 'Upload New CSV'.")
        st.stop()

    # We need the original raw data, not just results
    st.info("ℹ️ Upload the **same raw CSV** you used for batch prediction so we can "
            "re-engineer features for SHAP analysis.")
    uploaded = st.file_uploader("📤 Re-upload raw customer CSV", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
else:
    uploaded = st.file_uploader("📤 Upload raw customer CSV", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)

# ── Also allow pre-computed SHAP plot images ──────────────────────────────────
st.divider()
st.subheader("🖼️ Pre-computed SHAP Plots")

SHAP_DIR = "shap_plots"
if os.path.isdir(SHAP_DIR):
    plot_files = [f for f in os.listdir(SHAP_DIR)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if plot_files:
        selected_plot = st.selectbox("Select a SHAP plot", sorted(plot_files))
        img_path = os.path.join(SHAP_DIR, selected_plot)
        st.image(img_path, use_container_width=True)
    else:
        st.info("No pre-computed SHAP plots found in `shap_plots/`.")
else:
    st.info("Create a `shap_plots/` folder and add saved SHAP images to display them here.")

# ── Live SHAP from uploaded CSV ───────────────────────────────────────────────
if df_raw is not None:
    st.divider()
    st.subheader("⚡ Live SHAP Analysis")

    customers = df_raw["customer_id"].unique()
    selected_customer = st.selectbox("Select Customer for SHAP Waterfall", customers)

    n_background = st.slider("Background sample size (affects speed)", 20, 200, 50, 10)

    if st.button("🔍 Compute SHAP Values", type="primary", use_container_width=True):
        with st.spinner("Engineering features and computing SHAP values…"):
            # ── Build full feature matrix (last row per customer) ──────────────
            all_last_rows = []
            for cid in customers:
                df_cust = df_raw[df_raw["customer_id"] == cid].copy()
                df_eng  = engineer_features(df_cust)
                tree_cols = models["tree_feature_cols"]
                last_row  = df_eng.reindex(columns=tree_cols, fill_value=0).iloc[[-1]]
                last_row.index = [cid]
                all_last_rows.append(last_row)

            X_all = pd.concat(all_last_rows)

            # ── XGBoost SHAP (TreeExplainer – fastest) ─────────────────────────
            explainer = shap.TreeExplainer(
                models["xgb"],
                data=X_all.sample(min(n_background, len(X_all)), random_state=42),
                feature_perturbation="interventional",
            )

            shap_values = explainer(X_all)

        # ── Summary plot ───────────────────────────────────────────────────────
        st.subheader("📊 SHAP Summary Plot (all customers, XGBoost)")
        fig_summary, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_all, show=False, plot_size=None)
        st.pyplot(fig_summary, use_container_width=True)
        plt.close()

        # ── Waterfall for selected customer ────────────────────────────────────
        if selected_customer in X_all.index:
            cust_idx = list(X_all.index).index(selected_customer)

            st.subheader(f"🌊 SHAP Waterfall — {selected_customer}")
            fig_wf, ax2 = plt.subplots(figsize=(10, 7))
            shap.waterfall_plot(shap_values[cust_idx], max_display=15, show=False)
            st.pyplot(fig_wf, use_container_width=True)
            plt.close()

        # ── Top features table ─────────────────────────────────────────────────
        st.subheader("📋 Mean |SHAP| Feature Importance")
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        shap_importance = pd.DataFrame({
            "Feature":         X_all.columns,
            "Mean |SHAP|":     mean_abs,
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

        st.dataframe(shap_importance.head(20), use_container_width=True, hide_index=True)