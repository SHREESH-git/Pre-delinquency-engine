import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from feature_engineering import engineer_features
from predictor import prepare_inputs, predict_pd
from risk_engine import expected_loss, risk_bucket

st.set_page_config(
    page_title="Single Customer Prediction",
    page_icon="🔍",
    layout="wide"
)

# ── Guard: models must be loaded from app.py ──────────────────────────────────
if "models" not in st.session_state:
    st.error("⚠️ Models not loaded. Please launch the app via **app.py** (run `streamlit run app.py`).")
    st.stop()

models = st.session_state["models"]

st.title("🔍 Single Customer Risk Prediction")
st.markdown("Enter up to **3 months** of customer behavioural data. "
            "The engine will run feature engineering and predict next-month delinquency probability.")

st.divider()

# ── How many months of history? ───────────────────────────────────────────────
n_months = st.selectbox("📅 How many months of history are you entering?", [1, 2, 3, 4, 5], index=0)

NUMERIC_INPUTS = [
    ("active_products_count",      "Active Products Count",          0,   10,  2),
    ("credit_card_utilization",    "Credit Card Utilization (0–1)",  0.0, 1.0, 0.45),
    ("total_monthly_obligation",   "Total Monthly Obligation (₹)",   0,   200000, 15000),
    ("emi_amount",                 "EMI Amount (₹)",                 0,   200000, 8000),
    ("days_to_emi",                "Days to Next EMI",               0,   31,  10),
    ("emi_to_income_ratio",        "EMI-to-Income Ratio (0–1)",      0.0, 1.0, 0.30),
    ("salary_delay_days",          "Salary Delay Days",              0,   30,  2),
    ("weekly_balance_change_pct",  "Weekly Balance Change %",        -1.0,1.0, 0.02),
    ("atm_withdrawal_amount",      "ATM Withdrawal Amount (₹)",      0,   50000, 3000),
    ("monthly_income",             "Monthly Income (₹)",             5000,500000, 35000),
]

# ── Collect monthly rows ──────────────────────────────────────────────────────
rows = []
customer_id = st.text_input("Customer ID", value="CUST_001")

for m in range(1, n_months + 1):
    with st.expander(f"📆 Month {m} Data", expanded=(m == n_months)):
        cols = st.columns(2)
        row = {"customer_id": customer_id, "month": m}
        for i, (col_name, label, min_v, max_v, default) in enumerate(NUMERIC_INPUTS):
            with cols[i % 2]:
                if isinstance(default, float):
                    row[col_name] = st.number_input(
                        label, min_value=float(min_v), max_value=float(max_v),
                        value=float(default), step=0.01,
                        key=f"{col_name}_m{m}"
                    )
                else:
                    row[col_name] = st.number_input(
                        label, min_value=int(min_v), max_value=int(max_v),
                        value=int(default), step=1,
                        key=f"{col_name}_m{m}"
                    )
        rows.append(row)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🚀 Predict Delinquency Risk", type="primary", use_container_width=True):
    with st.spinner("Running feature engineering and hybrid model..."):
        df_raw = pd.DataFrame(rows)
        df_eng = engineer_features(df_raw)

        tree_input, lstm_input = prepare_inputs(models, df_eng)
        pd_score = predict_pd(models, tree_input, lstm_input)
        bucket    = risk_bucket(pd_score)

        last = df_raw.iloc[-1]
        salary_flag = int(last["salary_delay_days"] > 5)
        util_flag   = int(last["credit_card_utilization"] > 0.75)

        el, lgd, ead = expected_loss(
            models, pd_score,
            last["emi_amount"],
            last["credit_card_utilization"],
            last["monthly_income"],
            salary_flag,
            util_flag
        )

    # ── Results ───────────────────────────────────────────────────────────────
    st.subheader("📊 Prediction Results")

    BUCKET_COLORS = {
        "VERY LOW": "#2ecc71",
        "LOW":      "#27ae60",
        "MEDIUM":   "#f39c12",
        "HIGH":     "#e67e22",
        "VERY HIGH":"#e74c3c",
    }
    color = BUCKET_COLORS.get(bucket, "#95a5a6")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("🎯 Probability of Default", f"{pd_score:.2%}")
    r2.metric("🪣 Risk Bucket", bucket)
    r3.metric("💸 Expected Loss (₹)", f"₹{el:,.0f}")
    r4.metric("📐 LGD", f"{lgd:.2%}")

    st.markdown(
        f"""<div style="background:{color}22;border-left:5px solid {color};
            padding:12px 18px;border-radius:6px;margin-top:8px">
            <b style="color:{color};font-size:1.1rem">Risk Level: {bucket}</b>
            &nbsp;—&nbsp; PD = <b>{pd_score:.2%}</b>
        </div>""",
        unsafe_allow_html=True
    )

    st.divider()

    # ── Gauge chart ───────────────────────────────────────────────────────────
    st.subheader("🔵 PD Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pd_score * 100,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis":  {"range": [0, 100]},
            "bar":   {"color": color},
            "steps": [
                {"range": [0, 5],   "color": "#2ecc71"},
                {"range": [5, 15],  "color": "#27ae60"},
                {"range": [15, 35], "color": "#f1c40f"},
{"range": [35, 60], "color": "#e67e22"},
                {"range": [60, 100],"color": "#e74c3c"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "value": 35},
        },
        title={"text": "Probability of Default (%)"}
    ))
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Engineered features table ─────────────────────────────────────────────
    with st.expander("🔬 Engineered Features (last row)"):
        st.dataframe(df_eng.iloc[[-1]].T.rename(columns={df_eng.index[-1]: "Value"}),
                     use_container_width=True)