import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Portfolio Risk Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Portfolio Risk Analytics")
st.markdown("""
Visualise **PD distributions**, **risk bucket breakdowns**, and **expected-loss trends**
across your portfolio. Run a **Batch Prediction** first to populate live data,
or browse the simulated demo below.
""")

st.divider()

# ── Source ────────────────────────────────────────────────────────────────────
BUCKET_ORDER  = ["VERY LOW", "LOW", "MEDIUM", "HIGH", "VERY HIGH"]
BUCKET_COLORS = {
    "VERY LOW":  "#2ecc71",
    "LOW":       "#27ae60",
    "MEDIUM":    "#f39c12",
    "HIGH":      "#e67e22",
    "VERY HIGH": "#e74c3c",
}

use_batch = "batch_results" in st.session_state

if use_batch:
    df = st.session_state["batch_results"].copy()
    st.success(f"✅ Using live batch results — {len(df):,} customers scored.")
else:
    st.info("ℹ️ No batch results found. Showing **simulated demo data**. "
            "Run **Batch Prediction** for live analytics.")

    rng = np.random.default_rng(42)
    n   = 500
    pd_vals = np.clip(rng.beta(2, 10, n) * 100, 0.5, 95)

    def _bucket(p):
        if p >= 60:  return "VERY HIGH"
        if p >= 40:  return "HIGH"
        if p >= 15:  return "MEDIUM"
        if p >= 5:   return "LOW"
        return "VERY LOW"

    df = pd.DataFrame({
        "customer_id":       [f"CUST_{i:04d}" for i in range(n)],
        "PD (%)":            pd_vals.round(2),
        "Risk Bucket":       [_bucket(p) for p in pd_vals],
        "Expected Loss (₹)": (pd_vals / 100 * rng.uniform(5000, 80000, n)).round(2),
        "LGD":               np.clip(rng.normal(0.45, 0.12, n), 0.1, 0.9).round(4),
        "EAD (₹)":           rng.uniform(5000, 200000, n).round(2),
    })

st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("👥 Customers",       f"{len(df):,}")
k2.metric("📉 Avg PD",          f"{df['PD (%)'].mean():.1f}%")
k3.metric("📊 Median PD",       f"{df['PD (%)'].median():.1f}%")
k4.metric("🚨 High Risk (≥40%)", int((df["PD (%)"] >= 40).sum()))
k5.metric("💸 Total Exp. Loss", f"₹{df['Expected Loss (₹)'].sum():,.0f}")

st.divider()

# ── Row 1: Distribution + Bucket bar ─────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 PD Distribution")
    fig_hist = px.histogram(
        df, x="PD (%)", nbins=40,
        color_discrete_sequence=["#636EFA"],
        title="Probability of Default — Portfolio Histogram"
    )
    fig_hist.add_vline(x=df["PD (%)"].mean(), line_dash="dash",
                       line_color="red", annotation_text="Mean")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_b:
    st.subheader("🪣 Risk Bucket Distribution")
    bucket_counts = (
        df["Risk Bucket"]
        .value_counts()
        .reindex(BUCKET_ORDER, fill_value=0)
        .reset_index()
    )
    bucket_counts.columns = ["Risk Bucket", "Count"]
    bucket_counts["Color"] = bucket_counts["Risk Bucket"].map(BUCKET_COLORS)

    fig_bar = px.bar(
        bucket_counts, x="Risk Bucket", y="Count",
        color="Risk Bucket", color_discrete_map=BUCKET_COLORS,
        title="Customers per Risk Bucket",
        category_orders={"Risk Bucket": BUCKET_ORDER}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Row 2: EL breakdown + Scatter ─────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("💸 Expected Loss by Risk Bucket")
    el_bucket = (
        df.groupby("Risk Bucket")["Expected Loss (₹)"]
        .sum()
        .reindex(BUCKET_ORDER, fill_value=0)
        .reset_index()
    )
    el_bucket.columns = ["Risk Bucket", "Total Expected Loss (₹)"]

    fig_el = px.bar(
        el_bucket, x="Risk Bucket", y="Total Expected Loss (₹)",
        color="Risk Bucket", color_discrete_map=BUCKET_COLORS,
        title="Total Expected Loss per Bucket",
        category_orders={"Risk Bucket": BUCKET_ORDER}
    )
    st.plotly_chart(fig_el, use_container_width=True)

with col_d:
    st.subheader("🔵 PD vs EAD Scatter")
    fig_scatter = px.scatter(
        df, x="EAD (₹)", y="PD (%)",
        color="Risk Bucket", color_discrete_map=BUCKET_COLORS,
        hover_data=["customer_id", "Expected Loss (₹)"],
        title="PD vs Exposure at Default",
        category_orders={"Risk Bucket": BUCKET_ORDER},
        opacity=0.75
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ── Row 3: LGD distribution + Box plot ────────────────────────────────────────
col_e, col_f = st.columns(2)

with col_e:
    st.subheader("📉 LGD Distribution")
    fig_lgd = px.histogram(
        df, x="LGD", nbins=30,
        color_discrete_sequence=["#AB63FA"],
        title="Loss Given Default Distribution"
    )
    st.plotly_chart(fig_lgd, use_container_width=True)

with col_f:
    st.subheader("📦 PD by Risk Bucket (Box Plot)")
    fig_box = px.box(
        df, x="Risk Bucket", y="PD (%)",
        color="Risk Bucket", color_discrete_map=BUCKET_COLORS,
        title="PD Spread Within Each Risk Bucket",
        category_orders={"Risk Bucket": BUCKET_ORDER}
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ── Detailed table with filters ───────────────────────────────────────────────
st.subheader("🗂️ Customer Risk Table")

filter_buckets = st.multiselect(
    "Filter by Risk Bucket", BUCKET_ORDER, default=BUCKET_ORDER
)
df_filtered = df[df["Risk Bucket"].isin(filter_buckets)]
st.dataframe(
    df_filtered.sort_values("PD (%)", ascending=False).reset_index(drop=True),
    use_container_width=True, hide_index=True
)

csv_out = df_filtered.to_csv(index=False).encode()
st.download_button("⬇️ Download Filtered Results", csv_out,
                   "portfolio_risk.csv", "text/csv")