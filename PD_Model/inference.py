import pandas as pd
from model_loader import load_models
from feature_engineering import engineer_features
from predictor import prepare_inputs, predict_pd
from risk_engine import expected_loss, risk_bucket


# LOAD MODELS

models = load_models("content/models")


# LOAD RAW DATA (customer monthly history)

# df_raw = pd.read_csv("new_customer_data.csv")

df_raw = pd.DataFrame([
    {
        "customer_id": "CUST_TEST_001",
        "month": "2024-04",
        "customer_segment": "salaried",
        "region_tier": "tier_1",
        "product_type": "personal_loan",
        "active_products_count": 2,
        "credit_card_utilization": 0.4,
        "total_monthly_obligation": 25000,
        "emi_amount": 18000,
        "days_to_emi": 15,
        "emi_to_income_ratio": 0.20,
        "salary_delay_days": 3,
        "weekly_balance_change_pct": -5,
        "atm_withdrawal_amount": 15000,
        "monthly_income": 60000
    },
    {
        "customer_id": "CUST_TEST_001",
        "month": "2024-05",
        "customer_segment": "salaried",
        "region_tier": "tier_1",
        "product_type": "personal_loan",
        "active_products_count": 2,
        "credit_card_utilization": 0.41,
        "total_monthly_obligation": 25050,
        "emi_amount": 18000,
        "days_to_emi": 15,
        "emi_to_income_ratio": 0.20,
        "salary_delay_days": 0,
        "weekly_balance_change_pct": -1,
        "atm_withdrawal_amount": 10000,
        "monthly_income": 60000
    }
])


# FEATURE ENGINEERING

df_engineered = engineer_features(df_raw)


# ONE HOT ENCODE (TREE NEEDS IT)

cat_cols = df_engineered.select_dtypes(include=["object"]).columns.tolist()

df_engineered = pd.get_dummies(
    df_engineered,
    columns=cat_cols,
    drop_first=True
)


# PREPARE INPUTS

tree_input, lstm_input = prepare_inputs(models, df_engineered)


# PREDICT PD

pd_score = predict_pd(models, tree_input, lstm_input)

print("Predicted PD:", pd_score)


# expected loss
from risk_engine import expected_loss, risk_bucket

last_row = df_engineered.iloc[-1]

emi = last_row["emi_amount"]
utilization = last_row["credit_card_utilization"]
income = last_row["monthly_income"]

salary_flag = int(last_row["salary_delay_days"] > 5)
util_flag = int(last_row["credit_card_utilization"] > 0.75)

el, lgd, ead = expected_loss(
    models,
    pd_score,
    emi,
    utilization,
    income,
    salary_flag,
    util_flag
)

bucket = risk_bucket(pd_score)

print("Risk Bucket:", bucket)
print("Expected Loss: ₹", round(el,2))


# SHAP FOR FINAL ENSEMBLE (TREE + LSTM)


import shap
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd


# PREPARE TREE BACKGROUND (use all available rows, not just last row) 
# Rebuild a full tree-input frame from the engineered dataframe so SHAP background
# sampling is representative. This avoids sampling from a single-row `tree_input`.
tree_cols = models["tree_feature_cols"]
tree_input_all = df_engineered.reindex(columns=tree_cols, fill_value=0)
tree_input = tree_input_all.iloc[[-1]]  # keep the single-row last record for printing/prediction


#  LSTM BASE INPUT (keep fixed for explanations) 
# prepare_inputs already returned a lstm input for the full history earlier — reuse it.
# If not present, compute it now to be safe.
if 'lstm_input_base' not in globals():
    # compute LSTM input from full engineered history if it's not already in scope
    _, lstm_input_base = prepare_inputs(models, df_engineered)


#  ENSEMBLE WRAPPER (robust, vectorized) 
def ensemble_predict(X):
    """
    Predict final calibrated PD for an array-like or DataFrame of tree features.

    This matches the logic in `predict_pd` (tree ensemble weights, LSTM blend,
    calibration, and floor/cap). For SHAP we keep the LSTM input fixed for the
    customer being explained (we replicate the same LSTM sequence for each
    evaluated tree feature perturbation).
    """

    # Accept numpy arrays or DataFrames
    if isinstance(X, np.ndarray):
        df_X = pd.DataFrame(X, columns=tree_cols)
    else:
        df_X = pd.DataFrame(X, columns=getattr(X, "columns", tree_cols))

    # Ensure columns align with model's expected tree features
    df_X = df_X.reindex(columns=tree_cols, fill_value=0)

    n = len(df_X)
    if n == 0:
        return np.array([])

    # -------- TREE PD (vectorized) --------
    xgb = models["xgb"]
    lgb = models["lgb"]
    cat = models["cat"]

    px = xgb.predict_proba(df_X)[:, 1]
    pl = lgb.predict_proba(df_X)[:, 1]
    pc = cat.predict_proba(df_X)[:, 1]

    wx = models.get("wx", 1.0)
    wl = models.get("wl", 1.0)
    wc = models.get("wc", 1.0)
    s = wx + wl + wc
    tree_pd = (wx * px + wl * pl + wc * pc) / s

    # -------- LSTM PD (replicate base sequence for each perturbed row) --------
    lstm_model = models["lstm"]
    # lstm_input_base shape: (1, seq_len, feat)
    lstm_batch = np.repeat(lstm_input_base, n, axis=0)
    device = next(lstm_model.parameters()).device
    tensor_input = torch.tensor(lstm_batch, dtype=torch.float32).to(device)
    with torch.no_grad():
        lstm_pd = lstm_model(tensor_input).cpu().numpy().reshape(-1)

    # -------- HYBRID BLEND & CALIBRATION (same as predict_pd) --------
    best_weight = models.get("best_weight", 0.6)
    blended = best_weight * tree_pd + (1 - best_weight) * lstm_pd

    calibrator = models["calibrator"]
    calibrated = calibrator.predict_proba(blended.reshape(-1, 1))[:, 1]

    # Apply floor / cap if present
    config = models.get("config", {})
    pd_floor = config.get("calibration", {}).get("pd_floor") if isinstance(config.get("calibration"), dict) else config.get("pd_floor")
    pd_cap = config.get("calibration", {}).get("pd_cap") if isinstance(config.get("calibration"), dict) else config.get("pd_cap")

    if pd_floor is not None or pd_cap is not None:
        calibrated = np.clip(calibrated, pd_floor if pd_floor is not None else -np.inf,
                             pd_cap if pd_cap is not None else np.inf)

    return calibrated


#  BACKGROUND DATA (representative) 
background = tree_input_all.sample(n=min(50, len(tree_input_all)), random_state=42)


#  CREATE EXPLAINER 
explainer = shap.KernelExplainer(ensemble_predict, background)


#  EXPLAIN LAST CUSTOMER 
X_test = tree_input  # already a single-row DataFrame aligned to tree_cols

shap_values = explainer.shap_values(X_test)

# shap returns array-like; normalize to a 2D numpy array of shape (n_samples, n_features)
if isinstance(shap_values, list):
    shap_arr = np.array(shap_values[0])
else:
    shap_arr = np.array(shap_values)

shap_sample = shap_arr[0] if shap_arr.ndim == 2 else shap_arr

#  BUILD IMPORTANCE TABLE (used by both plots) 
importance = pd.DataFrame({
    "Feature": tree_cols,
    "SHAP_Value": shap_sample
})
importance["Impact"] = importance["SHAP_Value"].abs()
importance = importance.sort_values("Impact", ascending=False)


#  MODERN WATERFALL (RESPONSIVE LAYOUT) 

# Build Explanation object
exp = shap.Explanation(
    values=shap_sample,
    base_values=explainer.expected_value,
    data=X_test.iloc[0].values,
    feature_names=tree_cols
)

# Create figure and plot waterfall - use fewer features to avoid overlap
fig, ax = plt.subplots(figsize=(14, 12), dpi=100)

# Use SHAP's waterfall plot with reduced max_display to avoid crowding
shap.plots.waterfall(exp, max_display=8, show=False)

# Get the current figure (SHAP creates its own, so use gcf)
fig = plt.gcf()
fig.set_size_inches(14, 12)

plt.title("Final PD Explanation (Tree + LSTM Ensemble)", fontsize=14, pad=20)

# Get figure and axis references after SHAP has drawn
if fig.axes:
    ax = fig.axes[0]
    
    # Move y-axis tick labels to the right side so long feature names don't overlap
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", which="both", labelleft=False, labelright=True, labelsize=10)
    for label in ax.get_yticklabels():
        label.set_ha('right')
    
    # Increase spacing between y-ticks
    ax.tick_params(axis='y', pad=10)

    # Increase left margin so long feature names are fully visible
    max_label_len = max((len(str(f)) for f in tree_cols), default=20)
    left_margin = min(0.75, max(0.15, 0.15 + max_label_len * 0.012))
    fig.subplots_adjust(left=left_margin, right=0.95, top=0.92, bottom=0.15)

# Annotate base value and predicted PD on the waterfall (improves readability)
try:
    base_val = float(np.atleast_1d(explainer.expected_value).ravel()[0])
except Exception as e:
    base_val = None

pred_text = ""
try:
    if pd_score is not None:
        pred_text = f"Predicted PD: {float(pd_score):.6f}"
    else:
        pred_text = "Predicted PD: N/A"
except Exception:
    pred_text = "Predicted PD: N/A"

if base_val is not None:
    ann = f"Base PD: {base_val:.6f}"
    if pred_text:
        ann = ann + "\n" + pred_text
    # place annotation at top-right corner with better visibility
    fig.text(0.98, 0.98, ann, ha='right', va='top', fontsize=10, fontweight='bold',
             bbox=dict(facecolor='lightyellow', alpha=0.95, edgecolor='black', boxstyle='round,pad=0.8'))

# Ensure all elements render before saving
plt.tight_layout()

# Save a high-res copy of the waterfall for reports
try:
    fig.savefig('shap_waterfall.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print("✓ Saved: shap_waterfall.png (waterfall plot with feature contributions)")
except Exception as e:
    print(f"Error saving waterfall: {e}")

plt.show()


#  SECOND PLOT: HORIZONTAL BAR CHART WITH NUMERIC VALUES 
# Build an easily readable numeric plot of top SHAP contributions (signed values)
top_n = 12
importance_full = importance.copy()
top_features = importance_full.head(top_n).copy()
# Keep original sign from shap_sample where available
top_features = top_features.set_index('Feature')

# order for horizontal bar plot: from smallest to largest (so bars go left->right)
top_plot = top_features.sort_values('SHAP_Value')

# Dynamically size figure based on number of features
fig_height = max(6, 0.5 * len(top_plot))
fig2, ax2 = plt.subplots(figsize=(14, fig_height), dpi=150)

colors = [('#2ca02c' if v < 0 else '#d62728') for v in top_plot['SHAP_Value']]
bars = ax2.barh(range(len(top_plot)), top_plot['SHAP_Value'].values, color=colors, height=0.6)
ax2.set_yticks(range(len(top_plot)))
ax2.set_yticklabels(top_plot.index, fontsize=10)
ax2.set_xlabel('SHAP Value (Impact on PD)', fontsize=11, fontweight='bold')
ax2.set_title('Top SHAP Feature Contributions (with numeric values)', fontsize=13, fontweight='bold', pad=15)

# Add gridlines for easier reading
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)

# Add numeric labels clearly at the end of each bar with background
for idx, (feature, row) in enumerate(top_plot.iterrows()):
    val = row['SHAP_Value']
    # Compute x position: end of bar + small offset
    x_max = top_plot['SHAP_Value'].max()
    x_min = top_plot['SHAP_Value'].min()
    x_range = x_max - x_min if x_max != x_min else 1
    
    if val >= 0:
        x_pos = val + 0.01 * x_range
        ha = 'left'
    else:
        x_pos = val - 0.01 * x_range
        ha = 'right'
    
    # Draw text with background box for visibility
    ax2.text(x_pos, idx, f"{val:.6f}", va='center', ha=ha, fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

# Save numeric bar chart
fig2.savefig('shap_top_features.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: shap_top_features.png (numeric SHAP values)")
plt.show()

print("\nTop 5 Risk Drivers:")
print(importance.head(5))