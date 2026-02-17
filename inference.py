import pandas as pd
from model_loader import load_models
from feature_engineering import engineer_features
from predictor import prepare_inputs, predict_pd
from risk_engine import expected_loss, risk_bucket

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
models = load_models("content/models")

# ---------------------------------------------------
# LOAD RAW DATA (customer monthly history)
# ---------------------------------------------------
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

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------
df_engineered = engineer_features(df_raw)

# ---------------------------------------------------
# ONE HOT ENCODE (TREE NEEDS IT)
# ---------------------------------------------------
cat_cols = df_engineered.select_dtypes(include=["object"]).columns.tolist()

df_engineered = pd.get_dummies(
    df_engineered,
    columns=cat_cols,
    drop_first=True
)

# ---------------------------------------------------
# PREPARE INPUTS
# ---------------------------------------------------
tree_input, lstm_input = prepare_inputs(models, df_engineered)

# ---------------------------------------------------
# PREDICT PD
# ---------------------------------------------------
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