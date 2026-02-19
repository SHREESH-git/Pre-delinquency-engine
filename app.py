from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd

from PD_Model.model_loader import load_models
from PD_Model.feature_engineering import engineer_features
from PD_Model.predictor import prepare_inputs, predict_pd
from PD_Model.risk_engine import expected_loss, risk_bucket


# INIT APP
 
app = FastAPI(
    title="Pre-Delinquency Intervention Engine",
    version="1.0"
)

# Load models once at startup
models = load_models("PD_Model/content/models")



# INPUT SCHEMA

class MonthlyRecord(BaseModel):
    customer_id: str
    month: str
    customer_segment: str
    region_tier: str
    product_type: str
    active_products_count: int
    credit_card_utilization: float
    total_monthly_obligation: float
    emi_amount: float
    days_to_emi: int
    emi_to_income_ratio: float
    salary_delay_days: int
    weekly_balance_change_pct: float
    atm_withdrawal_amount: float
    monthly_income: float


class CustomerHistory(BaseModel):
    records: List[MonthlyRecord]



# PREDICTION ENDPOINT

@app.post("/predict")
def predict(customer_history: CustomerHistory):

    # Convert input list → DataFrame
    df_raw = pd.DataFrame([r.dict() for r in customer_history.records])

    # 1️⃣ Feature Engineering
    df_engineered = engineer_features(df_raw)

    # 2️⃣ One-hot encoding
    df_engineered = pd.get_dummies(df_engineered)

    # 3️⃣ Prepare model inputs
    tree_input, lstm_input = prepare_inputs(models, df_engineered)

    # 4️⃣ Predict PD
    pd_score = predict_pd(models, tree_input, lstm_input)

    # 5️⃣ Expected Loss Calculation
    last_row = df_engineered.iloc[-1]

    el, lgd, ead = expected_loss(
        models,
        pd_score,
        last_row["emi_amount"],
        last_row["credit_card_utilization"],
        last_row["monthly_income"],
        int(last_row["salary_delay_days"] > 5),
        int(last_row["credit_card_utilization"] > 0.75),
    )

    # 6️⃣ Return response
    return {
        "probability_of_default": round(pd_score, 4),
        "risk_bucket": risk_bucket(pd_score),
        "expected_loss": round(el, 2),
        "lgd": round(lgd, 3),
        "ead": round(ead, 2)
    }