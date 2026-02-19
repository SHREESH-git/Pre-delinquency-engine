
import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

try:
    import pandas as pd
    import numpy as np
    import joblib
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Add backend to path
sys.path.append(os.getcwd())

from ml_core.model_loader import load_models
from ml_core.feature_engineering import engineer_features
from ml_core.predictor import prepare_inputs, predict_pd

def simulate_history():
    csv_path = r"d:\Games\Frontend\content_backup (2)\content_backup\content\financial_stress_full_bank_grade_dataset.csv"
    models_path = r"d:\Games\Frontend\backend\models"
    
    print(f"Loading models from {models_path}...")
    try:
        models = load_models(models_path)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Model load failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Loading data...")
    df_raw = pd.read_csv(csv_path)
    # Ensure correct types
    numeric_cols = ["active_products_count", "credit_card_utilization", "total_monthly_obligation", 
                    "emi_amount", "days_to_emi", "emi_to_income_ratio", "salary_delay_days", 
                    "weekly_balance_change_pct", "atm_withdrawal_amount", "monthly_income"]
    
    for col in numeric_cols:
         df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
         
    cust_df = df_raw[df_raw['customer_id'] == 'CUST0000001'].sort_values('month').reset_index(drop=True)
    
    print("\n--- Simulating PD Over Time ---")
    
    for i in range(2, len(cust_df)):
        current_slice = cust_df.iloc[:i+1].copy()
        month_label = current_slice.iloc[-1]['month']
        
        # ENGINEER FEATURES
        df_eng = engineer_features(current_slice)
        
        # Taking the last row as the "Current Status"
        last_row = df_eng.iloc[[-1]] 
        
        # Prepare inputs
        try:
             tree_in_full, lstm_in = prepare_inputs(models, df_eng)
             tree_in_last = tree_in_full.iloc[[-1]]
             
             pd_val, risk_bucket_val, el, lgd = predict_pd(models, tree_in_last, lstm_in)
             
             print(f"Month: {month_label} | History Len: {len(current_slice)} | Prediction PD: {pd_val:.4f} | Risk: {risk_bucket_val}")
             
        except Exception as e:
            print(f"Prediction failed for {month_label}: {e}")

if __name__ == "__main__":
    simulate_history()
