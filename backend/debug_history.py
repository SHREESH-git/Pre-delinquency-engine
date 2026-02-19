
import pandas as pd
import sys
import os

# Assume running from backend dir
sys.path.append(os.getcwd())

from main import AnalyticsService, DATASET_PATH
from ml_core.feature_engineering import engineer_features

def check_history():
    print(f"Loading data from {DATASET_PATH}")
    service = AnalyticsService(DATASET_PATH)
    service.load_data()
    
    cust_id = "CUST0000000"
    hist = service.get_customer_history(cust_id)
    
    print(f"Customer: {cust_id}")
    if hist.empty:
        print("History is empty!")
        return

    print(f"History Length: {len(hist)}")
    print("Months present:", hist['month'].tolist())
    
    try:
        df_eng = engineer_features(hist)
        print("\nEngineered Data (Columns):", df_eng.columns.tolist())
        
        cols = ['month', 'credit_card_utilization', 'credit_card_utilization_mean_3m', 'credit_card_utilization_lag1']
        valid_cols = [c for c in cols if c in df_eng.columns]
        
        print("\nLast 6 rows of key features:")
        print(df_eng[valid_cols].tail(6))
        
    except Exception as e:
        print("Error in FE:", e)

    import joblib
    try:
        lstm_cols = joblib.load("models/lstm_feature_columns.pkl")
        print("\nLSTM Feature Columns:", lstm_cols)
        
        missing = [c for c in lstm_cols if c not in df_eng.columns]
        if missing:
             print("MISSING IN DATAFRAME:", missing)
        else:
             print("All LSTM columns present in DF.")
             
    except Exception as e:
        print("Error loading pickle:", e)
