
import pandas as pd
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from backend.main import AnalyticsService, DATASET_PATH
from backend.ml_core.feature_engineering import engineer_features

def check_history():
    service = AnalyticsService(DATASET_PATH)
    service.load_data()
    
    cust_id = "CUST0000000"
    hist = service.get_customer_history(cust_id)
    
    print(f"Customer: {cust_id}")
    print(f"History Length: {len(hist)}")
    print("Months present:", hist['month'].tolist())
    
    if len(hist) > 0:
        df_eng = engineer_features(hist)
        print("\nEngineered Data (Last 3 rows):")
        cols = ['month', 'credit_card_utilization', 'credit_card_utilization_mean_3m', 'credit_card_utilization_lag1']
        valid_cols = [c for c in cols if c in df_eng.columns]
        print(df_eng[valid_cols].tail(3))
        
        # Check if lag works
        print("\nLag Check:")
        print(df_eng[['month', 'credit_card_utilization', 'credit_card_utilization_lag1']].head(6))

if __name__ == "__main__":
    check_history()
