
import pandas as pd
import numpy as np
import sys
import os
import joblib

# Add backend to path
sys.path.append("d:/Games/Frontend/backend")

from main import AnalyticsService, DATASET_PATH
from ml_core.feature_engineering import engineer_features
from ml_core.predictor import prepare_inputs, predict_pd
from ml_core.model_loader import load_models

def verify_fix():
    # Redirect stdout to a file
    with open("verify_log.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        
        print(f"Loading data from {DATASET_PATH}")
        service = AnalyticsService(DATASET_PATH)
        service.load_data()
        
        cid = "CUST0000001"
        print(f"\nAnalyzing Customer: {cid}")
        
        hist_df = service.get_customer_history(cid)
        if hist_df.empty:
            print("Customer not found!")
            return
            
        print(f"History Length: {len(hist_df)} months")
        
        # Run Feature Engineering
        df_eng = engineer_features(hist_df)
        
        # Check a rolling feature for the last row
        # credit_card_utilization is a good candidate
        last_row = df_eng.iloc[-1]
        
        print("\n--- Feature Verification ---")
        print(f"Month: {last_row['month']}")
        
        
        # Manually calculate 5-month mean for verification (Window changed from 6 to 5)
        # Note: 'rolling(5)' includes current month + 4 previous.
        util_history = hist_df['credit_card_utilization'].tail(5)
        manual_mean = util_history.mean()
        manual_max = util_history.max()
        
        feat_mean = last_row['credit_card_utilization_mean_3m'] # Name is still _3m
        feat_max = last_row['credit_card_utilization_max_3m']
        
        print(f"Manual 5m Mean: {manual_mean:.4f}")
        print(f"Feature Mean (stored as _3m): {feat_mean:.4f}")
        
        if abs(manual_mean - feat_mean) < 0.001:
            print("✅ SUCCESS: Feature matches 5-month rolling mean.")
        else:
            print("❌ FAILURE: Feature does not match 5-month rolling mean.")
            
        print(f"LSTM Sequence Shape Check: We expect (... , 3, 15)")
            
        
        # Run Prediction
        print("\n--- Prediction Verification ---")
        try:
            models = load_models("d:/Games/Frontend/backend/models")
            
            cat_cols = df_eng.select_dtypes(include=["object"]).columns.tolist()
            df_prep = pd.get_dummies(df_eng, columns=cat_cols, drop_first=True)
            
            tree_input, lstm_input = prepare_inputs(models, df_prep)
            pd_score = predict_pd(models, tree_input, lstm_input)
            
            final_score = 0
            if isinstance(pd_score, (np.ndarray, list)):
                final_score = float(np.mean(pd_score))
            else:
                final_score = float(pd_score)
                
            print(f"New Prediction PD: {final_score:.4f}")
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            
    # Restore stdout
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    verify_fix()
