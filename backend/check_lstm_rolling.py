
import joblib
import pandas as pd
import sys

# Path to the pickle file
pkl_path = r"d:\Games\Frontend\backend\models\lstm_feature_columns.pkl"

try:
    cols = joblib.load(pkl_path)
    print(f"Loaded {len(cols)} LSTM columns.")
    
    rolling_cols = [c for c in cols if "_mean_" in c or "_std_" in c or "_max_" in c or "_min_" in c]
    print(f"Rolling columns in LSTM: {len(rolling_cols)}")
    for c in rolling_cols:
        print(f" - {c}")
        
except Exception as e:
    print(f"Error: {e}")
