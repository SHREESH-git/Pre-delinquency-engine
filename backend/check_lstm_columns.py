
import joblib
import pandas as pd
import sys

# Path to the pickle file
pkl_path = r"d:\Games\Frontend\backend\models\lstm_feature_columns.pkl"

try:
    cols = joblib.load(pkl_path)
    print(f"Loaded {len(cols)} LSTM columns.")
    print("First 10 columns:")
    for c in cols[:10]:
        print(f" - {c}")
        
except Exception as e:
    print(f"Error: {e}")
