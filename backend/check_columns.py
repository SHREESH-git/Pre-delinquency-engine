
import joblib
import pandas as pd
import sys

# Path to the pickle file
pkl_path = r"d:\Games\Frontend\backend\models\tree_feature_columns.pkl"

try:
    cols = joblib.load(pkl_path)
    print(f"Loaded {len(cols)} columns.")
    
    rolling_6m = [c for c in cols if "6m" in c]
    print(f"Columns with '6m': {len(rolling_6m)}")
    for c in rolling_6m:
        print(f" - {c}")
        
except Exception as e:
    print(f"Error: {e}")
