
import joblib
import sys

# Path to the pickle file
pkl_path = r"d:\Games\Frontend\backend\models\tree_feature_columns.pkl"

try:
    cols = joblib.load(pkl_path)
    print("All Feature Columns:")
    for c in cols:
        print(c)
        
except Exception as e:
    print(f"Error: {e}")
