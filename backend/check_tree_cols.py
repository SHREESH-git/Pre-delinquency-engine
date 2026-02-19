
import joblib
import sys
import os

try:
    cols = joblib.load("models/tree_feature_columns.pkl")
    print("Tree Feature Columns:")
    for c in cols:
        print(c)
except Exception as e:
    print("Error:", e)
