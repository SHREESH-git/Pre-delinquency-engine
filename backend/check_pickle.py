
import joblib
import sys
import os

try:
    cols = joblib.load("models/lstm_feature_columns.pkl")
    print("Columns:", cols)
except Exception as e:
    print("Error:", e)
