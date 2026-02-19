
import sys
import os
import joblib

# Add backend to path
sys.path.append("d:/Games/Frontend/backend")

from ml_core.model_loader import load_models

def check_features():
    try:
        models = load_models("d:/Games/Frontend/backend/models")
        tree_cols = models.get("tree_feature_cols", [])
        lstm_cols = models.get("lstm_feature_cols", [])
        
        print(f"Total Tree Features: {len(tree_cols)}")
        print("Tree Features looking for '6m':")
        for col in tree_cols:
            if "6m" in col:
                print(f" - {col}")
                
        print(f"\nTotal LSTM Features: {len(lstm_cols)}")
        
    except Exception as e:
        print(f"Error loading models: {e}")

if __name__ == "__main__":
    check_features()
