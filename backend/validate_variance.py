
import pandas as pd
import numpy as np
import sys
import os
import joblib
import torch

# Add backend to path
sys.path.append(os.getcwd())

from main import AnalyticsService, DATASET_PATH
from ml_core.feature_engineering import engineer_features
from ml_core.predictor import prepare_inputs, predict_pd

def validate_variance():
    print(f"Loading data from {DATASET_PATH}")
    service = AnalyticsService(DATASET_PATH)
    service.load_data()
    
    # Stratified sample of 10 customers
    sample_ids = service.sample_df['customer_id'].head(10).tolist()
    
    # Also pick some specific ones if possible?
    # Let's just take top 10 unique IDs from full DF
    unique_ids = service.full_df['customer_id'].unique()[:10]
    
    # Load models
    from ml_core.model_loader import load_models
    models = load_models("models")
    
    results = []
    
    print(f"\nAnalyzing {len(unique_ids)} customers...")
    
    for cid in unique_ids:
        hist_df = service.get_customer_history(cid)
        if hist_df.empty:
            continue
            
        try:
            df_eng = engineer_features(hist_df)
            
            cat_cols = df_eng.select_dtypes(include=["object"]).columns.tolist()
            df_prep = pd.get_dummies(df_eng, columns=cat_cols, drop_first=True)
            
            tree_input, lstm_input = prepare_inputs(models, df_prep)
            
            # Deconstruct predict_pd to see internals
            xgb = models["xgb"]
            lgb = models["lgb"]
            # cat = models["cat"] # assuming loaded
            
            # Tree Probs (Vector)
            px = xgb.predict_proba(tree_input)[:, 1]
            pl = lgb.predict_proba(tree_input)[:, 1]
            
            tree_mean = np.mean((px + pl)/2) # Approximation
            
            # Final Prediction (using current main.py logic)
            pd_score = predict_pd(models, tree_input, lstm_input)
            
            final_score = 0
            if isinstance(pd_score, np.ndarray) and pd_score.size > 1:
                final_score = float(np.mean(pd_score))
            elif hasattr(pd_score, "item"):
                final_score = float(pd_score.item())
            else:
                 final_score = float(pd_score)
                 
            # Actual
            actual_risk = df_eng.iloc[-1].get("risk_level_latent", -1)
            
            results.append({
                "id": cid,
                "tree_min": round(px.min(), 3),
                "tree_max": round(px.max(), 3),
                "tree_mean": round(float(tree_mean), 3),
                "final_score": round(final_score, 3),
                "actual_latent": actual_risk,
                "months": len(hist_df)
            })
            
        except Exception as e:
            print(f"Failed for {cid}: {e}")

    # Print Summary
    print("\nResults Summary:")
    print(f"{'ID':<15} {'TreeRange':<15} {'TreeMean':<10} {'Final':<10} {'Actual':<10} {'Months'}")
    for r in results:
        print(f"{r['id']:<15} {r['tree_min']}-{r['tree_max']:<10} {r['tree_mean']:<10} {r['final_score']:<10} {r['actual_latent']:<10} {r['months']}")

if __name__ == "__main__":
    validate_variance()
