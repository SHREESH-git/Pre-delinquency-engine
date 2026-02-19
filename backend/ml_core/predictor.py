import numpy as np
import torch


# =========================================================
# PREPARE MODEL INPUTS
# =========================================================
def prepare_inputs(models, df):

    tree_cols = models["tree_feature_cols"]
    lstm_cols = models["lstm_feature_cols"]
    seq_len = models["seq_len"]
    scaler = models["scaler"]

    # ---------------- TREE INPUT ----------------
    df_tree = df.reindex(columns=tree_cols, fill_value=0)
    tree_input = df_tree  # Use ALL rows to capture rolling progress across history

    # ---------------- LSTM INPUT ----------------
    df_lstm = df.reindex(columns=lstm_cols, fill_value=0)

    # If not enough history → PAD from top
    if len(df_lstm) < seq_len:
        padding_rows = seq_len - len(df_lstm)
        padding = np.zeros((padding_rows, len(lstm_cols)))
        sequence = np.vstack([padding, df_lstm.values])
    else:
        # User requested strictly 3 months rolling history for LSTM
        # So we take the LAST seq_len rows
        sequence = df_lstm.values[-seq_len:]
    
    # Scale
    sequence_scaled = scaler.transform(
        sequence.reshape(-1, sequence.shape[-1])
    ).reshape(1, sequence.shape[0], -1)

    return tree_input, sequence_scaled


# =========================================================
# PREDICT PD
# =========================================================
def predict_pd(models, tree_input, lstm_input):
    from ml_core.risk_engine import expected_loss, risk_bucket

    xgb = models["xgb"]
    lgb = models["lgb"]
    cat = models["cat"]
    lstm = models["lstm"]
    calibrator = models["calibrator"]

    wx = models["wx"]
    wl = models["wl"]
    wc = models["wc"]
    best_weight = models["best_weight"]

    config = models["config"]

    # ---------------- TREE ENSEMBLE ----------------
    px = xgb.predict_proba(tree_input)[:, 1]
    pl = lgb.predict_proba(tree_input)[:, 1]
    
    if cat is not None:
        pc = cat.predict_proba(tree_input)[:, 1]
        s = wx + wl + wc
        tree_pd = (wx * px + wl * pl + wc * pc) / s
    else:
        # Redistribute wc to wx and wl equally or proportionally
        s = wx + wl
        tree_pd = (wx * px + wl * pl) / s

    # ---------------- LSTM ----------------
    device = next(lstm.parameters()).device
    tensor_input = torch.tensor(lstm_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        lstm_pd = lstm(tensor_input).cpu().numpy()

    # ---------------- HYBRID BLEND ----------------
    blended = best_weight * tree_pd + (1 - best_weight) * lstm_pd
    
    return blended
