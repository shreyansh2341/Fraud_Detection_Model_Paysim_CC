import numpy as np
import pandas as pd

from src.model_loader import (
    load_paysim_models,
    load_creditcard_models,
    load_lstm_model
)

# Load models once
PAYSIM = load_paysim_models()
CREDITCARD = load_creditcard_models()
LSTM = load_lstm_model()


def predict_paysim(engineered_df: pd.DataFrame, lstm_sequence=None):
    """
    PaySim prediction with proper feature handling.
    Scale 13 features, then add ae_error as 14th.
    """
    
    # Base features (13) - these get scaled
    base_features = [
        'amount', 'oldbalanceorg', 'newbalanceorig', 'oldbalancedest',
        'newbalancedest', 'hour', 'dayofweek', 'is_weekend',
        'errorbalanceorig', 'errorbalancedest', 'has_balance_mismatch',
        'upi_type_upi_payment', 'upi_type_upi_transfer'
    ]
    
    X = engineered_df.copy()
    
    # Ensure base features exist
    for col in base_features:
        if col not in X.columns:
            X[col] = 0.0
    
    X_base = X[base_features]
    
    # ✅ STEP 1: Scale the 13 base features
    X_scaled = PAYSIM["scaler"].transform(X_base)
    
    # ✅ STEP 2: Convert to DataFrame and add ae_error
    X_scaled_df = pd.DataFrame(X_scaled, columns=base_features)
    X_scaled_df['ae_error'] = 0.0  # 14th feature (NOT scaled)
    
    # ✅ STEP 3: Convert to numpy array with all 14 features
    all_features = base_features + ['ae_error']
    X_final = X_scaled_df[all_features].values
    
    # ✅ STEP 4: Predict (XGBoost expects 14 features)
    prob = float(PAYSIM["xgb"].predict_proba(X_final)[0, 1])
    decision = prob >= PAYSIM["threshold"]
    
    explanation = [f"XGBoost Stage2 prob={prob:.4f}"]
    
    # LSTM support
    if lstm_sequence is not None:
        lstm_sequence = np.asarray(lstm_sequence, dtype=np.float32)
        if lstm_sequence.shape == (LSTM["seq_len"], len(LSTM["features"])):
            try:
                lstm_prob = float(
                    LSTM["model"].predict(
                        lstm_sequence.reshape(1, *lstm_sequence.shape),
                        verbose=0
                    )[0, 0]
                )
                if lstm_prob > 0.75:
                    decision = True
                    explanation.append(f"LSTM fraud pattern ({lstm_prob:.3f})")
            except:
                pass  # LSTM failed, continue without it
    
    return {
        "decision": decision,
        "score": prob,
        "explanation": " | ".join(explanation),
    }

def predict_creditcard(engineered_df: pd.DataFrame):
    """Credit Card prediction (unchanged)."""
    features = CREDITCARD["features"]
    
    missing = set(features) - set(engineered_df.columns)
    if missing:
        raise ValueError(f"Missing Credit Card features: {missing}")
    
    X = engineered_df[features]
    X_scaled = CREDITCARD["scaler"].transform(X)
    
    prob = CREDITCARD["xgb"].predict_proba(X_scaled)[0, 1]
    decision = prob >= CREDITCARD["threshold"]
    
    explanation = f"XGBoost prob={prob:.4f}"
    
    return {
        "decision": decision,
        "score": float(prob),
        "explanation": explanation,
    }


def ensemble_predict(transaction_type, raw_df, lstm_sequence=None):
    """Unified entry point."""
    if transaction_type == "paysim":
        return predict_paysim(raw_df, lstm_sequence)
    
    if transaction_type == "creditcard":
        return predict_creditcard(raw_df)
    
    raise ValueError("Invalid transaction_type")