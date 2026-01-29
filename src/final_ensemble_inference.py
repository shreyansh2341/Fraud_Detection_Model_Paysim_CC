# import os
# import numpy as np
# import joblib
# import tensorflow as tf

# # ======================================================
# # ABSOLUTE PATH RESOLUTION
# # ======================================================
# CURRENT_DIR = os.path.dirname(__file__)
# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# # ======================================================
# # CONFIG
# # ======================================================
# LSTM_THRESHOLD_LOW  = 0.20
# LSTM_THRESHOLD_HIGH = 0.35

# LSTM_N_FEATURES = 12
# PAYSIM_STAGE2_N_FEATURES = 13

# # ======================================================
# # LOAD MODELS & ARTIFACTS
# # ======================================================
# print("Loading models and artifacts...")

# lstm_model = tf.keras.models.load_model(
#     os.path.join(MODELS_DIR, "fraud_lstm_model_focal.keras"),
#     compile=False
# )

# paysim_scaler = joblib.load(
#     os.path.join(MODELS_DIR, "paysim_stage2_scaler.pkl")
# )
# paysim_xgb = joblib.load(
#     os.path.join(MODELS_DIR, "paysim_xgb_stage2.pkl")
# )
# paysim_threshold = np.load(
#     os.path.join(MODELS_DIR, "paysim_stage2_threshold.npy")
# )[0]

# paysim_ae = tf.keras.models.load_model(
#     os.path.join(MODELS_DIR, "paysim_ae_model.keras"),
#     compile=False
# )

# cc_scaler = joblib.load(
#     os.path.join(MODELS_DIR, "cc_scaler.pkl")
# )
# cc_xgb = joblib.load(
#     os.path.join(MODELS_DIR, "cc_xgb_model.pkl")
# )
# cc_threshold = np.load(
#     os.path.join(MODELS_DIR, "cc_threshold.npy")
# )[0]

# print("All models loaded successfully.")

# # ======================================================
# # FINAL ENSEMBLE
# # ======================================================
# def ensemble_predict(
#     transaction_type: str,
#     tabular_features: np.ndarray,
#     lstm_sequence: np.ndarray = None
# ):
#     # ==================================================
#     # STAGE 1: LSTM (PaySim)
#     # ==================================================
#     lstm_score = None

#     if transaction_type == "paysim":
#         if lstm_sequence is None:
#             raise ValueError("lstm_sequence required for PaySim")

#         if lstm_sequence.shape[-1] != LSTM_N_FEATURES:
#             lstm_sequence = lstm_sequence[:, :, :LSTM_N_FEATURES]

#         lstm_score = float(
#             lstm_model.predict(lstm_sequence, verbose=0)[0][0]
#         )

#         if lstm_score < LSTM_THRESHOLD_LOW:
#             return 0, f"Approved: low temporal risk (LSTM={lstm_score:.3f})"

#         if lstm_score < LSTM_THRESHOLD_HIGH:
#             return 0, f"Approved: medium temporal risk (LSTM={lstm_score:.3f})"

#     # ==================================================
#     # STAGE 2: PAYSIM (AE + XGB)
#     # ==================================================
#     if transaction_type == "paysim":
#         if tabular_features.shape[1] != PAYSIM_STAGE2_N_FEATURES:
#             tabular_features = tabular_features[:, :PAYSIM_STAGE2_N_FEATURES]

#         X_scaled = paysim_scaler.transform(tabular_features)

#         recon = paysim_ae.predict(X_scaled, verbose=0)
#         ae_error = np.log1p(
#             np.mean((X_scaled - recon) ** 2, axis=1)
#         )

#         X_final = np.column_stack([X_scaled, ae_error])
#         prob = paysim_xgb.predict_proba(X_final)[0, 1]

#         if prob >= paysim_threshold:
#             return 1, (
#                 f"Fraud: PaySim confirmed "
#                 f"(prob={prob:.4f}, LSTM={lstm_score:.3f})"
#             )

#         return 0, (
#             f"Approved: PaySim Stage-2 rejected "
#             f"(LSTM={lstm_score:.3f})"
#         )

#     # ==================================================
#     # STAGE 2: CREDIT CARD (XGB)
#     # ==================================================
#     if transaction_type == "creditcard":
#         X_scaled = cc_scaler.transform(tabular_features)
#         prob = cc_xgb.predict_proba(X_scaled)[0, 1]

#         if prob >= cc_threshold:
#             return 1, (
#                 f"Fraud: Credit Card confirmed "
#                 f"(prob={prob:.4f})"
#             )

#         return 0, (
#             f"Approved: Credit Card rejected "
#             f"(prob={prob:.4f})"
#         )

#     raise ValueError("transaction_type must be 'paysim' or 'creditcard'")


import os
import numpy as np
import joblib
import tensorflow as tf

# ======================================================
# CONFIG & PATHS
# ======================================================
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Weights for generalization (adjust based on model confidence)
W_XGB = 0.50   # Tabular patterns (Strongest)
W_LSTM = 0.30  # Temporal sequences
W_AE = 0.20    # Reconstruction anomaly

# ======================================================
# LOAD MODELS & ARTIFACTS
# ======================================================
print("Initializing Parallel Ensemble Stack...")

lstm_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "fraud_lstm_model_focal.keras"),
    compile=False
)

paysim_scaler = joblib.load(
    os.path.join(MODELS_DIR, "paysim_stage2_scaler.pkl")
)

paysim_xgb = joblib.load(
    os.path.join(MODELS_DIR, "paysim_xgb_stage2.pkl")
)

paysim_ae = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "paysim_ae_model.keras"),
    compile=False
)

paysim_threshold = np.load(
    os.path.join(MODELS_DIR, "paysim_stage2_threshold.npy")
)[0]

def ensemble_predict(transaction_type, raw_df, lstm_sequence=None):
    """
    Parallel Expert Fusion: Runs all models and combines scores.
    Replaces the sequential 'gatekeeper' for better generalization.
    """
    
    if transaction_type != "paysim":
        return 0, "Non-PaySim logic not yet integrated into parallel stack."

    try:
        # 1. Scaling & Autoencoder (Expert 1: Unsupervised Anomaly)
        # We use the raw_df passed from inference.py (already engineered)
        X_scaled = paysim_scaler.transform(raw_df)
        
        recon = paysim_ae.predict(X_scaled, verbose=0)
        ae_error = np.log1p(np.mean((X_scaled - recon) ** 2, axis=1))
        
        # 2. XGBoost (Expert 2: Supervised Tabular)
        # We augment the scaled features with the AE error
        X_final = np.column_stack([X_scaled, ae_error])
        xgb_prob = paysim_xgb.predict_proba(X_final)[0, 1]

        # 3. LSTM (Expert 3: Temporal Context)
        lstm_prob = float(lstm_model.predict(lstm_sequence, verbose=0)[0][0])

        # 4. Weighted Fusion Decision
        # AE error is converted to a 0-1 probability-like score for fusion
        ae_prob = min(ae_error[0] / 10.0, 1.0) 
        
        final_score = (W_XGB * xgb_prob) + (W_LSTM * lstm_prob) + (W_AE * ae_prob)

        # 5. Final Output
        if final_score >= paysim_threshold:
            return 1, (
                f"FRAUD DETECTED: Aggregate Score {final_score:.3f} "
                f"(XGB: {xgb_prob:.2f}, LSTM: {lstm_prob:.2f}, AE: {ae_prob:.2f})"
            )

        return 0, f"LEGIT: Aggregate Score {final_score:.3f} is below threshold."

    except Exception as e:
        return -1, f"ENSEMBLE ERROR: {str(e)}"