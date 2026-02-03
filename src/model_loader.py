import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

# Path setup
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def load_paysim_models():
    """Load PaySim Stage 2 models including autoencoder."""
    return {
        # Main models
        "xgb": joblib.load(MODELS_DIR / "paysim_xgb_stage2.pkl"),
        "rf": joblib.load(MODELS_DIR / "paysim_rf_stage2.pkl"),
        
        # Scalers
        "scaler": joblib.load(MODELS_DIR / "paysim_stage2_scaler.pkl"),
        
        # Autoencoder (for ae_error feature)
        "ae": tf.keras.models.load_model(
            MODELS_DIR / "paysim_ae_model.keras",
            compile=False
        ),
        "ae_scaler": joblib.load(MODELS_DIR / "paysim_stage2_scaler.pkl"),
        
        # Metadata
        "features": joblib.load(MODELS_DIR / "paysim_stage2_features.pkl"),
        "threshold": float(np.load(MODELS_DIR / "paysim_stage2_threshold.npy").item()),
    }


def load_creditcard_models():
    """Load Credit Card models."""
    return {
        "xgb": joblib.load(MODELS_DIR / "cc_xgb_model.pkl"),
        "scaler": joblib.load(MODELS_DIR / "cc_scaler.pkl"),
        "features": joblib.load(MODELS_DIR / "cc_features.pkl"),
        "threshold": float(np.load(MODELS_DIR / "cc_threshold.npy").item()),
    }


def load_lstm_model():
    """Load LSTM model."""
    return {
        "model": tf.keras.models.load_model(
            MODELS_DIR / "fraud_lstm_model_focal.keras",
            compile=False
        ),
        "features": joblib.load(MODELS_DIR / "lstm_features.pkl"),
        "seq_len": joblib.load(MODELS_DIR / "lstm_seq_len.pkl"),
    }