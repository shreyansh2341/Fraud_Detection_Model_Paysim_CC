import sys
import os

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from schemas import FraudRequest, FraudResponse
from inference import run_inference


app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="LSTM + AE + XGBoost Ensemble Fraud System",
    version="1.0"
)

@app.post("/predict", response_model=FraudResponse)
def predict_fraud(request: FraudRequest):
    decision, reason = run_inference(request)

    return FraudResponse(
        decision="FRAUD" if decision == 1 else "LEGIT",
        explanation=reason
    )
