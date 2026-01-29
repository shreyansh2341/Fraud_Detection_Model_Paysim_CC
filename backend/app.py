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
