from pydantic import BaseModel
from typing import List, Optional

class FraudRequest(BaseModel):
    transaction_type: str  # "paysim" or "creditcard"
    tabular_features: List[float]
    lstm_sequence: Optional[List[List[float]]] = None

class FraudResponse(BaseModel):
    decision: str
    explanation: str
