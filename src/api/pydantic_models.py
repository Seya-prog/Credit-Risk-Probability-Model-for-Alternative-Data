"""
Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any

class PredictionRequest(BaseModel):
    """Request model for credit risk prediction."""
    TransactionYear: int = Field(..., ge=2000, le=2100)
    TransactionMonth: int = Field(..., ge=1, le=12)
    TransactionDay: int = Field(..., ge=1, le=31)
    TransactionHour: int = Field(..., ge=0, le=23)
    TransactionDayOfWeek: int = Field(..., ge=0, le=6)
    TotalTransactionAmount: float = Field(..., gt=0)
    AverageTransactionAmount: float = Field(..., gt=0)
    TransactionAmountStd: float = Field(..., ge=0)
    TransactionCount: int = Field(..., gt=0)

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionYear": 2023,
                "TransactionMonth": 7,
                "TransactionDay": 2,
                "TransactionHour": 14,
                "TransactionDayOfWeek": 0,
                "TotalTransactionAmount": 1000.0,
                "AverageTransactionAmount": 250.0,
                "TransactionAmountStd": 100.0,
                "TransactionCount": 4
            }
        }

class PredictionResponse(BaseModel):
    """Response model for credit risk prediction."""
    risk_probability: float = Field(..., ge=0, le=1)
    is_high_risk: bool
    model_version: str
    prediction_id: str 