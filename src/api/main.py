"""
FastAPI application for credit risk prediction.
"""

import mlflow
import pandas as pd
import uvicorn
import uuid
import os
from fastapi import FastAPI, HTTPException
from .pydantic_models import PredictionRequest, PredictionResponse

# Create FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk probability using ML model",
    version="1.0.0"
)

# Load the best model from MLflow
def load_model():
    """Load the best model from MLflow registry."""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("file:/app/mlruns")
        
        # Try loading using runs URI first
        try:
            model = mlflow.sklearn.load_model("runs:/latest/gradient_boosting_model")
            return model
        except Exception:
            # Fallback to direct file path
            model_path = "/app/mlruns/0/latest/artifacts/gradient_boosting_model"
            if os.path.exists(model_path):
                model = mlflow.sklearn.load_model(model_path)
                return model
            else:
                raise RuntimeError(f"Model not found at {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

# Initialize model at startup
model = load_model()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Credit Risk Prediction API",
        "docs_url": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk for a customer.
    """
    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.dict()])
        
        # Make prediction
        risk_prob = model.predict_proba(data.values)[0][1]
        is_high_risk = risk_prob >= 0.5
        
        # Create response
        response = PredictionResponse(
            risk_probability=float(risk_prob),
            is_high_risk=bool(is_high_risk),
            model_version="gradient_boosting_v1",
            prediction_id=str(uuid.uuid4())
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 