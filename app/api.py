"""
===========================================================
Fake News Detection API
===========================================================

REST API using FastAPI

Endpoints

GET  /
GET  /health
POST /predict
POST /batch

Author : Mahmoud Morsy
===========================================================
"""

# ===========================================================
# Imports
# ===========================================================

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from app.predict import predict
from app.predict import predict_batch


# ===========================================================
# FastAPI App
# ===========================================================

app = FastAPI(

    title="Fake News Detection API",

    description="Detect Fake and Real News using Linear SVM",

    version="1.0.0"

)


# ===========================================================
# Request Models
# ===========================================================

class PredictionRequest(BaseModel):

    text: str


class BatchPredictionRequest(BaseModel):

    texts: List[str]


# ===========================================================
# Home Endpoint
# ===========================================================

@app.get("/")

def home():

    return {

        "project": "Fake News Detection",

        "model": "Linear SVM",

        "status": "Running"

    }


# ===========================================================
# Health Check
# ===========================================================

@app.get("/health")

def health():

    return {

        "status": "healthy"

    }


# ===========================================================
# Predict One News
# ===========================================================

@app.post("/predict")

def predict_news(

    request: PredictionRequest

):

    result = predict(

        request.text

    )

    return result


# ===========================================================
# Predict Multiple News
# ===========================================================

@app.post("/batch")

def predict_multiple(

    request: BatchPredictionRequest

):

    return predict_batch(

        request.texts

    )


# ===========================================================
# Run API
# ===========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "api:app",

        host="0.0.0.0",

        port=8000,

        reload=True

    )