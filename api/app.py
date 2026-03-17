from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

model = joblib.load("model/model.pkl")

Instrumentator().instrument(app).expose(app)


class PatientData(BaseModel):
    features: list


@app.get("/")
def home():
    return {"message": "Healthcare MLOps API"}


@app.post("/predict")
def predict(data: PatientData):

    arr = np.array(data.features).reshape(1, -1)

    prediction = model.predict(arr)
    probability = model.predict_proba(arr)

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0][1])
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)