"""Minimal prediction API. Run after train_models.py:
    pip install fastapi uvicorn joblib pandas
    uvicorn api:app --reload
POST /predict with JSON like {"age": 62, "bp": 90, "sc": 4.2, ...}
Missing fields are filled with 0, so send all numeric fields the model was trained on.
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
xgb, lgbm = joblib.load("xgb.joblib"), joblib.load("lgbm.joblib")
cols = joblib.load("feature_columns.joblib")

@app.post("/predict")
def predict(record: dict):
    row = pd.DataFrame([record]).reindex(columns=cols, fill_value=0)
    return {"xgboost": float(xgb.predict_proba(row)[0, 1]),
            "lightgbm": float(lgbm.predict_proba(row)[0, 1])}
