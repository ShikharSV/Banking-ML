from fastapi import FastAPI
from pydantic import BaseModel
import joblib

bundle = joblib.load("intent_model.joblib")
vec, model, labels = bundle["vec"], bundle["model"], bundle["labels"]

app = FastAPI(title="Banking Intent Classifier")

class Query(BaseModel):
    text: str

@app.post("/predict")
def predict(q: Query):
    pred = model.predict(vec.transform([q.text]))[0]
    return {"intent": labels[pred]}