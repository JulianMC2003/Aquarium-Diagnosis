from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

app = FastAPI(title="Fish Disease Predictor")

# FastAPI CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SymptomRequest(BaseModel):
    symptoms: str

# ML artifacts
model = joblib.load("fish_disease_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
df = pd.read_json("aquarium_diseases.json")

# Predictions
@app.post("/predict", response_class=PlainTextResponse)
def predict_disease(request: SymptomRequest):
    vec = vectorizer.transform([request.symptoms])
    predicted = model.predict(vec)[0]
    
    # try to find the row safely
    matches = df[df['Disease / Syndrome'].str.strip().str.lower() == predicted.strip().lower()]
    
    if not matches.empty:
        row = matches.iloc[0]
        return (
            f"Your fish may be suffering from the following:\n"
            f"Disease: {row['Disease / Syndrome']}\n"
            f"Symptoms: {row['Common Symptoms']}\n"
            f"Treatment: {row['Common Treatment Options']}"
        )
    else:
        return f"No exact match found for '{predicted}'. Please try again with different symptoms."

@app.get("/health")
def health():
    return {"status": "ok"}

