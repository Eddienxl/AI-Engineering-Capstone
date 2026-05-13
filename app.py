from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib # Untuk load scaler jika disimpan terpisah, atau gunakan logic MinMaxScaler manual

app = FastAPI(title="MindEase AI API")

# 1. Load Model
try:
    model = tf.keras.models.load_model('model_mindease_final.keras')
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error Loading Model: {e}")

# 2. Schema Input (Sesuaikan dengan 16 fitur Anda)
class HealthData(BaseModel):
    features: list # List berisi 16 angka fitur yang sudah di-scale (0-1)

@app.get("/")
def home():
    return {"message": "MindEase AI API is Running", "status": "Ready"}

@app.post("/predict")
def predict(data: HealthData):
    if len(data.features) != 16:
        raise HTTPException(status_code=400, detail="Input must contain exactly 16 features")
    
    # Preprocessing Input
    input_array = np.array([data.features], dtype=np.float32)
    
    # Inference
    predictions = model.predict(input_array)
    
    # Parsing Results
    risk_probs = predictions[0][0]
    burnout_score = float(predictions[1][0][0])
    
    risk_idx = int(np.argmax(risk_probs))
    risk_labels = ['High', 'Low', 'Medium'] # Sesuaikan urutan label encoder Anda
    
    return {
        "risk_level": risk_labels[risk_idx],
        "burnout_score": round(burnout_score, 2),
        "probabilities": {
            "high": round(float(risk_probs[0]), 4),
            "low": round(float(risk_probs[1]), 4),
            "medium": round(float(risk_probs[2]), 4)
        }
    }

# Cara menjalankan: uvicorn app:app --reload
