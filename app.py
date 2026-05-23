from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from google import genai as google_genai

# Setup Gemini API menggunakan SDK terbaru (google-genai)
# Model: gemini-2.5-flash (dikonfirmasi aktif)
gemini_client = google_genai.Client(api_key="AIzaSyBMu5HWGVlx531shbMfD0vBcIXzqoOH-94")
GEMINI_MODEL = "models/gemini-2.5-flash"

app = FastAPI(title="MindEase AI API")

# 1. Load Model
try:
    model = tf.keras.models.load_model('model_mindease_final.keras')
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error Loading Model: {e}")

# 2. Schema Input (21 fitur setelah Feature Engineering dari tim DS)
class HealthData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "MindEase AI API is Running", "status": "Ready"}

@app.post("/predict")
def predict(data: HealthData):
    if len(data.features) != 21:
        raise HTTPException(status_code=400, detail="Input must contain exactly 21 features")
    
    # Preprocessing Input
    input_array = np.array([data.features], dtype=np.float32)
    
    # Inference
    predictions = model.predict(input_array)
    
    # Parsing Results
    risk_probs = predictions[0][0]
    burnout_score = float(predictions[1][0][0])
    
    risk_idx = int(np.argmax(risk_probs))
    risk_labels = ['High', 'Low', 'Medium']
    risk_level = risk_labels[risk_idx]
    
    # Memanggil Gemini 2.5 Flash API untuk rekomendasi
    try:
        prompt = f"Seorang mahasiswa diprediksi memiliki tingkat risiko mental '{risk_level}' dengan skor burnout {burnout_score:.2f}. Berikan 2 kalimat saran yang sangat empatik dan memotivasi untuknya."
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        ai_recommendation = response.text.strip()
    except Exception as e:
        ai_recommendation = "Maaf, sistem rekomendasi AI sedang mengalami gangguan."
    
    return {
        "risk_level": risk_level,
        "burnout_score": round(burnout_score, 2),
        "probabilities": {
            "high": round(float(risk_probs[0]), 4),
            "low": round(float(risk_probs[1]), 4),
            "medium": round(float(risk_probs[2]), 4)
        },
        "genai_recommendation": ai_recommendation
    }

# Cara menjalankan: uvicorn app:app --reload
