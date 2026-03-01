import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="AgroShield ML Damage Detection API")

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "agrimodel.h5")
IMG_SIZE = 224
PREDICTION_THRESHOLD = 0.5

print("📂 Model path:", MODEL_PATH)
print("📂 Exists?:", os.path.exists(MODEL_PATH))

# ================= LOAD MODEL =================
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None



# ================= IMAGE PREPROCESS =================
def preprocess_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception:
        return None


# ================= PREDICT ENDPOINT =================
@app.post(
    "/predict",
    summary="Predict crop damage from multiple images",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "format": "binary"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def predict(files: List[UploadFile] = File(...)):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No images uploaded")

    results = []
    total_confidence = 0
    valid_predictions = 0

    for file in files:
        try:
            contents = await file.read()

            if not contents:
                continue

            img = preprocess_image(contents)

            if img is None:
                continue

            prediction = model.predict(img, verbose=0)[0][0]
            score = float(prediction)

            # 🔥 CORRECT LOGIC (Matches your original Flask app)
            # 1 = Undamaged
            # 0 = Damaged
            if score > PREDICTION_THRESHOLD:
                label = "Undamaged"
                confidence = score
            else:
                label = "Damaged"
                confidence = 1 - score

            total_confidence += confidence
            valid_predictions += 1

            results.append({
                "filename": file.filename,
                "label": label,
                "confidence": round(confidence * 100, 2)
            })

        except Exception:
            continue

    if valid_predictions == 0:
        return {
            "results": [],
            "finalConfidence": 0
        }

    avg_confidence = total_confidence / valid_predictions

    return {
        "results": results,
        "finalConfidence": round(avg_confidence * 100, 2)
    }