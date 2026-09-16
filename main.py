import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AgroShield ML Damage Detection API",
    description="Deep learning powered crop leaf damage classification REST API",
    version="1.0.0"
)

# Enable CORS for external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "agrimodel.h5")
IMG_SIZE = 224
PREDICTION_THRESHOLD = 0.5

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("[INFO] Model path:", MODEL_PATH)
print("[INFO] Exists?:", os.path.exists(MODEL_PATH))

# ================= LOAD MODEL =================
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("[SUCCESS] Model loaded successfully")
except Exception as e:
    print("[ERROR] Model loading failed:", e)
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


# ================= ROOT & HEALTH CHECK =================
@app.get("/", summary="Backend API Status")
async def root():
    return {
        "status": "online",
        "service": "AgroShield ML Damage Detection API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "predict": {
                "method": "POST",
                "path": "/predict",
                "description": "Upload leaf image(s) for damage classification"
            },
            "health": {
                "method": "GET",
                "path": "/health",
                "description": "API & model health status"
            }
        }
    }


@app.get("/health", summary="Health Check")
async def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None
    }


@app.get("/samples/{kind}/{name}", summary="Serve sample test images")
async def get_sample(kind: str, name: str):
    folder_map = {
        "damaged": os.path.join(BASE_DIR, "image data", "damaged image"),
        "undamaged": os.path.join(BASE_DIR, "image data", "undamaged image")
    }
    target_dir = folder_map.get(kind.lower())
    if not target_dir:
        raise HTTPException(status_code=404, detail="Category not found")
    
    file_path = os.path.join(target_dir, name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Sample image not found")

    media_type = "image/jpeg" if name.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return FileResponse(file_path, media_type=media_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)