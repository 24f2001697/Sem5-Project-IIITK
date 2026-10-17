import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Plant Disease Diagnostics API",
    description="Backend API for Potato and Tomato Leaf Disease Prediction"
)

# Enable CORS so the web app can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# LOAD MODELS (Using exact filenames from your Model folder)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POTATO_MODEL_PATH = os.path.join(BASE_DIR, "Model", "potatoes.h5")
TOMATO_MODEL_PATH = os.path.join(BASE_DIR, "Model", "tomato_model.h5")

print(f"Loading Potato Model from: {POTATO_MODEL_PATH}")
potato_model = tf.keras.models.load_model(POTATO_MODEL_PATH)

print(f"Loading Tomato Model from: {TOMATO_MODEL_PATH}")
tomato_model = tf.keras.models.load_model(TOMATO_MODEL_PATH)

# Class mappings
POTATO_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def preprocess_image(image_bytes: bytes, target_size: tuple) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    tensor = np.array(image, dtype=np.float32)
    return np.expand_dims(tensor, axis=0)

@app.get("/")
def read_root():
    return {"status": "Active", "message": "LeafGuard AI Backend API"}

@app.post("/predict/potato")
async def predict_potato(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    contents = await file.read()
    tensor = preprocess_image(contents, target_size=(224, 224))
    
    preds = potato_model.predict(tensor)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    
    return {
        "crop": "Potato",
        "predicted_class": POTATO_CLASSES[idx],
        "confidence": round(confidence * 100, 2),
        "all_predictions": {POTATO_CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(POTATO_CLASSES))}
    }

@app.post("/predict/tomato")
async def predict_tomato(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    contents = await file.read()
    tensor = preprocess_image(contents, target_size=(300, 300))
    
    preds = tomato_model.predict(tensor)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    
    return {
        "crop": "Tomato",
        "predicted_class": TOMATO_CLASSES[idx],
        "confidence": round(confidence * 100, 2),
        "all_predictions": {TOMATO_CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(TOMATO_CLASSES))}
    }