from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import time
import mlflow

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
model = tf.keras.models.load_model("models/image_classifier.h5")
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    print(f"Image shape: {image_array.shape}")
    prediction = model.predict(image_array)
    print(f"Prediction probabilities: {prediction[0]}")
    class_idx = np.argmax(prediction)
    latency = time.time() - start_time
    with mlflow.start_run():
        mlflow.log_metric("prediction_latency", latency)
    return {"class": class_names[class_idx], "confidence": float(prediction[0][class_idx])}