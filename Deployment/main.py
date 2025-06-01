from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import base64
import time
import uuid

app = FastAPI()

# Initialize the ONNX model session
session = ort.InferenceSession("resnet-model.onnx")
input_name = session.get_inputs()[0].name
input_size = (224, 224)

class ImageRequest(BaseModel):
    image_base64: str

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize(input_size)
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_np = ((image_np - mean) / std).astype(np.float32)
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)
    return image_np


@app.get("/health")
def health():
    return "OK"

@app.post("/hello")
def hello():
    return {"message": "Hello Cerebrium! ONNX ResNet model inference API. POST base64 image to /predict"}

@app.post("/predict")
async def predict(request: ImageRequest):
    run_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    image_base64 = request.image_base64

    if len(image_base64) > 5 * 1024 * 1024:  # 5MB limit
        raise HTTPException(status_code=400, detail="Image size exceeds limit (5MB)")

    try:
        print(f"[{run_id}] Decoding image...")
        image_data = base64.b64decode(image_base64)

        print(f"[{run_id}] Preprocessing image...")
        input_tensor = preprocess_image(image_data)

        if time.time() - start_time > 10:
            raise HTTPException(status_code=408, detail="Preprocessing timeout exceeded (10s)")

        print(f"[{run_id}] Running ONNX model inference...")
        outputs = session.run(None, {input_name: input_tensor})

        if time.time() - start_time > 15:
            raise HTTPException(status_code=408, detail="Inference timeout exceeded (15s)")

        predicted_class = int(np.argmax(outputs[0]))
        print(f"[{run_id}] Prediction complete: {predicted_class}")
        return {"predicted_class": predicted_class}

    except Exception as e:
        print(f"[{run_id}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))