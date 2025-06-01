# 🧠 Image Classification API Deployment (Cerebrium + ResNet)

This project showcases how to deploy a PyTorch-trained image classification model (ResNet18) to Cerebrium's serverless GPU platform using ONNX and Docker. It includes tools to test locally, convert the model, and deploy to production with ease.

---

## 📁 Folder Structure

```bash
├── Deployment/                        # Main deployment folder
│   ├── Dockerfile                     # Dockerfile for container-based deployment
│   ├── README.md                      # This README file
│   ├── cerebrium.toml                 # Cerebrium configuration
│   ├── main.py                        # FastAPI app to serve ONNX model
│   ├── requirements.txt               # Python dependencies
│   └── resnet-model.onnx             # Exported ONNX model
├── Project/
│   ├── deployment_wo_docker/         # Alternative deployment without Docker
│   │   ├── cerebrium.toml
│   │   ├── main.py
│   │   └── resnet-model.onnx
│   ├── test_images/                  # Sample images for testing
│   │   ├── n01440764_tench.jpeg
│   │   └── n01667114_mud_turtle.JPEG
│   ├── .env                          # Environment variables (optional)
│   ├── convert_to_onnx.py            # Script to convert PyTorch to ONNX
│   ├── model.py                      # Model loading & preprocessing utilities
│   ├── pytorch_model.py              # Original PyTorch model architecture
│   ├── pytorch_model_weights.pth     # Trained weights for PyTorch model
│   ├── test.py                       # Test ONNX model locally
│   └── test_server.py                # Sends a request to running API server
```

---

## 🚀 Features

* ✅ PyTorch to ONNX model conversion
* ✅ ONNX runtime inference with FastAPI
* ✅ Dockerized deployment for Cerebrium
* ✅ Local and remote testing utilities
* ✅ Sample test images included

---

## 🧠 Model Details

* **Base Model**: ResNet18 (from `torchvision`)
* **Input Shape**: `(1, 3, 224, 224)`
* **Preprocessing**: Image resized and normalized to match ImageNet standards
* **ONNX Output**: Top-1 classification prediction index

---

## 🛠️ Setup

### 🔧 Requirements

* Python 3.9+
* `onnxruntime`
* `fastapi`
* `uvicorn`
* `Pillow`
* `torch`, `torchvision`
* `requests`

Install via:

```bash
pip install -r Deployment/requirements.txt
```

---

## 🧪 Testing Locally

### 1. Convert PyTorch model to ONNX:

```bash
python Project/convert_to_onnx.py
```

### 2. Run Local Server:

```bash
cd Deployment
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Test Server:

```bash
python Other/test_server.py
```

---

## 🐳 Docker Deployment (Cerebrium)

### 1. Build Docker Image

```bash
cd Deployment
docker build -t resnet-classifier .
```

### 2. Run Locally with Docker

```bash
docker run -p 8192:8192 resnet-classifier
```

### 3. Deploy to Cerebrium

* Ensure you're logged in to Cerebrium
* Use `cerebrium.toml` for deployment configuration

```bash
cerebrium deploy
```

---

## 🖼️ Sample Input

* Format: JPEG or PNG
* Shape: Any (resized internally to 224x224)
* Sample image: `Other/test_images/n01440764_tench.jpeg`

---

## 📤 API Usage

### Endpoint

```http
POST /predict
```

### Request Body

```json
{
  "image_base64": "<base64_encoded_image>"
}
```

### Example with `test_server.py`

```bash
python Other/test_server.py
```

---

## 📝 Author Notes

* Deployment without Docker is available in `Other/deployment_wo_docker/`
* All utility and test files are kept modular for clarity and separation of concerns
* This project was built as part of a take-home MLOps deployment task

## 👤 Author

* **Name**: Arbaz Khan Shehzad
* **Email**: arbazkhanshehzad@gmail.com
* **Phone**: +92 346 627 4546
* **GitHub**: https://github.com/arbazkhanshehzad 
* **LinkedIn**: https://linkedin.com/in/arbaz-khan-shehzad
