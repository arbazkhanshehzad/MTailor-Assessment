import argparse
import requests
import json
from PIL import Image
import base64
from dotenv import load_dotenv
import os


load_dotenv() 

def image_to_base64(image_path):
    """Converts an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred while encoding the image: {e}")
        return None

BEARER_TOKEN = os.getenv("BEARER_TOKEN")
print("Using Bearer Token:", BEARER_TOKEN)
# Cerebrium deployment URL
API_URL = "https://api.cortex.cerebrium.ai/v4/p-e8715703/onnx-inference-mtailor/predict"
HEADERS = {
    "Authorization": BEARER_TOKEN,  
    "Content-Type": "application/json"
}

def predict_from_server(image_path):
    # Preprocess image and convert to list for JSON serialization
    image_base64 = image_to_base64(image_path)
    payload = {
        "image_base64": image_base64
    }

    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
    print("Status Code:", response.status_code)
    try:
        print("Prediction:", response.json())
    except Exception as e:
        print("Error parsing response:", e)

def run_custom_tests():
    print("Running custom tests on Cerebrium deployment...")
    test_images = [
        "./test_images/n01440764_tench.jpeg",
        "./test_images/n01667114_mud_turtle.JPEG"
    ]
    for path in test_images:
        print(f"\nTesting image: {path}")
        predict_from_server(path)

if __name__ == "__main__":
    predict_from_server("./test_images/n01440764_tench.jpeg")  # Example image path
    run_custom_tests()

