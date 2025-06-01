import os
import numpy as np
import traceback

from model import ImagePreprocessor, OnnxModel


def test_model_loading(onnx_path: str):
    try:
        model = OnnxModel(onnx_path)
        print("[✓] ONNX model loaded successfully.")
        return model
    except Exception as e:
        print("[✗] Failed to load ONNX model.")
        traceback.print_exc()
        raise e


def test_preprocessing(image_path: str):
    try:
        preprocessor = ImagePreprocessor()
        processed_input = preprocessor.preprocess(image_path)
        assert processed_input.shape == (1, 3, 224, 224), f"Unexpected input shape: {processed_input.shape}"
        print("[✓] Image preprocessed successfully.")
        return processed_input
    except FileNotFoundError:
        print(f"[✗] Image not found: {image_path}")
        raise
    except Exception as e:
        print("[✗] Error during image preprocessing.")
        traceback.print_exc()
        raise e


def test_inference(model: OnnxModel, input_tensor: np.ndarray):
    try:
        output = model.predict(input_tensor)
        assert isinstance(output, np.ndarray), "Model output is not a numpy array."
        assert output.shape[0] == 1, f"Expected batch size 1, got {output.shape[0]}"
        print("[✓] Inference completed successfully.")
        print("Predicted class index:", np.argmax(output, axis=1)[0])
    except Exception as e:
        print("[✗] Inference failed.")
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    ONNX_MODEL_PATH = "./Resnet_Model.onnx"
    TEST_IMAGE_PATH = "n01667114_mud_turtle.JPEG"

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"[✗] ONNX model not found at: {ONNX_MODEL_PATH}")
        exit(1)

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[✗] Test image not found at: {TEST_IMAGE_PATH}")
        exit(1)

    print("🚀 Starting full model pipeline test...")

    try:
        model = test_model_loading(ONNX_MODEL_PATH)
        input_tensor = test_preprocessing(TEST_IMAGE_PATH)
        test_inference(model, input_tensor)
        print("✅ All tests passed successfully.")
    except Exception:
        print("❌ Tests failed. Check logs above for details.")
