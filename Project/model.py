import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, resize_size=224, crop_size=224):
        self.resize = transforms.Resize((resize_size, resize_size))
        self.crop = transforms.CenterCrop((crop_size, crop_size))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Load image, apply resizing, cropping, normalization, 
        and convert to a numpy array with batch dimension for ONNX input.
        """
        img = Image.open(image_path).convert('RGB')
        img = self.resize(img)
        img = self.crop(img)
        img = self.to_tensor(img)
        img = self.normalize(img)
        
        # Convert to numpy and add batch dimension
        img_np = img.unsqueeze(0).cpu().numpy()
        return img_np


class OnnxModel:
    def __init__(self, onnx_model_path: str):
        """
        Load ONNX model with onnxruntime InferenceSession.
        """
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on input tensor and return the output predictions.
        """
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return outputs[0]
