import torch
from PIL import Image
from torchvision import transforms
from pytorch_model import Classifier, BasicBlock  # Replace this

def preprocess_image(img_path):
    img = Image.open(img_path)
    resize = transforms.Resize((224, 224))
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    return img.unsqueeze(0)  # add batch dim

def export_model_to_onnx(model, onnx_path, dummy_input):
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load("./pytorch_model_weights.pth"))
    model.eval()

    image_path = "/content/n01440764_tench.jpeg"
    input_tensor = preprocess_image(image_path)
    
    # Run inference to verify
    with torch.no_grad():
        output = model(input_tensor)
    print("Predicted class index:", torch.argmax(output).item())

    # Export ONNX
    export_model_to_onnx(model, "resnet18.onnx", input_tensor)
