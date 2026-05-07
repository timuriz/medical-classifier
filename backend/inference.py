import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
from explainability import generate_gradcam

class ModelInference:
    def __init__(self, model_path):
        device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

        model = efficientnet_b0(weights=None)

        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, 3)

        model.load_state_dict(torch.load('models/efficientnet_isic_best.pth', map_location=device))
        model = model.to(device)
        model.eval()

        self.model = model
        self.device = device
    
    def predict(self, image):
        inference_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image_tensor = inference_transforms(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        heatmap_image = generate_gradcam(
            self.model,
            image_tensor,
            image
        )

        class_names = ["melanoma", "nevus", "seborrheic_keratosis"]
        return {
            "prediction": class_names[predicted.item()],
            "confidence": confidence.item(),
            "probabilities": probabilities.squeeze().tolist(),
            "heatmap": heatmap_image
        }




model = ModelInference('models/efficientnet_isic_best.pth')
pred = model.predict(Image.open('data/nevus/isic_0000049.jpg'))
print(pred)