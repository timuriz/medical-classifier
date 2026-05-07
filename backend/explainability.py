import torch
import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, image_tensor, original_image):
    """
    Generate Grad-CAM heatmap for an image
    
    Args:
        model: Your trained EfficientNet model
        image_tensor: Preprocessed tensor [1, 3, 224, 224]
        original_image: PIL Image (for overlay)
    
    Returns:
        visualization: numpy array with heatmap overlay
    """
    
    # 1. Choose which layer to visualize
    # For EfficientNet, use the last feature layer
    target_layer = model.features[-1]
    
    # 2. Create GradCAM object
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # 3. Generate heatmap
    heatmap = cam(input_tensor=image_tensor)
    
    # 4. Convert original image to numpy array [0,1]
    original_image = original_image.resize((224, 224))
    original_np = np.array(original_image).astype(np.float32) / 255.0
    
    # 5. Overlay heatmap on image
    visualization = show_cam_on_image(
        original_np,
        heatmap[0],
        use_rgb=True
    )
    
    # 6. Return as PIL Image
    return Image.fromarray(visualization)