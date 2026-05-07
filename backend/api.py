from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import base64

from inference import ModelInference
from explainability import image_to_base64

app = FastAPI(title="Medical Image Classifier")

# Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = ModelInference('models/efficientnet_isic_best.pth')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Get prediction and heatmap
    result = model.predict(image)
    
    # Convert heatmap to base64
    heatmap_base64 = image_to_base64(result['heatmap'])
    
    # Return JSON
    return {
        "prediction": result['prediction'],
        "confidence": float(result['confidence']),
        "probabilities": result['probabilities'],
        "heatmap": heatmap_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)