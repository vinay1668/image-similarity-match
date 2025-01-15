import io
import os
import faiss
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms
from PIL import Image
import cairosvg

app = FastAPI()

# Initialize ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()
# Remove the last layer to get features instead of classification
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# Transform pipeline for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize FAISS index
feature_dimension = 2048
index = faiss.IndexFlatL2(feature_dimension)
image_paths = []

def svg_to_pil(svg_content):
    png_data = cairosvg.svg2png(bytestring=svg_content)
    return Image.open(io.BytesIO(png_data))

def extract_features(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image_tensor)
    return features.squeeze().numpy()

@app.post("/vectorize")
async def vectorize(file: UploadFile = File(...)):
    # Read SVG content
    svg_content = await file.read()
    
    # Convert SVG to PIL Image
    pil_image = svg_to_pil(svg_content)
    
    # Extract features
    features = extract_features(pil_image)
    
    # Add to FAISS index
    index.add(features.reshape(1, -1))
    
    # Store file path
    image_paths.append(file.filename)
    
    return {"message": f"Successfully vectorized {file.filename}"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read SVG content
    svg_content = await file.read()
    
    # Convert SVG to PIL Image
    pil_image = svg_to_pil(svg_content)
    
    # Extract features
    features = extract_features(pil_image)
    
    # Search in FAISS index
    D, I = index.search(features.reshape(1, -1), 1)
    
    # Check if there's a match (using a threshold)
    threshold = 100  # Adjust this threshold based on your needs
    if D[0][0] < threshold:
        return {
            "match_found": True,
            "matched_image": image_paths[I[0][0]],
            "similarity_score": float(D[0][0])
        }
    
    return {
        "match_found": False,
        "similarity_score": float(D[0][0])
    }
