from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import faiss
import torch
from torchvision.models import ResNet50_Weights
from torchvision import models, transforms
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import glob
import os

app = FastAPI(title="SVG Similarity Matcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and FAISS
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

feature_dimension = 2048
index = faiss.IndexFlatL2(feature_dimension)
image_paths = []

def svg_to_pil(svg_content):
    # Convert SVG bytes to PIL Image using svglib
    drawing = svg2rlg(io.BytesIO(svg_content))
    return Image.frombytes('RGB', 
                         (drawing.width, drawing.height), 
                         renderPM.drawToString(drawing, fmt='PNG'))

def extract_features(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image_tensor)
    return features.squeeze().numpy()

@app.post("/vectorize-local")
async def vectorize_local(directory: str = "svg_files/"):
    """Vectorize all SVG files from a local directory"""
    try:
        svg_files = glob.glob(os.path.join(directory, "*.svg"))
        for svg_file in svg_files:
            with open(svg_file, 'rb') as f:
                svg_content = f.read()
                pil_image = svg_to_pil(svg_content)
                features = extract_features(pil_image)
                index.add(features.reshape(1, -1))
                image_paths.append(svg_file)
        
        return {
            "success": True,
            "message": f"Vectorized {len(svg_files)} SVG files",
            "files": image_paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        svg_content = await file.read()
        pil_image = svg_to_pil(svg_content)
        features = extract_features(pil_image)
        
        D, I = index.search(features.reshape(1, -1), 1)
        
        threshold = 100
        if D[0][0] < threshold and len(image_paths) > 0:
            return {
                "match_found": True,
                "matched_image": image_paths[I[0][0]],
                "similarity_score": float(D[0][0])
            }
        
        return {
            "match_found": False,
            "similarity_score": float(D[0][0]) if len(image_paths) > 0 else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status")
async def get_status():
    return {
        "vectorized_images": len(image_paths),
        "index_size": index.ntotal,
        "images": image_paths
    }
