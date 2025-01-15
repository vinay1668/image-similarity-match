# SVG Image Similarity Matcher

A FastAPI-based service that detects similar SVG images using ResNet50 and FAISS vector similarity search.

## Features

- Vectorize local SVG files using ResNet50 feature extraction
- Fast similarity search using FAISS indexing
- REST API endpoints for prediction and status
- Support for batch processing of local SVG files

## Installation

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
README.md
Install required packages:
pip install fastapi uvicorn torch torchvision faiss-cpu svglib python-multipart


Usage
Start the server:
uvicorn app:app --host 127.0.0.1 --port 8000 --reload


API Endpoints:
POST /vectorize-local: Vectorize all SVG files from a specified directory

Request body: directory (string, default: "svg_files/")
Returns: List of processed files
POST /predict: Find similar images for an uploaded SVG

Request body: SVG file upload
Returns: Match status and similarity score
GET /status: Get current system status

Returns: Number of vectorized images and index size
API Examples
Vectorize Local Files
curl -X POST "http://127.0.0.1:8000/vectorize-local" -H "Content-Type: application/json" -d '{"directory": "svg_files/"}'

Predict Similar Images
curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@your_svg_file.svg"

Directory Structure
project/
├── app.py
├── README.md
├── requirements.txt
└── svg_files/
    └── your_svg_files.svg

Technical Details
Model: ResNet50 (pretrained)
Feature Dimension: 2048
Similarity Search: FAISS L2 distance
Image Processing: svglib for SVG conversion
Response Formats
Prediction Response
{
    "match_found": true,
    "matched_image": "path/to/matched/image.svg",
    "similarity_score": 42.5
}



Status Response
{
    "vectorized_images": 10,
    "index_size": 10,
    "images": ["image1.svg", "image2.svg"]
}



Performance Considerations
Similarity threshold is set to 100 (adjustable)
FAISS index is kept in memory for fast searching
Batch processing available for local files
Requirements
Python 3.8+
FastAPI
PyTorch
FAISS
svglib
uvicorn
