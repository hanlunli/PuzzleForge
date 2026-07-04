import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jigsaw_generator import generate_puzzle_pieces
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/pieces", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/generate")
async def generate_puzzle(
    image: UploadFile = File(...),
    rows: int = Form(4),
    cols: int = Form(4)
):
    # Save uploaded image
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(image.filename)[1]
    if not ext:
        ext = ".png"
    image_path = f"uploads/{file_id}{ext}"
    
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    # Generate puzzle pieces
    pieces, width, height = generate_puzzle_pieces(image_path, rows, cols)
    
    if not pieces:
        return JSONResponse(status_code=500, content={"error": "Failed to generate puzzle"})
        
    # Save individual pieces and prepare response
    piece_data = []
    
    # Canvas dimensions for scattering
    canvas_w = width * 3
    canvas_h = height * 3
    
    for i, (piece_img, (ox, oy)) in enumerate(pieces):
        piece_filename = f"{file_id}_piece_{i}.png"
        piece_path = os.path.join("static", "pieces", piece_filename)
        piece_img.save(piece_path)
        
        # Random initial position and rotation for the frontend
        # We'll let the frontend handle the exact scattering logic to avoid overlaps if needed,
        # but we can provide random coordinates here.
        rx = random.randint(0, max(0, canvas_w - piece_img.width))
        ry = random.randint(0, max(0, canvas_h - piece_img.height))
        angle = random.uniform(0, 360)
        
        piece_data.append({
            "id": i,
            "url": f"/static/pieces/{piece_filename}",
            "original_x": ox,
            "original_y": oy,
            "scatter_x": rx,
            "scatter_y": ry,
            "scatter_angle": angle,
            "width": piece_img.width,
            "height": piece_img.height
        })
        
    return {
        "pieces": piece_data,
        "original_width": width,
        "original_height": height,
        "canvas_width": canvas_w,
        "canvas_height": canvas_h
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
