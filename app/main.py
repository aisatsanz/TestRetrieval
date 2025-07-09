from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import faiss, numpy as np, torch
from .registry import load_model
from .transforms import build_tf

app = FastAPI(title="Image-Retrieval API", version="1.0.0")
CACHE = {}   # model_name → (extractor, index, paths, tf)

def get_pipeline(name: str):
    if name not in CACHE:
        CACHE[name] = load_model(name)
        extractor, _, _, meta = CACHE[name]
        CACHE[name] += (build_tf(meta),)            # append transform
    return CACHE[name]

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("resnet50", description="one of /models/*")
):
    try:
        extractor, index, paths, tf = get_pipeline(model)
    except KeyError:
        raise HTTPException(404, f"unknown model '{model}'")

    img = Image.open(BytesIO(await file.read())).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(extractor.device)

    with torch.no_grad():
        vec = extractor.encode(tensor)              # (1,D)
    faiss.normalize_L2(vec)
    D, I = index.search(vec, 5)

    results = [
        {"image_path": str(paths[i]), "similarity": float(D[0][rank])}
        for rank, i in enumerate(I[0])
    ]
    return JSONResponse({"model": model, "results": results})
