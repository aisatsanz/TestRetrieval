import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from io import BytesIO
import faulthandler, signal, os
faulthandler.enable() 
from pathlib import Path
from PIL import Image
import faiss, torch, numpy as np

from .registry   import load_model
from .utils import build_tf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
APP_DIR   = Path(__file__).resolve().parent          # …/app
ROOT_DIR  = APP_DIR.parent                           # …/point3ds
MODELS_DIR = ROOT_DIR / "models"

app = FastAPI(title="Flower-Retrieval API", version="1.0.0")


@app.post("/predict")
async def predict(
    file : UploadFile = File(...),
    model: str        = Query("resnet50",
                              description="имя папки в /models/*")
):
    model_dir = MODELS_DIR / model
    if not model_dir.is_dir():
        raise HTTPException(404, f"unknown model '{model}'")

    extractor, index, paths, meta = load_model(model)
    tf = build_tf(meta)

    img = Image.open(BytesIO(await file.read())).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(DEVICE)

    try:
        with torch.no_grad():
            vec = extractor.encode(tensor)           # (1, D)
        faiss.normalize_L2(vec)
        D, I = index.search(vec, 5 + 5)  # чуть больше, чтобы после фильтрации осталось 5

        query_name = Path(file.filename).name
        results = []
        for rank, idx in enumerate(I[0]):
            cand_name = Path(paths[idx]).name
            if cand_name == query_name:
                continue
            results.append({
                "image_path": str(paths[idx]),
                "similarity": float(D[0][rank])
            })
            if len(results) == 5:
                break

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"faiss error: {e}")

    return JSONResponse({"model": model, "results": results})


@app.get("/models")
def list_models():
    return {"models": sorted([p.name for p in MODELS_DIR.iterdir() if p.is_dir()])}


if __name__ == "__main__":
    import uvicorn, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)

