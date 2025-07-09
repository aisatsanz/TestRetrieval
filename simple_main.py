from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import faiss
import numpy as np
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Image-Retrieval API", version="1.0.0")

# Простая загрузка данных без сложных экстракторов
def load_simple_model(name: str = "resnet50"):
    """Загружает индекс и пути для модели"""
    model_dir = BASE_DIR / "models" / name
    
    # Загружаем индекс
    index_file = model_dir / f"{name}.faiss"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    index = faiss.read_index(str(index_file))
    
    # Загружаем пути
    paths_file = model_dir / f"{name}_paths.npy"
    if not paths_file.exists():
        raise FileNotFoundError(f"Paths file not found: {paths_file}")
    
    paths = np.load(paths_file, allow_pickle=True)
    
    return index, paths

@app.get("/")
async def root():
    return {"message": "Image Retrieval API is running"}

@app.get("/models")
async def list_models():
    """Список доступных моделей"""
    models_dir = BASE_DIR / "models"
    if not models_dir.exists():
        return {"models": []}
    
    models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            faiss_file = model_dir / f"{model_dir.name}.faiss"
            paths_file = model_dir / f"{model_dir.name}_paths.npy"
            if faiss_file.exists() and paths_file.exists():
                models.append(model_dir.name)
    
    return {"models": models}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("resnet50", description="Model name")
):
    try:
        # Загружаем модель
        index, paths = load_simple_model(model)
        
        # Читаем изображение
        img_bytes = await file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        # Для демонстрации возвращаем случайные результаты
        # В реальной системе здесь должно быть извлечение признаков
        np.random.seed(42)  # для воспроизводимости
        random_indices = np.random.choice(len(paths), size=min(5, len(paths)), replace=False)
        random_scores = np.random.uniform(0.7, 0.95, size=len(random_indices))
        
        # Сортируем по убыванию схожести
        sorted_indices = np.argsort(random_scores)[::-1]
        
        results = []
        for i, idx in enumerate(sorted_indices):
            path_idx = random_indices[idx]
            results.append({
                "image_path": str(paths[path_idx]),
                "similarity_score": float(random_scores[idx])
            })
        
        return JSONResponse({
            "model": model,
            "input_image": file.filename,
            "results": results
        })
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
