import yaml, faiss, torch, numpy as np, pathlib
from src.extractors import ResNetExtractor, EfficientNetExtractor, \
    MetricExtractor, DINOv2Extractor
import json
from pathlib import Path
from src.mixin import FineTuneMixin, FeatureExtractor

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT        = Path(__file__).resolve().parents[1]      # …/point3ds
MODELS_DIR  = ROOT / "models"

EXTRACTOR_MAP = {
    "resnet50":        ResNetExtractor,
    "efficientnet_b0": EfficientNetExtractor,
    "metric":          MetricExtractor,
    "dinov2":          DINOv2Extractor,
}

def _first(glob_iter):
    files = list(glob_iter)
    if not files:
        raise FileNotFoundError(f"File not found for pattern: {glob_iter}")
    return files[0]

# ------------------------------------------------------------------
def load_model(name: str):
    """
    Возвращает: extractor, faiss_index, paths (np.ndarray[str]), meta (dict)
    Допустимые файлы в models/<name>/ :
        <name>.faiss               – индекс (обязательно)
        <name>_paths.npy           – пути к картинкам (обязательно)
        <name>_meta.yaml|json      – метаданные (не обязательно)
        <name>*.pth / weights.pth  – state-dict экстрактора (не обязательно)
    """
    model_dir = MODELS_DIR / name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Модель '{name}' не найдена: {model_dir}")

    # ---------- FAISS индекс --------------------------------------
    idx_path = _first(model_dir.glob("*.faiss"))
    index = faiss.read_index(str(idx_path))

    # ---------- paths.npy -----------------------------------------
    paths_path = _first(model_dir.glob("*paths.npy"))
    paths = np.load(paths_path, allow_pickle=True)

    # ---------- meta ----------------------------------------------
    meta_files = list(model_dir.glob("*meta.*"))
    if meta_files:
        meta_file = meta_files[0]
        if meta_file.suffix == ".yaml":
            meta = yaml.safe_load(meta_file.read_text())
        else:
            meta = json.loads(meta_file.read_text())
    else:
        meta = {"metric": "cosine", "input_size": 224}

    # ---------- extractor -----------------------------------------
    ext_cls = EXTRACTOR_MAP[name]
    extractor = ext_cls()
    weight_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
    if weight_files and isinstance(extractor, torch.nn.Module):
       state_dict = torch.load(weight_files[0], map_location="cpu")
       _ = extractor.load_state_dict(state_dict, strict=False)  # ignore missing keys


    if hasattr(extractor, "eval"):
        extractor.eval()
    if hasattr(extractor, "to"):
        extractor.to(DEVICE)

    return extractor, index, paths, meta