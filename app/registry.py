import yaml, faiss, torch, numpy as np, pathlib
from src.extractors import ResNetExtractor, EfficientNetExtractor, \
    FastMetricExtractor, DINOv2Extractor
import json
from pathlib import Path
import torch.nn.functional as F

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT        = Path(__file__).resolve().parents[1]      # …/point3ds
MODELS_DIR  = ROOT / "models"

EXTRACTOR_MAP = {
    "resnet50":        ResNetExtractor,
    "efficientnet_b0": EfficientNetExtractor,
    "metric":          FastMetricExtractor,
    "dinov2":          DINOv2Extractor,
}


def _first(glob_iter):
    files = list(glob_iter)
    if not files:
        raise FileNotFoundError(f"File not found for pattern: {glob_iter}")
    return files[0]

def load_model(name: str):
    model_dir = MODELS_DIR / name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Модель '{name}' не найдена: {model_dir}")

    idx_file = _first(model_dir.glob("*.faiss"))
    index    = faiss.read_index(str(idx_file))

    paths_file = _first(model_dir.glob("*paths.npy"))
    paths      = np.load(paths_file, allow_pickle=True)


    meta_files = list(model_dir.glob("*meta.*"))
    if meta_files:
        mf   = meta_files[0]
        meta = yaml.safe_load(mf.read_text()) if mf.suffix==".yaml" \
               else json.loads(mf.read_text())
    else:
        meta = {"metric":"cosine","input_size":224}
    ext_cls      = EXTRACTOR_MAP[name]
    extractor    = ext_cls()
    weight_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
    if weight_files and isinstance(extractor, torch.nn.Module):
        state_dict = torch.load(weight_files[0], map_location="cpu")
        extractor.load_state_dict(state_dict, strict=False)

    if hasattr(extractor, "eval"):
        extractor.eval()
    if hasattr(extractor, "to"):
        extractor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if not hasattr(extractor, "encode"):
        def encode_fn(x):
            extractor.eval()
            with torch.no_grad():
                out = extractor(x.to(DEVICE))
                if hasattr(extractor, "embed"):
                    emb = extractor.embed(out)
                else:
                    emb = out
                emb = F.normalize(emb, dim=-1)
            return emb.cpu().numpy().astype("float32")
        extractor.encode = encode_fn

    return extractor, index, paths, meta