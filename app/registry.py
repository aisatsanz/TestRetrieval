import yaml, faiss, torch, numpy as np, pathlib
from src.extractors import ResNetExtractor, EfficientNetExtractor
from src.extractors import MetricExtractor, 

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACTOR_MAP = {
    "resnet50":        ResNetExtractor,
    "efficientnet_b0": EfficientNetExtractor,
    "metric":          MetricExtractor,
    "clip":         MetricExtractor,
    "dinov2":          None,
}

def load_model(name: str):
    """возвращает (extractor, index, paths, meta)"""
    model_dir = ROOT / "models" / name
    with open(model_dir / "meta.yaml") as f:
        meta = yaml.safe_load(f)

    ext_cls = EXTRACTOR_MAP[name]
    extractor = ext_cls()
    weights = model_dir / "weights.pth"
    if weights.exists():
        extractor.load_state_dict(torch.load(weights, map_location=DEVICE))
    extractor.eval().to(DEVICE)

    index = faiss.read_index(str(model_dir / "index.faiss"))
    paths = np.load(model_dir / "paths.npy", allow_pickle=True)

    return extractor, index, paths, meta