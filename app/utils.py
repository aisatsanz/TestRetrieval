from torchvision import transforms
import numpy as np
import faiss
import yaml
from types import SimpleNamespace

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    def to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
        return d

    return to_namespace(raw_cfg)


def build_tf(meta):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std  = (0.26862954, 0.26130258, 0.27577711)
    size      = meta.get("input_size", 224)
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def build_index(vecs: np.ndarray, metric: str = "cosine") -> faiss.Index:
    """
    vecs – (N,D) float32,
    metric: 'cosine' | 'l2'
    """
    vecs = np.ascontiguousarray(vecs, dtype="float32")

    if metric == "cosine":
        faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(vecs.shape[1])
    elif metric == "l2":
        index = faiss.IndexFlatL2(vecs.shape[1])
    else:
        raise ValueError("метрика должна быть 'cosine' or 'l2'")

    index.add(vecs)
    return index

