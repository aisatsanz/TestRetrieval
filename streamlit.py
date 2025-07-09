import streamlit as st
from PIL import Image
from io import BytesIO
import requests, pathlib


DEFAULT_API = "http://localhost:8000"
MODELS_DIR  = pathlib.Path(__file__).parents[0] / "models"
model_names = sorted([p.name for p in MODELS_DIR.iterdir() if p.is_dir()])

st.sidebar.title("Settings")
api_url     = st.sidebar.text_input("API base URL", DEFAULT_API)
model_choice= st.sidebar.selectbox("Model", model_names, index=0)
topk        = st.sidebar.slider("Top-K", 1, 10, 5)

st.title("🌼 Flower Retrieval (via REST API)")
upload = st.file_uploader("Choose an image", type=["jpg","png","jpeg"])

if upload:
    files = {"file": (upload.name, upload.getvalue())}
    params = {"model": model_choice}
    try:
        resp = requests.post(f"{api_url}/predict", params=params, files=files, timeout=30)
        resp.raise_for_status()
        data = resp.json()["results"][:topk]      # [{image_path, similarity}]
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    cols = st.columns(topk + 1)
    cols[0].image(upload.getvalue(), caption="query", use_column_width=True)

    for i, res in enumerate(data, start=1):
        img_path = pathlib.Path(MODELS_DIR.parent) / res["image_path"]
        if not img_path.exists():
            cols[i].write("no image")
            continue
        cols[i].image(img_path, caption=f'{res["similarity"]:.3f}', use_column_width=True)