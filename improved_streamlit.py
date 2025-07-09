import streamlit as st
from PIL import Image
from io import BytesIO
import requests, pathlib
import os

DEFAULT_API = os.getenv("API_URL", "http://localhost:8080")
ROOT_DIR    = pathlib.Path(__file__).resolve().parent
MODELS_DIR  = ROOT_DIR / "models"

st.set_page_config(
    page_title="🌼 Flower Retrieval System",
    page_icon="🌼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌼 Flower Retrieval System")
st.markdown("**Система поиска похожих изображений цветов на основе машинного обучения**")
st.sidebar.title("Настройки")
api_url = st.sidebar.text_input("API URL", DEFAULT_API)


@st.cache_data(show_spinner=False)
def fetch_models(api: str):
    try:
        r = requests.get(f"{api.rstrip('/')}/models", timeout=5)
        r.raise_for_status()
        return r.json().get("models", [])
    except Exception:
        return []

model_list = fetch_models(api_url)
if not model_list:  
    model_list = sorted([p.name for p in MODELS_DIR.iterdir() if p.is_dir()])

if not model_list:
    st.sidebar.error("Модели не найдены"); st.stop()

model_choice = st.sidebar.selectbox("Модель", model_list, index=0)

with st.sidebar.expander("ℹО системе"):
    st.markdown(
        """
        - Загрузите изображение цветка  
        - Система найдёт наиболее похожие изображения  
        - Результаты отсортированы по убыванию схожести  
        **Форматы:** JPG · PNG · JPEG
        """
    )

col_query, col_results = st.columns([1, 2])

with col_query:
    st.subheader("📤 Загрузка изображения")
    upload = st.file_uploader(
        "Выберите изображение цветка",
        type=["jpg", "jpeg", "png"],
        help="Поддерживаемые форматы: JPG, PNG, JPEG"
    )
    if upload:
        st.image(upload, caption="Загруженное изображение",
                 use_container_width=True)
        st.info(f"**Файл:** {upload.name}\n**Размер:** {len(upload.getvalue())} байт")

with col_results:
    st.subheader("🔍 Результаты поиска")

    if upload and st.button("Найти похожие изображения"):
        with st.spinner("Поиск похожих изображений…"):
            try:
                files  = {"file": (upload.name, upload.getvalue(), "image/jpeg")}
                params = {"model": model_choice}

                resp = requests.post(f"{api_url.rstrip('/')}/predict",
                                     params=params, files=files, timeout=30)

                if resp.status_code != 200:
                    st.error(f"API error {resp.status_code}: {resp.text}")
                    st.stop()

                results = resp.json().get("results", [])[:5]

            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка запроса к API: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Произошла ошибка: {e}")
                st.stop()

            if not results:
                st.warning("!!Результаты не найдены")
                st.stop()

            st.success(f"✅ Найдено {len(results)} похожих изображений")
            grid_cols = st.columns(min(3, len(results)))

            for i, res in enumerate(results):
                c      = grid_cols[i % len(grid_cols)]
                sim    = res.get("similarity", res.get("similarity_score", 0))
                img_p  = (
                    ROOT_DIR / res["image_path"]
                    if (ROOT_DIR / res["image_path"]).exists()
                    else pathlib.Path(res["image_path"]).resolve()
                )
                if img_p.exists():
                    c.image(str(img_p), caption=f"{sim:.3f}",
                            use_container_width=True)
                else:
                    c.warning("🖼️ not found")
                    c.caption(res["image_path"])

            with st.expander("Детальные результаты"):
                st.json(resp.json())
    elif not upload:
        st.info("👆 Сначала загрузите изображение.")


st.markdown("---")
st.markdown("**Тестовое задание ML-инженера** – Image Retrieval for Flowers")
