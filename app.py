import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.dataset import CLASS_NAMES, normalize_image
from src.models import build_model
from src.predict import colorize


st.set_page_config(page_title="Terrain Segmentation", layout="wide")
st.title("🌍 Terrain & Vegetation Segmentation")
st.caption("U-Net / DeepLabV3 semantic segmentation for remote-sensing land-cover mapping.")

checkpoint_options = list(Path("outputs/checkpoints").glob("*.pt"))

if not checkpoint_options:
    st.warning(
        "No trained checkpoint found. First run create_demo_dataset.py and train the model."
    )
    st.stop()

checkpoint = st.sidebar.selectbox(
    "Checkpoint",
    checkpoint_options,
    format_func=lambda x: x.name
)

uploaded = st.file_uploader(
    "Upload a satellite / aerial RGB image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)

    model = build_model(ckpt["model"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tensor = normalize_image(image, ckpt.get("image_size", 256))
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, dict):
            output = output["out"]
        pred = torch.argmax(output, dim=1)[0].cpu().numpy()

    pred_img = Image.fromarray(pred.astype(np.uint8), "L").resize(
        image.size, Image.Resampling.NEAREST
    )
    colored = Image.fromarray(colorize(np.asarray(pred_img)), "RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Segmentation")
        st.image(colored, use_container_width=True)

    st.subheader("Legend")
    cols = st.columns(len(CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        swatch = Image.new("RGB", (30, 30), tuple(colorize(np.array([[i]]))[0, 0]))
        cols[i].image(swatch)
        cols[i].caption(f"{i}: {name}")
