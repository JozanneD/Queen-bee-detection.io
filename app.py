import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
model = YOLO("bee.pt")

st.title("🐝 Queen Bee Detection using YOLOv8")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image with PIL and convert to numpy array
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Run YOLO prediction
    results = model(image_np)

    # Plot with Ultralytics' built-in method (returns PIL Image)
    annotated_image = results[0].plot(pil=True)

    # Resize the image (optional)
    new_width = int(annotated_image.width * 0.4)
    new_height = int(annotated_image.height * 0.4)
    resized_image = annotated_image.resize((new_width, new_height))

    # Display in Streamlit
    st.image(resized_image, caption="Detected Image", use_column_width=True)
