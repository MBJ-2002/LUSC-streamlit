import streamlit as st
import numpy as np
import time
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Class names
class_names = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark", 
    "overpass", "parkinglot", "river", "runway", "sparseresidential", 
    "storagetanks", "tenniscourt"
]

# Load model only once
@st.cache_resource
def load_cached_model():
    return load_model("models/Xception.keras")

def preprocess_image(img: Image.Image, img_size=128):
    img = img.resize((img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

st.set_page_config(layout="wide")
# Title
st.title("Land Usage Classification using Xception")

# Layout: Image & Upload (Left), Metrics (Right)
left_col, right_col = st.columns([3, 2])  # Wider left panel

with left_col:
    uploaded_img = st.file_uploader("", label_visibility="collapsed", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)

        # Run inference
        model = load_cached_model()
        input_data = preprocess_image(img)
        
        with st.spinner("🔍 Making prediction..."):
            start_time = time.time()
            predictions = model.predict(input_data)
            inference_time = time.time() - start_time
        
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence_score = predictions[0][predicted_index]

        # Probability plot (below everything)
        st.markdown("### Top 5 Predictions")
        probs = pd.Series(predictions[0], index=class_names).sort_values(ascending=False)
        st.bar_chart(probs.head(5), use_container_width=True)


with right_col:
    st.markdown("### Prediction Metrics")
    if uploaded_img:
        col1, col2, col3 = st.columns(3)
        col1.metric("Class", predicted_class)
        col2.metric("Confidence", f"{confidence_score:.2f}")
        col3.metric("Time", f"{inference_time*1000:.1f} ms")
    else:
        st.info("Upload an image to see prediction results.")
