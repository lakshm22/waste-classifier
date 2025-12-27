import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import pandas as pd

# -----------------------
# Helper Functions
# -----------------------

@st.cache_resource
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_img(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_waste(img, model):
    processed = preprocess_img(img)
    preds = model.predict(processed)
    decoded = decode_predictions(preds, top=3)[0]

    for _, name, prob in decoded:
        if name in ["banana", "orange", "lemon", "potato", "apple"]:
            return "Organic"
        elif name in ["plastic_bag", "bottle", "cup", "container"]:
            return "Plastic"
        elif name in ["envelope", "book", "paper_towel", "newspaper"]:
            return "Paper"
        elif name in ["computer_keyboard", "monitor", "cellular_telephone", "laptop"]:
            return "E-waste"
        elif name in ["screwdriver", "hammer", "spoon", "fork"]:
            return "Metal"
    return "Unknown"

# -----------------------
# Initialize Session State
# -----------------------
if 'counts' not in st.session_state:
    st.session_state.counts = {
        "Plastic": 0,
        "Organic": 0,
        "Paper": 0,
        "Metal": 0,
        "E-waste": 0,
        "Unknown": 0
    }

# -----------------------
# Streamlit App Layout
# -----------------------
st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="wide")

# Sidebar
with st.sidebar:
    st.image("assets/logo.png", use_column_width=True)  # optional
    st.title("Waste Classifier")
    st.markdown("**AI + Sustainable Development** project")
    st.subheader("Project Intro")
    st.write("Upload images of waste and classify them into categories like Plastic, Organic, Paper, Metal, or E-waste. Get quick tips on recycling and disposal while contributing to sustainability goals.")
    st.subheader("Tools Used")
    st.write("- Python\n- Streamlit\n- TensorFlow / Keras\n- Pillow\n- NumPy")
    st.subheader("SDG Goals")
    st.write("""
    - **SDG 12:** Responsible Consumption and Production  
    - **SDG 11:** Sustainable Cities and Communities  
    - **SDG 13:** Climate Action
    """)
    st.markdown("---")
    
    # Mini Dashboard: Show counts
    st.subheader("📊 Waste Classification Stats")
    counts_df = pd.DataFrame.from_dict(st.session_state.counts, orient='index', columns=['Count'])
    st.bar_chart(counts_df)

# Main app
st.title("♻️ Waste Classification Tool")
st.write("Upload an image of waste to see it classified and get sustainability tips!")

model = load_model()
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"], label_visibility="visible")
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        pred_class = predict_waste(img, model)
    
    # Update session state counts
    st.session_state.counts[pred_class] += 1

    # Colored result box
    color_map = {
        "Plastic": "#FAD02E",
        "Organic": "#76B041",
        "Paper": "#7FC6A4",
        "Metal": "#A0AEC0",
        "E-waste": "#FF6B6B",
        "Unknown": "#B0B0B0"
    }
    st.markdown(
        f"<div style='padding: 15px; border-radius: 10px; background-color: {color_map.get(pred_class, '#B0B0B0')}; font-size:20px; font-weight:bold; text-align:center;'>Predicted Waste Type: {pred_class}</div>",
        unsafe_allow_html=True
    )

    # Tips
    tips = {
        "Plastic": "Recycle plastics and avoid single-use plastic.",
        "Organic": "Compost organic waste to make nutrient-rich soil.",
        "Paper": "Reuse or recycle paper products.",
        "Metal": "Collect and recycle metals at proper centers.",
        "E-waste": "Take to e-waste recycling centers safely.",
        "Unknown": "Try uploading a clearer image."
    }
    st.info(tips[pred_class])

# Mini Dashboard: Show counts
st.subheader("📊 Waste Classification Stats")
counts_df = pd.DataFrame.from_dict(st.session_state.counts, orient='index', columns=['Count'])
st.bar_chart(counts_df)

# Reset button
if st.button("🔄 Reset Stats"):
    for key in st.session_state.counts:
        st.session_state.counts[key] = 0
    st.experimental_rerun()  # Refresh app to update chart

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>🌍 Promoting sustainability through AI | Developed with Streamlit</p>", unsafe_allow_html=True)
