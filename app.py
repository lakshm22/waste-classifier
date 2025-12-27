import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# -----------------------
# Helper Functions
# -----------------------

# Load pre-trained MobileNetV2
@st.cache_resource  # Cache the model so it loads only once
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

# Preprocess image for prediction
def preprocess_img(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Predict waste type
def predict_waste(img, model):
    processed = preprocess_img(img)
    preds = model.predict(processed)
    decoded = decode_predictions(preds, top=3)[0]

    # Map general ImageNet classes to simple waste types
    for _, name, prob in decoded:
        if name in ["banana", "orange", "lemon", "potato"]:
            return "Organic"
        elif name in ["plastic_bag", "bottle", "cup", "container"]:
            return "Plastic"
        elif name in ["envelope", "book", "paper_towel"]:
            return "Paper"
        elif name in ["computer_keyboard", "monitor", "cellular_telephone"]:
            return "E-waste"
        elif name in ["screwdriver", "hammer", "spoon"]:
            return "Metal"
    return "Unknown"

# -----------------------
# Streamlit App
# -----------------------

st.set_page_config(page_title="Waste Classifier", page_icon="♻️")
st.title("♻️ AI Waste Classification Tool")
st.write("Upload an image of waste and get sustainability tips!")

# Load the model
model = load_model()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    pred_class = predict_waste(img, model)
    st.success(f"Predicted Waste Type: **{pred_class}**")

    # Tips for each waste type
    tips = {
        "Plastic": "Recycle plastics and avoid single-use plastic.",
        "Organic": "Compost organic waste to make nutrient-rich soil.",
        "Paper": "Reuse or recycle paper products.",
        "Metal": "Collect and recycle metals at proper centers.",
        "E-waste": "Take to e-waste recycling centers safely.",
        "Unknown": "Try uploading a clearer image."
    }
    st.info(tips[pred_class])

# Optional SDG impact section
st.markdown("---")
st.subheader("🌍 SDG Impact")
st.markdown("""
This project contributes to:
- **SDG 12:** Responsible Consumption and Production  
- **SDG 11:** Sustainable Cities and Communities  
- **SDG 13:** Climate Action
""")
