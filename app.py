import streamlit as st
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle

# -----------------------
# Helper Functions
# -----------------------

# Extract simple color histogram features from image
def extract_features(img):
    img = img.resize((100, 100))  # resize for consistency
    img_array = np.array(img)
    hist = []
    for i in range(3):  # R, G, B channels
        channel_hist = np.histogram(img_array[:, :, i], bins=16, range=(0, 256))[0]
        hist.extend(channel_hist)
    hist = np.array(hist) / np.sum(hist)  # normalize
    return hist

# Train a simple KNN classifier (or load pre-trained)
@st.cache_resource
def load_model():
    model_file = "knn_model.pkl"
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            knn = pickle.load(f)
    else:
        # If no model exists, train a dummy classifier (for demo)
        X_dummy = np.random.rand(10, 48)  # 10 random feature vectors
        y_dummy = ["Plastic", "Organic", "Paper", "Metal", "E-waste"] * 2
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_dummy, y_dummy)
        with open(model_file, "wb") as f:
            pickle.dump(knn, f)
    return knn

# Predict waste type
def predict_waste(img, model):
    features = extract_features(img)
    pred = model.predict([features])
    return pred[0]

# -----------------------
# Streamlit App
# -----------------------

st.set_page_config(page_title="Waste Classifier (Lightweight)", page_icon="♻️")
st.title("♻️ Waste Classifier (Lightweight)")
st.write("Upload an image of waste and get sustainability tips!")

# Load lightweight model
model = load_model()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    pred_class = predict_waste(img, model)
    st.success(f"Predicted Waste Type: **{pred_class}**")

    # Recycling tips
    tips = {
        "Plastic": "Recycle plastics and avoid single-use plastic.",
        "Organic": "Compost organic waste to make nutrient-rich soil.",
        "Paper": "Reuse or recycle paper products.",
        "Metal": "Collect and recycle metals at proper centers.",
        "E-waste": "Take to e-waste recycling centers safely.",
        "Unknown": "Try uploading a clearer image."
    }
    st.info(tips.get(pred_class, "Unknown"))

st.markdown("---")
st.subheader("🌍 SDG Impact")
st.markdown("""
This project contributes to:
- **SDG 12:** Responsible Consumption and Production  
- **SDG 11:** Sustainable Cities and Communities  
- **SDG 13:** Climate Action
""")
