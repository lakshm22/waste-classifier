import streamlit as st
from PIL import Image
from utils.helpers import load_model, load_labels, predict

st.set_page_config(page_title="Waste Classifier", page_icon="♻️")

st.title("♻️ AI Waste Classification Tool")
st.write("Upload an image of waste and get recycling tips!")

# Load model and labels
model = load_model()
labels = load_labels()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    prediction = predict(img, model, labels)
    st.success(f"Predicted Waste Type: **{prediction}**")

    # Tips for each waste type
    tips = {
        "Plastic": "Recycle plastics and avoid single-use plastic.",
        "Organic": "Compost organic waste to make nutrient-rich soil.",
        "Paper": "Reuse or recycle paper products.",
        "Metal": "Collect and recycle metals at proper centers.",
        "E-waste": "Take to e-waste recycling centers safely."
    }
    st.info(tips.get(prediction, "No tips available"))
