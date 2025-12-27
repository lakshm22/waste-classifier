import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# -----------------------
# Helper Functions
# -----------------------

# Load pre-trained ResNet18
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()  # set to evaluation mode
    return model

# Image preprocessing
def preprocess_img(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return preprocess(img).unsqueeze(0)  # add batch dimension

# Map ImageNet classes to waste types
imagenet_to_waste = {
    "banana": "Organic",
    "orange": "Organic",
    "lemon": "Organic",
    "potato": "Organic",
    "plastic_bag": "Plastic",
    "bottle": "Plastic",
    "cup": "Plastic",
    "container": "Plastic",
    "envelope": "Paper",
    "book": "Paper",
    "paper_towel": "Paper",
    "computer_keyboard": "E-waste",
    "monitor": "E-waste",
    "cellular_telephone": "E-waste",
    "screwdriver": "Metal",
    "hammer": "Metal",
    "spoon": "Metal"
}

# Load ImageNet labels
@st.cache_resource
def load_labels():
    import json, urllib.request
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = []
    with urllib.request.urlopen(url) as f:
        for line in f:
            labels.append(line.decode("utf-8").strip())
    return labels

# Predict waste type
def predict_waste(img, model, labels):
    input_tensor = preprocess_img(img)
    with torch.no_grad():
        outputs = model(input_tensor)
    _, predicted = outputs.max(1)
    imagenet_class = labels[predicted.item()]
    return imagenet_to_waste.get(imagenet_class, "Unknown")

# -----------------------
# Streamlit App
# -----------------------

st.set_page_config(page_title="Waste Classifier", page_icon="♻️")
st.title("♻️ AI Waste Classification Tool (PyTorch)")
st.write("Upload an image of waste and get sustainability tips!")

# Load model and labels
model = load_model()
labels = load_labels()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    pred_class = predict_waste(img, model, labels)
    st.success(f"Predicted Waste Type: **{pred_class}**")

    tips = {
        "Plastic": "Recycle plastics and avoid single-use plastic.",
        "Organic": "Compost organic waste to make nutrient-rich soil.",
        "Paper": "Reuse or recycle paper products.",
        "Metal": "Collect and recycle metals at proper centers.",
        "E-waste": "Take to e-waste recycling centers safely.",
        "Unknown": "Try uploading a clearer image."
    }
    st.info(tips[pred_class])

st.markdown("---")
st.subheader("🌍 SDG Impact")
st.markdown("""
This project contributes to:
- **SDG 12:** Responsible Consumption and Production  
- **SDG 11:** Sustainable Cities and Communities  
- **SDG 13:** Climate Action
""")
