import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

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
# Sidebar
# -----------------------
st.sidebar.title("Waste Classifier ♻️")
with st.sidebar.expander("Project Description"):
    st.write(
        "An AI-powered app built with Python and Streamlit that classifies waste types from images "
        "and provides quick recycling tips to promote sustainable practices."
    )

with st.sidebar.expander("Tools Used"):
    st.write("- Python  \n- Streamlit  \n- TensorFlow / Keras  \n- Pillow  \n- NumPy")

with st.sidebar.expander("SDG Impact 🌍"):
    st.write(
        "- SDG 12: Responsible Consumption and Production  \n"
        "- SDG 11: Sustainable Cities and Communities  \n"
        "- SDG 13: Climate Action"
    )

# -----------------------
# Main App
# -----------------------
st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="wide")
st.title("♻️ AI Waste Classification Tool")
st.markdown("Upload an image of waste and see instant classification with recycling tips!")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    # Layout: Columns
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.write("Classifying...")
        pred_class = predict_waste(img, model)

        # Emoji mapping
        emoji_map = {
            "Plastic": "🧴",
            "Organic": "🍌",
            "Paper": "📄",
            "Metal": "🔩",
            "E-waste": "💻",
            "Unknown": "❓"
        }

        st.markdown(f"<h2 style='color:green'>{emoji_map.get(pred_class,'❓')} {pred_class}</h2>", unsafe_allow_html=True)

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

        # Collapsible Hazards Section
        with st.expander("⚠️ Hazards of Improper Waste Disposal"):
            st.markdown("""
            ### Health Hazards
            **Plastic:** Releases toxic chemicals into soil and water; blocks drainage; microplastics enter food chain.  
            **Organic Waste:** Produces foul odors and methane; attracts rodents and insects, spreading diseases.  
            **Paper Waste:** Can accumulate and become a fire hazard.  
            **Metal Waste:** Rust contaminates water; sharp edges can cause injuries.  
            **E-waste:** Contains heavy metals like lead and mercury; can damage kidneys, liver, nervous system, and cause developmental issues in children.

            ### Environmental Hazards
            - Soil and water contamination from plastics, metals, and chemicals.  
            - Accumulation of non-biodegradable waste in landfills.  
            - Harm to wildlife due to ingestion or entanglement.  

            ### Climatic Hazards
            - Methane emissions from decomposing organic waste contribute to global warming.  
            - Mismanaged waste can worsen urban flooding due to blocked drains.  
            - Incineration of waste produces greenhouse gases and particulate matter affecting climate and air quality.
            """)
