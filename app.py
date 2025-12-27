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

        # Display prediction with emoji inline
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

        # Hazards by waste type
        hazards = {
            "Plastic": {
                "Health": "Releases toxic chemicals into soil and water; blocks drainage; microplastics enter food chain.",
                "Environmental": "Soil and water contamination; harm to wildlife due to ingestion or entanglement.",
                "Climatic": "Production and incineration release greenhouse gases; contributes to global warming."
            },
            "Organic": {
                "Health": "Produces foul odors and methane; attracts rodents and insects, spreading diseases.",
                "Environmental": "Improper disposal pollutes soil and water; attracts pests.",
                "Climatic": "Methane emissions from decomposition; contributes to global warming."
            },
            "Paper": {
                "Health": "Can accumulate and become a fire hazard.",
                "Environmental": "Waste buildup can block drainage and pollute soil.",
                "Climatic": "Decomposition produces small amounts of methane."
            },
            "Metal": {
                "Health": "Rust contaminates water; sharp edges can cause injuries.",
                "Environmental": "Toxic metals can leach into soil and water.",
                "Climatic": "Energy-intensive production and improper disposal contribute to greenhouse gases."
            },
            "E-waste": {
                "Health": "Contains heavy metals like lead and mercury; can damage kidneys, liver, nervous system, and affect children.",
                "Environmental": "Leaches toxins into soil and water; harms wildlife.",
                "Climatic": "Improper incineration releases greenhouse gases and toxic fumes."
            },
            "Unknown": {
                "Health": "Unknown hazards. Try uploading a clearer image.",
                "Environmental": "Unknown hazards.",
                "Climatic": "Unknown hazards."
            }
        }

        # Display only hazards relevant to predicted waste type in colored cards
        with st.expander(f"⚠️ Hazards of {pred_class} Waste"):
            col1, col2, col3 = st.columns(3)

            col1.markdown(f"<div style='background-color:#ffcccc; padding:10px; border-radius:10px;'>"
                          f"<h4>💊 Health</h4>"
                          f"<p>{hazards[pred_class]['Health']}</p></div>", unsafe_allow_html=True)

            col2.markdown(f"<div style='background-color:#cce5ff; padding:10px; border-radius:10px;'>"
                          f"<h4>🌱 Environmental</h4>"
                          f"<p>{hazards[pred_class]['Environmental']}</p></div>", unsafe_allow_html=True)

            col3.markdown(f"<div style='background-color:#d4edda; padding:10px; border-radius:10px;'>"
                          f"<h4>🌍 Climatic</h4>"
                          f"<p>{hazards[pred_class]['Climatic']}</p></div>", unsafe_allow_html=True)
