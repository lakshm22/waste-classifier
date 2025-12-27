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
# Hazards Dictionary
# -----------------------
hazards = {
    "Plastic": {
        "Health": [
            "Toxic chemicals leach into soil and water.",
            "Microplastics enter the food chain, causing digestive and hormonal issues.",
            "Burning plastics releases dioxins and furans—respiratory irritants."
        ],
        "Environmental": [
            "Pollutes oceans, rivers, and soil.",
            "Wildlife ingests or gets entangled, leading to injury or death.",
            "Non-biodegradable; persists for hundreds of years in landfills."
        ],
        "Climatic": [
            "Incineration produces greenhouse gases.",
            "Manufacturing plastics consumes fossil fuels, adding to carbon footprint.",
            "Contributes indirectly to global warming and ocean acidification."
        ]
    },
    "Organic": {
        "Health": [
            "Rotting organic matter attracts rodents, flies, and disease vectors.",
            "Produces foul odors and harmful gases like methane and ammonia.",
            "Can cause respiratory issues and spread bacterial infections."
        ],
        "Environmental": [
            "Decomposing organic waste pollutes soil and water if unmanaged.",
            "Can lead to algal blooms when nutrients leach into water bodies.",
            "Attracts pests, increasing local ecological imbalance."
        ],
        "Climatic": [
            "Methane emissions from decomposition are ~28x more potent than CO₂.",
            "Contributes significantly to urban greenhouse gas emissions.",
            "Poor management can worsen urban flooding through blocked drains."
        ]
    },
    "Paper": {
        "Health": [
            "Accumulated paper waste can be a fire hazard.",
            "Dust from shredded paper may irritate respiratory systems."
        ],
        "Environmental": [
            "Landfill accumulation uses space and can block drainage.",
            "Decomposing paper produces small amounts of methane.",
            "Wasteful consumption of trees reduces forest cover."
        ],
        "Climatic": [
            "Deforestation for paper production reduces carbon sequestration.",
            "Decomposition releases greenhouse gases contributing to warming."
        ]
    },
    "Metal": {
        "Health": [
            "Sharp edges can cause physical injuries.",
            "Toxic metals (like lead, cadmium) can leach and affect organs."
        ],
        "Environmental": [
            "Heavy metals contaminate soil and water.",
            "Can bioaccumulate in plants and animals.",
            "Mining and disposal disrupt ecosystems."
        ],
        "Climatic": [
            "Energy-intensive production emits CO₂.",
            "Improper disposal can release harmful gases when incinerated."
        ]
    },
    "E-waste": {
        "Health": [
            "Contains lead, mercury, cadmium—damaging to kidneys, liver, nervous system.",
            "Children exposed to e-waste may suffer developmental delays.",
            "Burning or dismantling releases toxic fumes causing respiratory problems."
        ],
        "Environmental": [
            "Leaches toxins into soil and groundwater.",
            "Affects plant growth and contaminates food chains.",
            "Causes long-term ecological damage if not recycled properly."
        ],
        "Climatic": [
            "Incineration releases CO₂ and toxic gases.",
            "Energy used in processing and recycling contributes to carbon footprint.",
            "Improper disposal increases greenhouse gas emissions indirectly."
        ]
    }
}

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

        # Display hazards in colored cards with bullet points
        with st.expander(f"⚠️ Hazards of {pred_class} Waste"):
            col1, col2, col3 = st.columns(3)

            # Health Card
            col1.markdown(
                f"<div style='background-color:#ffcccc; padding:15px; border-radius:10px; color:black;'>"
                f"<h5>💊 Health</h5>"
                f"<ul>{''.join([f'<li>{item}</li>' for item in hazards[pred_class]['Health']])}</ul>"
                f"</div>", unsafe_allow_html=True
            )

            # Environmental Card
            col2.markdown(
                f"<div style='background-color:#cce5ff; padding:15px; border-radius:10px; color:black;'>"
                f"<h5>🌱 Environmental</h5>"
                f"<ul>{''.join([f'<li>{item}</li>' for item in hazards[pred_class]['Environmental']])}</ul>"
                f"</div>", unsafe_allow_html=True
            )

            # Climatic Card
            col3.markdown(
                f"<div style='background-color:#d4edda; padding:15px; border-radius:10px; color:black;'>"
                f"<h5>🌍 Climatic</h5>"
                f"<ul>{''.join([f'<li>{item}</li>' for item in hazards[pred_class]['Climatic']])}</ul>"
                f"</div>", unsafe_allow_html=True
            )
