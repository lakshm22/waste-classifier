import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# Load model
def load_model(model_path="model/waste_model.h5"):
    return tf.keras.models.load_model(model_path)

# Load labels
def load_labels(label_path="model/labels.json"):
    with open(label_path, "r") as f:
        return json.load(f)

# Preprocess image for prediction
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict
def predict(img, model, labels):
    processed = preprocess_image(img)
    pred = model.predict(processed)
    class_idx = np.argmax(pred, axis=1)[0]
    return labels[str(class_idx)]
