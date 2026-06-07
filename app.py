import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load model
model = tf.keras.models.load_model("fruit_mobilenet.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("🍎 Fruit Classifier AI")
st.write("Upload a fruit image and let AI identify it!")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224, 224))

    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(
        f"Prediction: {class_names[predicted_index]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )
