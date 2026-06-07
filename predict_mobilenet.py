import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("fruit_mobilenet.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

img_path = input("Enter image path: ")

img = image.load_img(img_path, target_size=(224,224))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)

print("\nPrediction:", class_names[predicted_index])
print("Confidence:", round(np.max(prediction)*100,2), "%")




