
import tensorflow as tf
import numpy as np
import json
from tkinter import Tk, filedialog

# Load trained model
print("Loading AdikeRakshak AI model...")
model = tf.keras.models.load_model("arecanut_model.keras")

# Load disease names
with open("classes.json", "r") as file:
    classes = json.load(file)

# Open image selection window
root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Arecanut Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

root.destroy()

if not image_path:
    print("No image selected.")
    raise SystemExit

# Load and resize image
img = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

# Convert image into array
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict disease
predictions = model.predict(img_array, verbose=0)

predicted_index = int(np.argmax(predictions[0]))
confidence = float(predictions[0][predicted_index]) * 100

# Display result
print("\n===== ADIKERAKSHAK AI RESULT =====")
print("Selected image:", image_path)
print("Predicted category:", classes[predicted_index])
print("Confidence:", round(confidence, 2), "%")
print("=================================")
