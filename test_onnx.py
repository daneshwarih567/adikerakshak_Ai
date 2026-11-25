
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from PIL import Image

# Load original Keras model
keras_model = tf.keras.models.load_model("arecanut_model.keras")

# Load ONNX model
onnx_session = ort.InferenceSession(
    "arecanut_model.onnx",
    providers=["CPUExecutionProvider"]
)

# Load class names
import json
with open("classes.json", "r") as f:
    classes = json.load(f)

# Select one test image
# Change this path to an actual arecanut image
image_path = input("Enter the path of an arecanut image: ")

img = Image.open(image_path).convert("RGB")
img = img.resize((224, 224))

# Prepare input
image_array = np.array(img, dtype=np.float32)
image_array = np.expand_dims(image_array, axis=0)

# Original model prediction
keras_prediction = keras_model.predict(image_array, verbose=0)

# ONNX prediction
input_name = onnx_session.get_inputs()[0].name
onnx_prediction = onnx_session.run(
    None,
    {input_name: image_array}
)[0]

# Compare predictions
keras_index = int(np.argmax(keras_prediction))
onnx_index = int(np.argmax(onnx_prediction))

print("\n===== MODEL COMPARISON =====")
print("Keras prediction:", keras_index)
print("ONNX prediction:", onnx_index)
print("Keras confidence:", float(np.max(keras_prediction)) * 100)
print("ONNX confidence:", float(np.max(onnx_prediction)) * 100)

if keras_index == onnx_index:
    print("\nSUCCESS: Both models predict the same class!")
else:
    print("\nWARNING: Predictions are different. Further checking is needed.")
