
import tensorflow as tf
import json
import os

# Load the trained ML model
model = tf.keras.models.load_model("arecanut_model.keras")

# Load disease names
with open("classes.json", "r") as file:
    classes = json.load(file)

print("ML MODEL LOADED SUCCESSFULLY!")
print("Available disease classes:")
print(classes)
