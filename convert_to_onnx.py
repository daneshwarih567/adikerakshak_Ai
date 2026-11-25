
import tensorflow as tf
import tf2onnx

# Load your already-trained model
model = tf.keras.models.load_model("arecanut_model.keras")

# Convert model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    opset=13
)

# Save ONNX model
with open("arecanut_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX conversion completed successfully!")
print("Saved as arecanut_model.onnx")
