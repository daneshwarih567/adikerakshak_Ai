
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "arecanut_model.keras"
IMAGE_SIZE = (224, 224)


def generate_gradcam(image_input):

    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

    base_model = model.get_layer("mobilenetv2_1.00_224")
    last_conv = base_model.get_layer("Conv_1")

    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[last_conv.output, base_model.output]
    )

    # Handle an already-open PIL image or an image file
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")

    elif isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")

    else:
        raise TypeError(
            "Please provide a PIL image or an image file path."
        )

    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    input_array = np.expand_dims(image_array, axis=0)

    # Apply augmentation layers
    x = input_array

    for layer in model.layers[1:3]:
        x = layer(x, training=False)

    # Calculate Grad-CAM
    with tf.GradientTape() as tape:

        feature_maps, base_output = feature_model(
            x, training=False
        )

        predictions = base_output

        for layer in model.layers[4:]:
            predictions = layer(
                predictions, training=False
            )

        class_index = tf.argmax(predictions[0])
        score = predictions[:, class_index]

    gradients = tape.gradient(score, feature_maps)

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )

    weights = tf.reduce_mean(
        gradients, axis=(0, 1, 2)
    )

    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(
        feature_maps * weights, axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap, cv2.COLORMAP_JET
    )

    original = np.array(image)
    original = cv2.cvtColor(
        original, cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(
        original, 0.6,
        heatmap, 0.4, 0
    )

    overlay = cv2.cvtColor(
        overlay, cv2.COLOR_BGR2RGB
    )

    return predictions.numpy()[0], overlay
