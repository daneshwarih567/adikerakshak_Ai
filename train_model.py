
import os
import json
import tensorflow as tf
from collections import Counter

# 1. Dataset location
DATASET_PATH = (
    r"C:\Users\Daneshwari\Downloads\archive"
    r"\Arecanut_dataset\Arecanut_dataset"
)

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# 2. Check dataset folders
if not os.path.isdir(TRAIN_PATH):
    raise FileNotFoundError("Training folder not found: " + TRAIN_PATH)

if not os.path.isdir(TEST_PATH):
    raise FileNotFoundError("Testing folder not found: " + TEST_PATH)

# 3. Basic settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

print("Loading training images...")

# 4. Load training data (80% for training)
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED
)

# 5. Load validation data (20% of training folder)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED
)

# 6. Get disease names
class_names = train_ds.class_names
num_classes = len(class_names)

print("\nDisease categories:")
for name in class_names:
    print("-", name)

print("Total categories:", num_classes)

# 7. Load separate test data
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 8. Calculate class weights for imbalanced data
counts = {}

for name in class_names:
    folder = os.path.join(TRAIN_PATH, name)
    counts[name] = sum(
        1 for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    )

total_images = sum(counts.values())

class_weights = {
    i: total_images / (num_classes * counts[name])
    for i, name in enumerate(class_names)
    if counts[name] > 0
}

print("\nClass image counts:", counts)

# 9. Improve data loading
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# 10. Load MobileNetV2
print("\nLoading MobileNetV2...")

try:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    print("Pretrained weights loaded.")
except Exception as e:
    print("Could not download pretrained weights:", e)
    print("Using randomly initialized MobileNetV2 instead.")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )

base_model.trainable = False

# 11. Build the disease classification model
inputs = tf.keras.Input(shape=(224, 224, 3))

x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.1)(x)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# 12. Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 13. Save the best model automatically
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "arecanut_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# 14. Train the model
print("\nStarting model training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

# 15. Evaluate using separate test images
print("\nTesting the trained model...")

best_model = tf.keras.models.load_model(
    "arecanut_model.keras"
)

test_loss, test_accuracy = best_model.evaluate(test_ds)

print("\nTest accuracy:", round(test_accuracy * 100, 2), "%")

# 16. Save class names
with open("classes.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("\nMODEL TRAINING COMPLETED!")
print("Model saved: arecanut_model.keras")
print("Classes saved: classes.json")
print(
    "Model file size:",
    os.path.getsize("arecanut_model.keras"),
    "bytes"
)
