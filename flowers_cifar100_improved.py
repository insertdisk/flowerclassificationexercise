import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense
)
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD CIFAR-100 DATASET
# ============================================================

print("Loading CIFAR-100 dataset...")

(fine_train_images, fine_train_labels), (fine_test_images, fine_test_labels) = (
    keras.datasets.cifar100.load_data(label_mode="fine")
)

(coarse_train_images, coarse_train_labels), (coarse_test_images, coarse_test_labels) = (
    keras.datasets.cifar100.load_data(label_mode="coarse")
)


# ============================================================
# 2. EXTRACT FLOWERS COARSE CLASS
# ============================================================

# CIFAR-100 coarse class:
# Flowers = 2

FLOWERS_COARSE_LABEL = 2

print("\nExtracting Flowers coarse class...")

# Training dataset
train_idx = []

for j in range(len(coarse_train_labels)):
    if coarse_train_labels[j][0] == FLOWERS_COARSE_LABEL:
        train_idx.append(j)

train_idx = np.array(train_idx)


# Testing dataset
test_idx = []

for j in range(len(coarse_test_labels)):
    if coarse_test_labels[j][0] == FLOWERS_COARSE_LABEL:
        test_idx.append(j)

test_idx = np.array(test_idx)


# Extract Flowers images using the selected indices
flowers_train_images = fine_train_images[train_idx]
flowers_train_labels = fine_train_labels[train_idx]

flowers_test_images = fine_test_images[test_idx]
flowers_test_labels = fine_test_labels[test_idx]


print("Flowers training images:", flowers_train_images.shape)
print("Flowers testing images:", flowers_test_images.shape)


# ============================================================
# 3. CONVERT FINE LABELS TO 0-4
# ============================================================

# Original CIFAR-100 fine labels:
#
# 54 = orchid
# 62 = poppy
# 70 = rose
# 82 = sunflower
# 92 = tulip

label_mapping = {
    54: 0,
    62: 1,
    70: 2,
    82: 3,
    92: 4
}


def convert_labels(labels):
    return np.array([
        label_mapping[int(label[0])]
        for label in labels
    ])


flowers_train_labels = convert_labels(flowers_train_labels)
flowers_test_labels = convert_labels(flowers_test_labels)


class_names = [
    "orchid",
    "poppy",
    "rose",
    "sunflower",
    "tulip"
]


# ============================================================
# 4. NORMALIZE IMAGE PIXELS
# ============================================================

flowers_train_images = flowers_train_images.astype("float32") / 255.0
flowers_test_images = flowers_test_images.astype("float32") / 255.0


# ============================================================
# 5. BUILD IMPROVED CNN MODEL
# ============================================================

model = tf.keras.Sequential([
    
    # First convolution block
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        input_shape=(32, 32, 3)
    ),

    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Dropout(0.25),


    # Second convolution block
    Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Dropout(0.25),


    # Classification section
    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        5,
        activation="softmax"
    )
])


# ============================================================
# 6. DISPLAY MODEL STRUCTURE
# ============================================================

print("\nImproved Model Summary:")
model.summary()


# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy()
    ]
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nStarting improved model training...")

metricInfo = model.fit(
    flowers_train_images,
    flowers_train_labels,
    epochs=15,
    validation_split=0.1
)


# ============================================================
# 9. PLOT TRAINING VS VALIDATION LOSS
# ============================================================

loss = metricInfo.history["loss"]
val_loss = metricInfo.history["val_loss"]

epochs = range(1, len(loss) + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    loss,
    "g-",
    label="Training loss"
)

plt.plot(
    epochs,
    val_loss,
    "b-",
    label="Validation loss"
)

plt.title("Improved Model: Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()


# ============================================================
# 10. PLOT TRAINING VS VALIDATION ACCURACY
# ============================================================

accuracy = metricInfo.history["sparse_categorical_accuracy"]
val_accuracy = metricInfo.history["val_sparse_categorical_accuracy"]

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    accuracy,
    "g-",
    label="Training accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    "b-",
    label="Validation accuracy"
)

plt.title("Improved Model: Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


# ============================================================
# 11. TEST THE MODEL
# ============================================================

print("\nTesting improved model...")

test_loss, test_acc = model.evaluate(
    flowers_test_images,
    flowers_test_labels
)


print("\n========================================")
print("IMPROVED MODEL RESULTS")
print("========================================")
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
print("Test accuracy percentage:", test_acc * 100, "%")