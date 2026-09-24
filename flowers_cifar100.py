# based on sir's original file


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
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

print("Fine training images:", fine_train_images.shape)
print("Coarse training labels:", coarse_train_labels.shape)
print("Fine testing images:", fine_test_images.shape)
print("Coarse testing labels:", coarse_test_labels.shape)


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


# Extract the actual images and fine labels
flowers_train_images = fine_train_images[train_idx]
flowers_train_labels = fine_train_labels[train_idx]

flowers_test_images = fine_test_images[test_idx]
flowers_test_labels = fine_test_labels[test_idx]


print("Flowers training images:", flowers_train_images.shape)
print("Flowers training labels:", flowers_train_labels.shape)
print("Flowers testing images:", flowers_test_images.shape)
print("Flowers testing labels:", flowers_test_labels.shape)


# ============================================================
# 3. CONVERT ORIGINAL FINE LABELS TO 0-4
# ============================================================

# Flowers fine classes:
#
# 54 = orchid
# 62 = poppy
# 70 = rose
# 82 = sunflower
# 92 = tulip
#
# We convert them to:
#
# orchid    -> 0
# poppy     -> 1
# rose      -> 2
# sunflower -> 3
# tulip     -> 4

label_mapping = {
    54: 0,  # orchid
    62: 1,  # poppy
    70: 2,  # rose
    82: 3,  # sunflower
    92: 4   # tulip
}


def convert_labels(labels):
    return np.array([
        label_mapping[int(label[0])]
        for label in labels
    ])


flowers_train_labels = convert_labels(flowers_train_labels)
flowers_test_labels = convert_labels(flowers_test_labels)


# ============================================================
# 4. DISPLAY DATASET INFORMATION
# ============================================================

class_names = [
    "orchid",
    "poppy",
    "rose",
    "sunflower",
    "tulip"
]

print("\nFlower classes:")
for i, name in enumerate(class_names):
    print(i, "=", name)

print("\nFinal training dataset:")
print("Images:", flowers_train_images.shape)
print("Labels:", flowers_train_labels.shape)

print("\nFinal testing dataset:")
print("Images:", flowers_test_images.shape)
print("Labels:", flowers_test_labels.shape)


# ============================================================
# 5. DISPLAY SAMPLE IMAGES
# ============================================================

plt.figure(figsize=(10, 2))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.imshow(flowers_train_images[i])
    plt.xlabel(class_names[flowers_train_labels[i]])

plt.tight_layout()
plt.show()


# ============================================================
# 6. NORMALIZE IMAGE PIXEL VALUES
# ============================================================

flowers_train_images = flowers_train_images.astype("float32") / 255.0
flowers_test_images = flowers_test_images.astype("float32") / 255.0


# ============================================================
# 7. BUILD THE CNN MODEL
# ============================================================

model = tf.keras.Sequential()

model.add(
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(32, 32, 3)
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

model.add(
    Flatten()
)

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dense(
        5,
        activation="softmax"
    )
)


# ============================================================
# 8. DISPLAY MODEL STRUCTURE
# ============================================================

print("\nModel Summary:")
model.summary()


# ============================================================
# 9. COMPILE THE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy()
    ]
)


# ============================================================
# 10. TRAIN THE MODEL
# ============================================================

print("\nStarting training...")

metricInfo = model.fit(
    flowers_train_images,
    flowers_train_labels,
    epochs=5,
    validation_split=0.1
)


# ============================================================
# 11. PLOT TRAINING VS VALIDATION LOSS
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

plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()


# ============================================================
# 12. TEST THE MODEL
# ============================================================

print("\nTesting model...")

test_loss, test_acc = model.evaluate(
    flowers_test_images,
    flowers_test_labels
)

print("\nTest loss:", test_loss)
print("Test accuracy:", test_acc)
