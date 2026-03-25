"""
Train ResNet50 for ASL Fingerspelling
======================================
Fine-tunes a pretrained ResNet50 on the collected hand gesture dataset
using KerasTuner to search over unfreezing depth, dense layer width,
dropout, and learning rate.

Steps:
    1. Split raw images into train / val / test (70 / 15 / 15).
    2. Augment training data (rotation, zoom, shift, flip).
    3. Run KerasTuner RandomSearch to find the best hyperparameters.
    4. Retrain the best model configuration and evaluate on the test set.
    5. Save the final model to resnet_sign_language_best_model.h5.

Usage:
    python train.py
"""

import os
import shutil
import random

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
import kerastuner as kt

# ── Configuration ────────────────────────────────────────────
RAW_DATA_DIR = "dataset"
SPLIT_DIR = "splitted_dataset"
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
MAX_EPOCHS = 10
MODEL_PATH = "resnet_sign_language_best_model.h5"


# ── 1. Dataset splitting ────────────────────────────────────
def split_dataset(raw_dir, base_dir):
    """Split raw class folders into train / val / test directories."""
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    for split in [train_dir, val_dir, test_dir]:
        os.makedirs(split, exist_ok=True)

    for class_name in os.listdir(raw_dir):
        class_path = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for split in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(split, class_name), exist_ok=True)

        files = os.listdir(class_path)
        random.shuffle(files)
        total = len(files)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        for i, fname in enumerate(files):
            src = os.path.join(class_path, fname)
            if i < train_end:
                dst = os.path.join(train_dir, class_name, fname)
            elif i < val_end:
                dst = os.path.join(val_dir, class_name, fname)
            else:
                dst = os.path.join(test_dir, class_name, fname)
            shutil.copy(src, dst)

    print(f"Dataset split into {base_dir}/  (train / val / test).")
    return train_dir, val_dir, test_dir


if not os.path.exists(SPLIT_DIR):
    train_dir, val_dir, test_dir = split_dataset(RAW_DATA_DIR, SPLIT_DIR)
else:
    train_dir = os.path.join(SPLIT_DIR, "train")
    val_dir = os.path.join(SPLIT_DIR, "val")
    test_dir = os.path.join(SPLIT_DIR, "test")
    print(f"Using existing split in '{SPLIT_DIR}/'.")


# ── 2. Data generators with augmentation ────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, class_mode="categorical",
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, class_mode="categorical",
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False,
)

num_classes = len(train_gen.class_indices)
print(f"Detected {num_classes} classes: {train_gen.class_indices}")

# Preview a few augmented samples
images, _ = next(train_gen)
plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i])
    plt.axis("off")
plt.suptitle("Augmented Training Samples")
plt.savefig("augmented_samples.png", dpi=100, bbox_inches="tight")
plt.show()


# ── 3. Model builder for KerasTuner ─────────────────────────
def build_model(hp):
    """Construct a ResNet50 + classification head with tuneable hyperparameters."""
    base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    )

    # Freeze early layers, fine-tune from unfreeze_from onward
    unfreeze_from = hp.Int("unfreeze_from", min_value=100, max_value=165, step=15)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= unfreeze_from

    x = tf.keras.layers.Flatten()(base.output)
    x = tf.keras.layers.Dense(
        hp.Int("dense_units", 64, 512, step=64), activation="relu"
    )(x)
    x = tf.keras.layers.Dropout(
        hp.Float("dropout_rate", 0.2, 0.6, step=0.1)
    )(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── 4. Hyperparameter search ────────────────────────────────
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    directory="kt_tuner_dir",
    project_name="sign_language_resnet",
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

print("\nStarting hyperparameter search...")
tuner.search(
    train_gen, epochs=MAX_EPOCHS,
    validation_data=val_gen, callbacks=[early_stop],
)
tuner.results_summary()


# ── 5. Final training + evaluation ──────────────────────────
best_model = tuner.get_best_models(num_models=1)[0]

print("\nRetraining best model...")
history = best_model.fit(
    train_gen, validation_data=val_gen,
    epochs=MAX_EPOCHS, callbacks=[early_stop],
)

test_loss, test_acc = best_model.evaluate(test_gen)
print(f"\nTest Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

best_model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
