
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data_utils import (
    DATA_DIR, MODEL_DIR, OUTPUT_DIR,
    IMAGE_SIZE, BATCH_SIZE, SEED
)
from src.model import build_model


def load_datasets():
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    for folder in (train_dir, val_dir):
        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

    # Grayscale images use a single channel.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Normalize pixels from 0-255 to 0-1.
    # Augmentation is applied only to training data.
    augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.02),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(0.02, 0.02)
    ])

    def prepare_train(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        images = augmentation(images, training=True)
        return images, labels

    def prepare_eval(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    train_ds = train_ds.map(
        prepare_train,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        prepare_eval,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def main():
    train_ds, val_ds = load_datasets()

    model = build_model()

    model.summary()

    best_model_path = MODEL_DIR / "medical_cnn.keras"

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks
    )

    # Save the history values for reproducibility.
    history_path = OUTPUT_DIR / "training_history.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history.history, file, indent=2)

    # Plot training and validation metrics.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(
        history.history["val_accuracy"],
        label="Validation"
    )
    axes[0].set_title("Accuracy by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(
        history.history["val_loss"],
        label="Validation"
    )
    axes[1].set_title("Loss by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "training_history.png",
        dpi=150
    )
    plt.close(fig)

    print(f"Best model saved at: {best_model_path}")
    print(f"Training graph saved at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
