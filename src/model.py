
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape=(160, 160, 1)):
    """Build a small CNN for educational binary image classification."""

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Extract simple and increasingly complex visual features.
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Convert learned feature maps into a feature vector.
        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.30),

        # One sigmoid score: probability of the positive class.
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
