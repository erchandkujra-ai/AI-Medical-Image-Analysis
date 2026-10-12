
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.data_utils import (
    IMAGE_SIZE, MODEL_DIR, CLASS_NAMES
)


def predict_image(image_path):
    image_path = Path(image_path)
    model_path = MODEL_DIR / "medical_cnn.keras"

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            "Model not found. Run: python main.py train"
        )

    # PIL handles common image formats consistently.
    image = Image.open(image_path).convert("L")
    image = image.resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0])
    )

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array[np.newaxis, ..., np.newaxis]

    model = tf.keras.models.load_model(model_path)
    pneumonia_score = float(
        model.predict(array, verbose=0)[0][0]
    )

    predicted_index = int(pneumonia_score >= 0.5)
    predicted_label = CLASS_NAMES[predicted_index]

    return {
        "predicted_label": predicted_label,
        "pneumonia_score": pneumonia_score,
        "normal_score": 1.0 - pneumonia_score
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python -m src.predict path/to/image.jpg"
        )
        raise SystemExit(1)

    result = predict_image(sys.argv[1])

    print("Educational model output:")
    print("Predicted class:", result["predicted_label"])
    print(
        "Pneumonia model score:",
        f"{result['pneumonia_score']:.4f}"
    )
    print(
        "Normal model score:",
        f"{result['normal_score']:.4f}"
    )
    print(
        "Not for clinical diagnosis or treatment decisions."
    )
