
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

from src.data_utils import (
    DATA_DIR, MODEL_DIR, OUTPUT_DIR,
    IMAGE_SIZE, BATCH_SIZE
)


def main():
    test_dir = DATA_DIR / "test"
    model_path = MODEL_DIR / "medical_cnn.keras"

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_dir}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            "Train the model first using: python main.py train"
        )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Normalize test images exactly as in training.
    test_ds = test_ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
    )

    model = tf.keras.models.load_model(model_path)

    # Because shuffle=False, predictions align with labels.
    y_true = np.concatenate([
        labels.numpy().reshape(-1)
        for _, labels in test_ds
    ]).astype(int)

    scores = model.predict(test_ds, verbose=1).reshape(-1)
    y_pred = (scores >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n--- Independent Test Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification report:")
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["NORMAL", "PNEUMONIA"],
        zero_division=0
    )
    print(report)

    report_path = OUTPUT_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Independent Test Results\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1-score: {f1:.4f}\n\n")
        file.write(report)

    # Save confusion matrix.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NORMAL", "PNEUMONIA"]
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Independent Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=150
    )
    plt.close()

    print(f"\nSaved evaluation files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
