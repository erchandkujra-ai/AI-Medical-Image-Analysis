
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

from src.data_utils import IMAGE_SIZE, MODEL_DIR, CLASS_NAMES


st.set_page_config(
    page_title="AI Medical Image Analysis",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 AI-Powered Medical Image Analysis")
st.caption("Educational chest X-ray classification prototype")

st.warning(
    "Research demonstration only. This model is not "
    "clinically validated and must not be used to diagnose, "
    "exclude disease, or make treatment decisions."
)

model_path = MODEL_DIR / "medical_cnn.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)


if not model_path.exists():
    st.error(
        "Model not found. First run: python main.py train"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a sample chest X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("L")
        st.image(
            image,
            caption="Uploaded image",
            width="stretch"
        )

        resized = image.resize(
            (IMAGE_SIZE[1], IMAGE_SIZE[0])
        )
        array = np.asarray(
            resized, dtype=np.float32
        ) / 255.0
        array = array[np.newaxis, ..., np.newaxis]

        if st.button("Run educational model"):
            model = load_model()
            score = float(
                model.predict(array, verbose=0)[0][0]
            )

            predicted_index = int(score >= 0.5)
            label = CLASS_NAMES[predicted_index]

            st.subheader("Model output")
            st.write(f"Predicted class: **{label}**")
            st.metric(
                "PNEUMONIA model score",
                f"{score:.1%}"
            )
            st.progress(min(max(score, 0.0), 1.0))

            st.caption(
                "The score is not a clinically calibrated "
                "probability. Model output is not a diagnosis."
            )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
