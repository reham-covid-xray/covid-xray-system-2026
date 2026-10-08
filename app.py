import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# عنوان المشروع
st.title("نظام التعرف على الامراض")
st.write("Chest X-ray Classification using CNN")

st.warning(
    "For educational and research purposes only. "
    "This system is not a medical diagnosis."
)

# تحميل الموديل الصغير
MODEL_PATH = "covid_normal_model_small.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# رفع الصورة
uploaded_file = st.file_uploader(
    "Upload a Chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded X-ray", width=400)

    # تجهيز الصورة
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    interpreter.set_tensor(
        input_details[0]["index"],
        img_array
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )[0][0]

    if prediction >= 0.5:
        result = "NORMAL"
    else:
        result = "COVID-19"

    st.subheader("Prediction Result")
    st.success(result)