import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# عنوان المشروع
st.title("COVID-19 X-ray Detection System")
st.write("Chest X-ray Classification using CNN")

st.warning(
    "For educational and research purposes only. "
    "This system is not a medical diagnosis."
)

# تحميل الموديل
model = tf.keras.models.load_model("covid_normal_model.keras")

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
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "NORMAL"
    else:
        result = "COVID-19"

    st.subheader("Prediction Result")
    st.success(result)