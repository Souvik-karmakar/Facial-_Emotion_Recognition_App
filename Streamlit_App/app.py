import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from tensorflow.keras.utils import img_to_array   # ✅ FIXED IMPORT

# -----------------------------
# 🎨 PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😃",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 🌈 CUSTOM CSS FOR BEAUTIFUL UI
# -----------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to bottom right, #1e3c72, #2a5298);
}
h1 {
    color: #ffffff;
    text-align: center;
    font-size: 48px !important;
    font-weight: bold;
}
.uploaded-img {
    border-radius: 15px;
    border: 3px solid #ffffff;
}
.pred-box {
    padding: 20px;
    background: #ffffff;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
.footer {
    text-align:center;
    color:white;
    margin-top:50px;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 🤖 LOAD MODEL
# -----------------------------
with st.spinner("Loading FER Model..."):
    model = load_model("best_model_new.h5")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Emotion labels
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# -----------------------------
# 🧠 PREPROCESSING FUNCTION
# -----------------------------
def preprocess_image(img, target_size=(48, 48)):
    img = img.convert("L")                 # Convert to grayscale
    img = img.resize(target_size)          # Resize to 48x48
    img_arr = img_to_array(img)            # ✅ Now works
    img_arr = img_arr / 255.0              # Normalize
    img_arr = np.reshape(img_arr, (1, 48, 48, 1))
    return img_arr

# -----------------------------
# 🚀 APP TITLE
# -----------------------------
st.markdown("<h1>Emotion Detection 😃🤖</h1>", unsafe_allow_html=True)
st.write("### Upload any human face image and the AI will detect the emotion.")

# -----------------------------
# 📤 FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    st.image(uploaded_file, width=300, caption="Uploaded Image", use_column_width=False)

    if st.button("🔍 Analyze Emotion"):
        with st.spinner("Analyzing emotion..."):
            img = Image.open(uploaded_file)

            # Preprocess
            processed = preprocess_image(img)

            # Prediction
            preds = model.predict(processed)
            label = np.argmax(preds)

        # Output box
        st.markdown(
            f"""
            <div class='pred-box'>
                Predicted Emotion: <span style='color:#1e3c72;'>{emotion_labels[label]}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# 📌 FOOTER
# -----------------------------
st.markdown(
    "<div class='footer'>Developed with ❤️ using Streamlit</div>",
    unsafe_allow_html=True
)