import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from keras.models import load_model
import cv2

model = load_model("best_model_new.h5")

emotion_labels = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear',
    3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            face_roi = gray[y:y+h, x:x+w]
            resized = cv2.resize(face_roi, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))

            predictions = model.predict(reshaped, verbose=0)
            label = np.argmax(predictions)
            emotion = emotion_labels[label]

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(img, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

        return img

st.title("Real-Time Emotion Detection")

webrtc_streamer(
    key="emotion",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
)
