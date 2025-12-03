# 🎭 Facial Emotion Recognition System

### *(CNN + ResNet50V2 + Streamlit UI + Real-Time OpenCV)*

This project is a complete **Facial Emotion Recognition System** built with **Deep Learning, OpenCV, and Streamlit**.

It detects human emotions from:

✔ Uploaded images
✔ Live webcam feed

---

# 🚀 Features

### **🔥 1. Image-Based Emotion Recognition (Streamlit App)**

* Upload any face image
* Model predicts emotion from 7 classes:
  *Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise*
* Shows prediction + confidence bar chart
* Clean and modern Streamlit UI

### **🎥 2. Real-Time Emotion Monitoring (OpenCV App)**

* Uses webcam feed
* Detects faces in real-time
* CNN or ResNet50V2 predicts emotion
* Bounding box + confidence score displayed

---

# 🧠 Deep Learning Models

Two models were trained:

### ✔ **Custom CNN Model (from scratch)**

### ✔ **ResNet50V2 (Fine-Tuned Transfer Learning)**

Training performed on the **FER2013 dataset**.

---

# 🛠️ Tech Stack

| Category        | Technologies                 |
| --------------- | ---------------------------- |
| Deep Learning   | TensorFlow, Keras            |
| Computer Vision | OpenCV, Haar Cascade         |
| UI / Frontend   | Streamlit                    |
| Programming     | Python                       |
| Tools           | VS Code, Virtual Environment |

---

# 📂 Project Structure

```
📦 Facial Emotion Recognition
│
├── Real_Time/
│     ├── webcam.py
│     ├── haarcascade_frontalface_default.xml
│
├── Streamlit_App/
│     ├── app.py
│
├── Models/
│     ├── best_model_new.h5   (not included in repo – add via Google Drive)
│     ├── best_resnet_model.h5
│
├── requirements.txt
├── README.md
```

---

# ▶️ How to Run

## **1️⃣ Run Streamlit App (Image-Based Detection)**

Activate virtual environment:

```
venv\Scripts\activate
```

Run app:

```
streamlit run Streamlit_App/app.py
```

---

## **2️⃣ Run Real-Time Webcam Emotion Detector**

```
python Real_Time/webcam.py
```

Press **Q** to exit webcam window.

---

# 🛠️ Installation

Create venv:

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# 📸 Screenshots

### **Static Image Emotion Detection**

(Add your screenshot here)

### **🎥 Real-Time Monitoring Demo**

Upload your video into the repo OR link to Google Drive.

---

# ⭐ Future Improvements

* MediaPipe face tracking
* MobileNetV2 lightweight deployment
* Multi-face real-time detection
* Cloud deployment (Streamlit Cloud)

---

# 🤝 Contributing

Pull requests are welcome.

---

# 📬 Contact

**Developer:** Souvik Karmakar
**Field:** Data Science / ML
**GitHub:** [https://github.com/Souvik-karmakar](https://github.com/Souvik-karmakar)

---

# 🎉 DONE!

You can copy-paste the README directly into GitHub now.


