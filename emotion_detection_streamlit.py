import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier(r'C:\Users\vivek\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')

try:
    classifier = load_model(r"C:\Users\vivek\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5", compile=False)
except Exception as e:
    st.error("Error loading model. Ensure compatibility and path correctness.")
    raise

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion_and_draw_rectangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    detected_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]
            detected_faces.append((x, y, w, h, label))
        else:
            detected_faces.append((x, y, w, h, "No Faces Detected"))

    return detected_faces

# Streamlit UI
st.title("Emotion Detection with Webcam")
FRAME_WINDOW = st.image([])
start_button = st.button("Start Webcam")
stop_button = st.button("Stop Webcam")

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if start_button:
    st.session_state.camera_active = True

if stop_button:
    st.session_state.camera_active = False

if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    st.write("Press 'Stop Webcam' to stop the feed.")

    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access the camera.")
            break

        faces = detect_emotion_and_draw_rectangle(frame)
        for (x, y, w, h, label) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
