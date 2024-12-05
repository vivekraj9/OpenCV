from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

# Load the face classifier and the pre-trained model
face_classifier = cv2.CascadeClassifier(r'C:\Users\vivek\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
classifier = load_model(r"C:\Users\vivek\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the webcam
cap = cv2.VideoCapture(0)

def detect_emotion():
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        root.quit()
        return

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not read frame.")
        lblVideo.after(10, detect_emotion)
        return

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around each face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Predict emotion
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert to ImageTk format for Tkinter
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lblVideo.imgtk = imgtk
    lblVideo.configure(image=imgtk)
    lblVideo.after(10, detect_emotion)

# Tkinter window setup
root = tk.Tk()
root.title("Emotion Detector")
lblVideo = tk.Label(root)
lblVideo.pack()

# Start emotion detection
detect_emotion()

# Exit the program when the window is closed
root.protocol("WM_DELETE_WINDOW", root.quit)
root.mainloop()

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
