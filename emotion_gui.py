import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection GUI")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_model = load_model('emotion_model.h5')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        self.label = Label(root)
        self.label.pack()

        self.start_button = Button(root, text="Start", command=self.start_detection)
        self.start_button.pack(side="left")

        self.stop_button = Button(root, text="Stop", command=self.stop_detection)
        self.stop_button.pack(side="right")

        self.cap = None
        self.running = False

    def start_detection(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.detect_emotion()

    def stop_detection(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.label.config(image='')

    def detect_emotion(self):
        if self.running:
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi / 255.0
                face_roi = face_roi.reshape(1, 48, 48, 1)
                prediction = self.emotion_model.predict(face_roi)
                max_index = np.argmax(prediction)
                emotion = self.emotion_labels[max_index]
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.config(image=imgtk)

            self.root.after(10, self.detect_emotion)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()