import cv2
import numpy as np
import csv
from datetime import datetime
from tensorflow.keras.models import load_model
import os

def start_screenshot_logging():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_model = load_model('emotion_model.h5')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    with open('emotion_log.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Emotion", "Coordinates", "Screenshot"])

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi / 255.0
                face_roi = face_roi.reshape(1, 48, 48, 1)
                prediction = emotion_model.predict(face_roi)
                max_index = np.argmax(prediction)
                emotion = emotion_labels[max_index]
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                # Capture screenshot
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                screenshot_path = f'screenshots/screenshot_{timestamp}.png'
                cv2.imwrite(screenshot_path, frame)
                
                # Log the detected emotion with timestamp and screenshot path
                writer.writerow([timestamp, emotion, (x, y, w, h), screenshot_path])

            cv2.imshow('Screenshot and Emotion Logging', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_screenshot_logging()