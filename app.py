from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import threading
import queue

app = Flask(__name__)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
classifier = Classifier("keras_model.h5", "labels.txt")

labels = ["A", "Again", "B", "Bathroom", "C", "Cat", "D", "Dog", "Done", "E", "Eat", "F", "Father", "Fine", "G", "Go to", "H", "Hello", "Help", "How", "I", "J", "K", "L", "Learn", "Like", "M", "Milk", "More", "Mother", "N", "Name", "No", "O", "P", "Please", "Q", "R", "S", "See", "Sign", "T", "Thank you", "U", "V", "W", "Want", "What", "X", "Y", "Yes", "You", "You later", "Z"]

cap = cv2.VideoCapture(0)
frame_queue = queue.Queue(maxsize=1)

def frame_reader():
    while True:
        success, frame = cap.read()
        if success:
            if not frame_queue.full():
                frame_queue.put(frame)

def process_frame(frame):
    if frame is None:
        return None, ""
    imgOutput = frame.copy()
    hands, _ = detector.findHands(frame)
    detected_text = ""
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure crop dimensions are within the frame bounds
        y1 = max(0, y - offset)
        y2 = min(frame.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(frame.shape[1], x + w + offset)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = frame[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            if wCal > 0 and imgCrop.shape[0] > 0:  # Ensure dimensions are valid
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            if hCal > 0 and imgCrop.shape[1] > 0:  # Ensure dimensions are valid
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

        detected_text = labels[index]
        cv2.rectangle(imgOutput, (x1, y1 - 70), (x1 + 400, y1 + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, detected_text, (x1, y1 - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 4)

    ret, jpeg = cv2.imencode('.jpg', imgOutput)
    return jpeg.tobytes(), detected_text

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        frame = frame_queue.get()
        frame_bytes, detected_text = process_frame(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        yield (f'data: {detected_text}\n\n').encode()

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_text')
def detected_text():
    frame = frame_queue.get()
    _, detected_text = process_frame(frame)
    return jsonify(detected_text=detected_text)

if __name__ == '__main__':
    threading.Thread(target=frame_reader, daemon=True).start()
    app.run(debug=True)
