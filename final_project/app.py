from flask import Flask, render_template, Response, jsonify, url_for
import cv2
import onnxruntime as ort
import numpy as np
from collections import Counter
import datetime
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import queue
import torch

app = Flask(__name__)

# Load MobileNetV2-SSD model (pre-trained) using OpenCV's DNN module
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd.prototxt", "mobilenet_iter_73000.caffemodel")

# confidence threshold
GENDER_CONFIDENCE_THRESHOLD = 0.9

# ONNX models (face detection, gender classification, and age classification) with optimized session options
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# Load the UltraFace, Gender, and Age models with session options
face_detector = ort.InferenceSession("ultraface/models/version-RFB-640.onnx", sess_options=session_options)
gender_classifier = ort.InferenceSession("gender_googlenet.onnx", sess_options=session_options)
age_classifier = ort.InferenceSession("age_googlenet.onnx", sess_options=session_options)

genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Thread pool executor for concurrent face detection, gender classification, and age classification
executor = ThreadPoolExecutor(max_workers=4)

# Thresholds
night_time_threshold = 19  # 7 PM as the start of night-time

# Global variables to store counts and alerts
counts = {'male_count': 0, 'female_count': 0, 'age_groups': {}, 'alerts': []}

# Queue for buffering frames
frame_queue = queue.Queue(maxsize=10)
# Add a sliding window for gender classification
class GenderClassificationWindow:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.gender_history = []

    def add_classification(self, gender, confidence):
        self.gender_history.append((gender, confidence))
        if len(self.gender_history) > self.window_size:
            self.gender_history.pop(0)

    def get_stable_classification(self):
        if not self.gender_history:
            return None, 0.0

        genders = [g for g, c in self.gender_history if c >= GENDER_CONFIDENCE_THRESHOLD]
        if not genders:
            return None, 0.0

        gender_counts = Counter(genders)
        most_common_gender, count = gender_counts.most_common(1)[0]
        confidence = count / len(self.gender_history)

        return most_common_gender, confidence

# Create a dictionary to store GenderClassificationWindow objects for each face
face_gender_windows = {}

class VideoStream:
    def __init__(self, source=0):
        self.stream = cv2.VideoCapture(source)
        self.stopped = False
        self.ret = None
        self.frame = None

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream.isOpened():
                self.ret, self.frame = self.stream.read()
                if self.ret and not frame_queue.full():
                    frame_queue.put(self.frame)  # Add frame to queue
            else:
                self.stopped = True

    def stop(self):
        self.stopped = True
        self.stream.release()

def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width) / 2)
    dy = int((maximum - height) / 2)
    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes

def cropImage(image, box):
    return image[box[1]:box[3], box[0]:box[2]]

def faceDetector(orig_image, threshold=0.7):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0).astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs

def genderClassifier(orig_image, face_id):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_mean = np.array([104, 117, 123])
    image = image - image_mean
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0).astype(np.float32)

    input_name = gender_classifier.get_inputs()[0].name
    genders = gender_classifier.run(None, {input_name: image})
    
    gender_probs = genders[0][0]
    gender_index = gender_probs.argmax()
    gender_confidence = gender_probs[gender_index]
    
    if face_id not in face_gender_windows:
        face_gender_windows[face_id] = GenderClassificationWindow()
    
    face_gender_windows[face_id].add_classification(genderList[gender_index], gender_confidence)
    
    stable_gender, stable_confidence = face_gender_windows[face_id].get_stable_classification()
    
    if stable_gender and stable_confidence >= GENDER_CONFIDENCE_THRESHOLD:
        return stable_gender, stable_confidence
    else:
        return "Unknown", 0.0

def ageClassifier(orig_image):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_mean = np.array([104, 117, 123])
    image = image - image_mean
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0).astype(np.float32)

    input_name = age_classifier.get_inputs()[0].name
    ages = age_classifier.run(None, {input_name: image})
    return ageList[ages[0].argmax()]

def detectSOSGesture(results):
    for result in results:
        if "sos_gesture" in result:
            return True
    return False

def isLoneWomanAtNight(male_count, female_count):
    current_time = datetime.datetime.now().hour
    if current_time >= night_time_threshold and female_count == 1 and male_count == 0:
        return True
    return False

def womanSurroundedByMen(male_count, female_count):
    if male_count > 1 and female_count == 1:
        return True
    return False

def process_frame(frame):
    global counts

    # Resize the frame for MobileNetV2-SSD
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Perform person detection using MobileNetV2-SSD
    net.setInput(blob)
    detections = net.forward()

    genders = []
    ages = []
    color = (255, 128, 0)  # Color for bounding boxes and text (orange)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Threshold for detection
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Class ID for 'person' in MobileNet SSD
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)

                # Draw bounding box around the detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Perform face detection, gender classification, and age classification
                face_box = scale([x1, y1, x2, y2])
                face = cropImage(frame, face_box)

                if face.size > 0:
                    face_id = hash(face.tobytes())  # Create a unique ID for each face
                    gender, gender_confidence = genderClassifier(face, face_id)
                    age = ageClassifier(face)
                    
                    if gender != "Unknown":
                        genders.append(gender)
                        ages.append(age)

                        # Annotate the frame with the gender, age, and confidence labels
                        label = f"{gender} ({gender_confidence:.2f}), {age}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "Unknown gender, " + age, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # Count the number of males and females detected
    gender_counter = Counter(genders)
    age_counter = Counter(ages)
    male_count = gender_counter['Male']
    female_count = gender_counter['Female']

    # Update global counts
    counts['male_count'] = male_count
    counts['female_count'] = female_count
    counts['age_groups'] = dict(age_counter)
    counts['alerts'] = []

    # Check for specific alert conditions
    if isLoneWomanAtNight(male_count, female_count):
        counts['alerts'].append('ALERT: Lone Woman Detected!')
        cv2.putText(frame, 'ALERT: Lone Woman Detected!', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    if womanSurroundedByMen(male_count, female_count):
        counts['alerts'].append('ALERT: Woman Surrounded by Men!')
        cv2.putText(frame, 'ALERT: Woman Surrounded by Men!', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    if detectSOSGesture(detections):
        counts['alerts'].append('SOS Gesture Detected!')
        cv2.putText(frame, 'SOS Gesture Detected!', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    return frame  # Return the processed frame


def generate_frames():
    stream = VideoStream().start()  # Start video stream in a separate thread

    if not stream.stream.isOpened():
        print("Error: Could not open video stream.")
        return

    frame_skip = 2  # Process every 2nd frame for performance optimization
    frame_counter = 0

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Skip frames for better performance
            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            # Process frame in a separate thread
            processed_frame = executor.submit(process_frame, frame).result()

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue

            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    stream.stop()  # Stop the video stream when done

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(counts)

if __name__ == "__main__":
    app.run(debug=True)