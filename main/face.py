import cv2
import numpy as np
import time
import pickle
import face_recognition
from ultralytics import YOLO
import torch
from threading import Thread, Lock

class FaceDetection:
    def __init__(self):
        print("Initializing FaceDetection")
        self.yolo_face = YOLO("D:/GradProje2/yolov8n-face.pt", verbose=False).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.width = 960  # Resize width
        self.height = 720  # Resize height
        self.fps_list = []
        self.recognition_active_time = 1  # Active time for recognition in seconds
        self.recognition_pause_duration = 3  # Pause duration in seconds
        self.last_recognition_time = 0
        self.recognition_thread = None
        self.lock = Lock()
        self.name = ""
        self.recognition_enabled = True

        print("Loading known face encodings")
        with open("D:/GradProje2/EncodeFile.p", "rb") as f:
            self.encodeListKnownWithNames = pickle.load(f)
        self.encodeListKnown, self.names = self.encodeListKnownWithNames
        print(f"Loaded {len(self.names)} known faces")

    def process_frame(self, frame):
        current_time = time.time()
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        detections = self.yolo_face(frame, verbose=False, conf=0.6)

        if len(detections) == 0:
            print("No faces detected")
        for detection in detections:
            boxes = detection.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                face_frame = frame[y1:y2, x1:x2]
                if current_time - self.last_recognition_time > self.recognition_pause_duration:
                    if self.recognition_thread is None or not self.recognition_thread.is_alive():
                        self.recognition_thread = Thread(target=self.recognize_face, args=(face_frame, current_time))
                        self.recognition_thread.start()

        return frame

    def recognize_face(self, face_frame, start_time):
        self.lock.acquire()
        self.last_recognition_time = start_time
        end_time = start_time + self.recognition_active_time
        current_time = time.time()

        while current_time < end_time:
            rgb_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face_frame)
            self.name = ""  # Reset name for each new recognition process
            if face_encodings:
                face_encoding = face_encodings[0]
                matches = face_recognition.compare_faces(self.encodeListKnown, face_encoding)
                if True in matches:
                    best_match_index = matches.index(True)
                    self.name = self.names[best_match_index]
                    print(f"Recognized: {self.name}")

            time.sleep(0.1)  # Simulate processing time
            current_time = time.time()
        self.lock.release()

    def cleanup(self):
        if self.recognition_thread is not None:
            self.recognition_thread.join()
        if self.fps_list:
            average_fps = sum(self.fps_list) / len(self.fps_list)
            print(f"Average FPS: {average_fps:.2f}")
