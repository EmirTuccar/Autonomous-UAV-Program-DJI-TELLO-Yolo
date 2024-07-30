import cv2
from ultralytics import YOLO
import torch
import time

class SecondObjectDetection:
    def __init__(self):
        self.width = 960  # Resize width
        self.height = 720  # Resize height
        self.fps_list = []
        self.model = YOLO('D:\GradProje2\yolov8nvis.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
        

    def process_frame(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        detections = self.model(frame, verbose=False)

        for detection in detections:
            boxes = detection.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        return frame

    def cleanup(self):
        average_fps = sum(self.fps_list) / len(self.fps_list) if self.fps_list else 0
        print(f"Average FPS: {average_fps:.2f}")
