import cv2
import time
from ultralytics import YOLO
import torch

class SecondObjectDetection:
    def __init__(self, cap):
        self.cap = cap
        self.width = 960
        self.height = 800
        self.fps_list = []
        self.model = None  

    
        self.model = YOLO("\directory\").to('cuda' if torch.cuda.is_available() else 'cpu')
        

    def process_frame(self):
        if self.model is None:
            print("Model is not loaded. Please call load_model() first.")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return None  

        # Detect objects with the model
        detections = self.model(frame, verbose=False)
        for detection in detections:
            boxes = detection.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        return frame 

    def process_fe(self):
        if self.model is None:
            print("Model is not loaded. Please call load_model() first.")
            return
        
        prev_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                return None  

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            self.fps_list.append(fps)
            
            prev_time = current_time

            
            frame_resized = cv2.resize(frame, (960, 800))
            detections = self.model(frame_resized, verbose=False, conf=0.5)
            for detection in detections:
                boxes = detection.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 255), 2)

            cv2.imshow("Object Detection", frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        average_fps = sum(self.fps_list) / len(self.fps_list) if self.fps_list else 0
        print(f"Average FPS: {average_fps:.2f}")

# To run this code
if __name__ == "__main__":
    cap = cv2.VideoCapture("\directory\ex.mp4")

    fr = SecondObjectDetection(cap)
    
    fr.process_fe()
    fr.cleanup()
