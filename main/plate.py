from ultralytics import YOLO
import cv2
import torch
import time
from utils_turkish import read_license_plate, write_csv

class IntegratedObjectDetection:
    
    def __init__(self):
        self.plate_model = YOLO('D:/GradProje2/pale_n.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.fps_list = []
        self.best_plate_text = ''  # Initializing attribute
        self.best_plate_text_score = 0  # Initializing attribute
        self.best_plate_crop = None  # Initializing attribute
        self.best_plate_timestamp = time.time()
        self.frame_nmr = 0  # Add a frame number attribute
        self.output_csv_path = "D:/GradProje2/output.csv"
        
    def process_frames(self, frame):
        self.frame_nmr += 1  # Increment frame number
        current_time = time.time()
        current_best_crop = None  # Initialize to None each call
        plate_detections = self.plate_model(frame, verbose=False)[0]
        
        if current_time - self.best_plate_timestamp > 3:  # 3 seconds timeout
            self.best_plate_text = ''  # Reset the best plate text
            self.best_plate_text_score = 0  # Reset the best plate score
            self.best_plate_crop = None  # Reset the best plate crop
        
        for detection in plate_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            cropppp = frame[int(y1)-100:int(y2)+100, int(x1)-100:int(x2)+100]
            plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            plate_crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            _, plate_crop_thresh = cv2.threshold(plate_crop_gray, 50, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            license_plate_text, license_plate_text_score = read_license_plate(plate_crop_thresh)

            if license_plate_text and license_plate_text_score > self.best_plate_text_score:
                self.best_plate_text = license_plate_text
                self.best_plate_text_score = license_plate_text_score
                self.best_plate_crop = cropppp
                current_best_crop = cropppp  # Update local variable with new best crop

                # Update results dictionary
                self.results[self.frame_nmr] = {
                    'car_bbox': [x1, y1, x2, y2],
                    'license_plate_bbox': [x1, y1, x2, y2],
                    'license_plate_bbox_score': score,
                    'license_number': license_plate_text,
                    'license_number_score': license_plate_text_score
                }

        if current_best_crop is not None:
            self.best_plate_timestamp = current_time
        
        
        
        return current_best_crop if current_best_crop is not None else self.best_plate_crop  # Return the best new crop or the last best crop
    
        

    def show_frames(self):
        while True:
            frame = self.process_frames()

            if frame is not None:
                cv2.imshow('Detected Plate', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()
                   
    def save_results(self):
        write_csv(self.results, self.output_csv_path)    
         
    def cleanup(self):
        average_fps = sum(self.fps_list) / len(self.fps_list) if self.fps_list else 0
        print(f"Average FPS: {average_fps:.2f}")
