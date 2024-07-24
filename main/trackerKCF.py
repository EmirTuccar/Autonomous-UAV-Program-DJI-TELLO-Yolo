import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utils_turkish import get_car, read_license_plate, write_csv
from pid import PID

class RealTimeTracker:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.coco_model = YOLO("D:/GradProje2/yolov8nvis.pt").to(self.device)
        self.license_plate_detector = YOLO("D:/GradProje2/pale_n.pt").to(self.device)

        self.results = {}
        self.trackers = {}
        self.frame_nmr = -1

        
        self.tracking_car = False

        # Initialize PID controllers
        self.pan_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.tilt_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.pan_pid.initialize()
        self.tilt_pid.initialize()

    def euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def relative_position(self, center_x, center_y, frame_width, frame_height):
        third_width = frame_width // 3
        third_height = frame_height // 3

        horizontal_region = (center_x // third_width) + 1
        vertical_region = (center_y // third_height) + 1

        if vertical_region == 2 and horizontal_region == 2:
            return 'center'
        else:
            vertical_pos = {1: 'up', 2: 'center', 3: 'down'}
            horizontal_pos = {1: 'left', 2: 'center', 3: 'right'}

            position = f"{vertical_pos[vertical_region]} {horizontal_pos[horizontal_region]}".strip()
            return position.replace("center center", "center")

    def draw_grid(self, frame):
        height, width = frame.shape[:2]
        third_width = width // 3
        third_height = height // 3
        for i in range(1, 3):
            cv2.line(frame, (0, third_height * i), (width, third_height * i), (255, 255, 255), 1)
            cv2.line(frame, (third_width * i, 0), (third_width * i, height), (255, 255, 255), 1)
        # Draw right-most vertical line and bottom-most horizontal line to complete the grid
        cv2.line(frame, (third_width * 2, 0), (third_width * 2, height), (255, 255, 255), 1)
        cv2.line(frame, (0, third_height * 2), (width, third_height * 2), (255, 255, 255), 1)

    def process_frame(self, frame, tello_controller):
        self.frame_nmr += 1
        self.results[self.frame_nmr] = {}

        height, width = frame.shape[:2]
        self.dot_x, self.dot_y = width // 2, height // 2

        if not self.tracking_car:
            detections = self.coco_model(frame, verbose=False)[0]
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                car_bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

                if class_id == 4:  # Class ID for vehicles
                    car_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                    license_plates = self.license_plate_detector(car_frame, verbose=False)[0]
                    for license_plate in license_plates.boxes.data.tolist():
                        lp_x1, lp_y1, lp_x2, lp_y2, lp_score, lp_class_id = license_plate

                        lp_x1 += x1
                        lp_y1 += y1
                        lp_x2 += x1
                        lp_y2 += y1

                        if x1 <= lp_x1 <= x2 and y1 <= lp_y1 <= y2 and x1 <= lp_x2 <= x2 and y1 <= lp_y2 <= y2:
                            tracker = cv2.TrackerKCF_create()
                            tracker.init(frame, tuple(car_bbox))
                            self.trackers[self.frame_nmr] = tracker
                            self.tracking_car = True

                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            license_plate_crop = frame[int(lp_y1):int(lp_y2), int(lp_x1):int(lp_x2)]
                            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                            print(license_plate_text)
                            if license_plate_text is not None:
                                self.results[self.frame_nmr][car_bbox] = {
                                    'car': {'bbox': [x1, y1, x2, y2]},
                                    'license_plate': {
                                        'bbox': [lp_x1, lp_y1, lp_x2, lp_y2],
                                        'text': license_plate_text,
                                        'bbox_score': lp_score,
                                        'text_score': license_plate_text_score
                                    }
                                }
                                cv2.rectangle(frame, (int(lp_x1), int(lp_y1)), (int(lp_x2), int(lp_y2)), (255, 0, 0), 2)
                                cv2.putText(frame, license_plate_text, (int(lp_x1), int(lp_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                            
                            break
                    if self.tracking_car:
                        break

                if class_id == 2:  # Class ID for humans
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, tuple(car_bbox))
                    self.trackers[self.frame_nmr] = tracker
                    self.tracking_car = True
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    break

        for tracker_id, tracker in list(self.trackers.items()):
            success, bbox = tracker.update(frame)
            if success:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                center_x = int(bbox[0] + bbox[2] / 2)
                center_y = int(bbox[1] + bbox[3] / 2)
                cv2.line(frame, (self.dot_x, self.dot_y), (center_x, center_y), (0, 255, 255), 2)

                position = self.relative_position(center_x, center_y, width, height)
                cv2.putText(frame, f"Position: {position}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                pan_error = self.dot_x - center_x
                pan_update = self.pan_pid.update(pan_error, sleep=0)

                tilt_error = self.dot_y - center_y
                tilt_update = self.tilt_pid.update(tilt_error, sleep=0)

                pan_update = max(min(pan_update, 40), -40)
                tilt_update = max(min(tilt_update, 40), -40)

                # Adjusting the pan_update for yaw and tilt_update for forward/backward movement
                yaw_velocity = int(pan_update * -1)
                forward_backward_velocity = int(tilt_update)

                print(yaw_velocity, forward_backward_velocity)
                tello_controller.send_rc_controler(0, forward_backward_velocity // 2, 0, yaw_velocity // 3)

            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                self.tracking_car = False
                del self.trackers[tracker_id]

        self.draw_grid(frame)
        return frame


    def run(self):
        cap = cv2.VideoCapture("D:/GradProje2/ex.mp4")
        #tello_controller = TelloController()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            #frame = self.process_frame(frame, tello_controller)

            cv2.imshow("Real-Time Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
       



if __name__ == "__main__":
    detector = RealTimeTracker()
    detector.run()
