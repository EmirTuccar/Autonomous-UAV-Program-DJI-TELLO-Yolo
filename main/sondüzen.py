import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QLineEdit, QFrame, QSizePolicy, QSpacerItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from face import FaceDetection
from car import SecondObjectDetection
from plate import IntegratedObjectDetection
from trackerKCF import RealTimeTracker
from tello import TelloController
import sys

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        

        # Initialize detection objects
        self.face_detector = FaceDetection()
        self.yolo_detector = SecondObjectDetection()
        self.plate_detector = IntegratedObjectDetection()
        self.tracker = RealTimeTracker()
        self.tello_controller = TelloController()
        
        self.initUI()
        # Timer setup for updating the license plate detection
        self.activate_detection_mode()

        self.best_plate = None
        self.best_plate_score = 0.0
        self.best_plate_text = ""
        self.best_plate_text_score = 0.0
        self.output_path = "D:/GradProje/plate_detection_results.csv"
        
        

    def initUI(self):
        self.setWindowTitle('Live Video Feed')
        self.setGeometry(50, 50, 1640, 800)
        self.setFixedSize(1640, 800)

        main_layout = QVBoxLayout()

        top_frame = QFrame()
        middle_frame = QFrame()
        bottom_frame = QFrame()

        top_frame.setFrameShape(QFrame.Box)
        middle_frame.setFrameShape(QFrame.Box)
        bottom_frame.setFrameShape(QFrame.Box)

        top_layout = QHBoxLayout()
        middle_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()

        left_top_frame = QFrame()
        center_top_frame = QFrame()
        right_top_frame = QFrame()

        left_top_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        center_top_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_top_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        left_top_layout = QHBoxLayout()
        center_top_layout = QHBoxLayout()
        right_top_layout = QHBoxLayout()

        left_top_frame.setLayout(left_top_layout)
        center_top_frame.setLayout(center_top_layout)
        right_top_frame.setLayout(right_top_layout)

        top_layout.addWidget(left_top_frame, 1)
        top_layout.addWidget(center_top_frame, 1)
        top_layout.addWidget(right_top_frame, 1)

        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setContentsMargins(0, 0, 10, 0)
        left_top_layout.addWidget(self.video_label)

        self.second_video_label = QLabel(self)
        self.second_video_label.setFixedSize(520, 400)
        self.second_video_label.setContentsMargins(0, 0, 10, 0)
        center_top_layout.addWidget(self.second_video_label)

        self.third_video_label = QLabel(self)
        self.third_video_label.setFixedSize(520, 400)
        self.third_video_label.setContentsMargins(0, 0, 10, 0)
        right_top_layout.addWidget(self.third_video_label)

        left_middle_frame = QFrame()
        center_middle_frame = QFrame()
        right_middle_frame = QFrame()

        left_middle_frame.setFrameShape(QFrame.StyledPanel)
        center_middle_frame.setFrameShape(QFrame.StyledPanel)
        right_middle_frame.setFrameShape(QFrame.StyledPanel)

        left_middle_layout = QHBoxLayout()
        center_middle_layout = QHBoxLayout()
        right_middle_layout = QHBoxLayout()

        left_middle_frame.setLayout(left_middle_layout)
        center_middle_frame.setLayout(center_middle_layout)
        right_middle_frame.setLayout(right_middle_layout)

        middle_layout.addWidget(left_middle_frame, 1)
        middle_layout.addWidget(center_middle_frame, 1)
        middle_layout.addWidget(right_middle_frame, 1)

        self.left_input = QLineEdit(self)
        self.left_input.setAlignment(Qt.AlignCenter)
        self.left_input.setFixedSize(520, 40)
        left_middle_layout.addWidget(self.left_input)
        self.left_input.setContentsMargins(0, 10, 10, 0)

        self.middle_input = QLineEdit(self)
        self.middle_input.setAlignment(Qt.AlignCenter)
        self.middle_input.setFixedSize(520, 40)
        center_middle_layout.addWidget(self.middle_input)
        self.middle_input.setContentsMargins(0, 10, 10, 0)

        self.right_input = QLineEdit(self)
        self.right_input.setAlignment(Qt.AlignCenter)
        self.right_input.setFixedSize(520, 40)
        right_middle_layout.addWidget(self.right_input)
        self.right_input.setContentsMargins(0, 10, 10, 0)

        

        first_inner_layout = QVBoxLayout()
        second_inner_layout = QHBoxLayout()
        third_inner_layout = QVBoxLayout()

        first_inner_bottom_frame = QFrame()
        second_inner_bottom_frame = QFrame()
        third_inner_bottom_frame = QFrame()

        first_inner_bottom_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        second_inner_bottom_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        third_inner_bottom_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        first_inner_bottom_frame.setLayout(first_inner_layout)
        second_inner_bottom_frame.setLayout(second_inner_layout)
        third_inner_bottom_frame.setLayout(third_inner_layout)

        bottom_layout.addWidget(first_inner_bottom_frame, 1)
        bottom_layout.addWidget(second_inner_bottom_frame, 1)
        bottom_layout.addWidget(third_inner_bottom_frame, 1)

        first_row_layout = QHBoxLayout()
        second_row_layout = QHBoxLayout()

        first_row_buttons = ["Activate Tracking Mode", "Activate Object Detection"]
        for name in first_row_buttons:
            button = QPushButton(name)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            first_row_layout.addWidget(button)
            if name == "Activate Tracking Mode":
                button.clicked.connect(self.activate_tracking_mode)
            elif name == "Activate Object Detection":
                button.clicked.connect(self.activate_detection_mode)

        second_row_buttons = ["Manual Mode", "Path Tracking", "Stop Path Tracking"]
        for name in second_row_buttons:
            button = QPushButton(name)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            second_row_layout.addWidget(button)
            if name == "Manual Mode":
                button.clicked.connect(self.activate_manual_mode)
            elif name == "Path Tracking":
                button.clicked.connect(self.start_path_tracking)
            elif name == "Stop Path Tracking":
                button.clicked.connect(self.stop_path_tracking)

        first_inner_layout.addLayout(first_row_layout)
        first_inner_layout.addLayout(second_row_layout)

        row1_layout = QHBoxLayout()
        row2_layout = QHBoxLayout()

        buttons_row1 = ["Down", "Forward", "Up"]
        for name in buttons_row1:
            button = QPushButton(name)
            row1_layout.addWidget(button)
            if name == "Up":
                button.clicked.connect(self.tello_controller.move_up)
            elif name == "Forward":
                button.clicked.connect(self.tello_controller.move_forward)
            elif name == "Back":
                button.clicked.connect(self.tello_controller.move_back)

        buttons_row2 = ["Left", "Back", "Right"]
        for name in buttons_row2:
            button = QPushButton(name)
            row2_layout.addWidget(button)
            if name == "Left":
                button.clicked.connect(self.tello_controller.move_left)
            elif name == "Down":
                button.clicked.connect(self.tello_controller.move_down)
            elif name == "Right":
                button.clicked.connect(self.tello_controller.move_right)

        takeoff_button = QPushButton("Take Off")
        land_button = QPushButton("Land")
        second_inner_layout.addWidget(takeoff_button)
        second_inner_layout.addWidget(land_button)

        takeoff_button.clicked.connect(self.tello_controller.takeoff)
        land_button.clicked.connect(self.tello_controller.land)

        third_inner_layout.addLayout(row1_layout)
        third_inner_layout.addLayout(row2_layout)

        top_frame.setLayout(top_layout)
        middle_frame.setLayout(middle_layout)
        bottom_frame.setLayout(bottom_layout)

        main_layout.addWidget(top_frame, 3)
        main_layout.addWidget(middle_frame, 1)
        main_layout.addWidget(bottom_frame, 1)

        self.setLayout(main_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

        self.timer_yolo = QTimer(self)
        self.timer_yolo.timeout.connect(self.update_yolo_frame)
        self.timer_yolo.start(10)

        self.timer_plate = QTimer(self)
        self.timer_plate.timeout.connect(self.update_plate_frame)
        self.timer_plate.start(10)

    def activate_tracking_mode(self):
        print("Tracking mode activated!")
        self.tracking_mode = True
        self.tello_controller.stop_path_tracking()
        self.tello_controller.stop_manual_mode()
        self.update_mode_label()

    def activate_detection_mode(self):
        print("Detection mode activated!")
        self.tracking_mode = False
        self.tello_controller.stop_path_tracking()
        self.tello_controller.set_manual_mode()
        self.update_mode_label()

    def activate_manual_mode(self):
        print("Manual mode activated!")
        self.tello_controller.set_manual_mode()
        self.tello_controller.stop_path_tracking()
        self.tracking_mode = False
        self.update_mode_label()

    def start_path_tracking(self):
        self.tello_controller.start_path_tracking()
        self.tracking_mode = False
        self.update_mode_label()

    def stop_path_tracking(self):
        self.tello_controller.stop_path_tracking()
        self.update_mode_label()

    def update_mode_label(self):
        if self.tello_controller.manual_mode:
            self.middle_input.setText("Manual Mode Activated")
        elif self.tracking_mode:
            self.middle_input.setText("Tracking Mode Activated")
        elif self.tello_controller.tracking_active:
            self.middle_input.setText("Path Tracking Mode Activated")
        else:
            self.middle_input.setText("Detection Mode Activated")

    def update_yolo_frame(self):
        frame = self.tello_controller.get_frame()
        if self.tracking_mode:
            frame = self.tracker.process_frame(frame, self.tello_controller)
        else:
            frame = self.yolo_detector.process_frame(frame)
        
        if frame is not None:
            self.display_frame(frame, self.second_video_label)

    def update_plate_frame(self):
        frame = self.tello_controller.get_frame()
        crop = self.plate_detector.process_frames(frame)
        if crop is not None and self.plate_detector.best_plate_text:
            self.display_image(crop)
            self.right_input.setText(self.plate_detector.best_plate_text)
        

    def display_image(self, img):
        if img is None or img.size == 0:
            print("Error: The image is empty or None")
            return
        
        try:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_Qt_format).scaled(540, 400, Qt.KeepAspectRatio)
            self.third_video_label.setPixmap(pixmap)
        except cv2.error as e:
            print("Error during cv2.cvtColor:", e)
            return


        
    def update_frame(self):
        frame = self.tello_controller.get_frame()
        frame = self.face_detector.process_frame(frame)
        if frame is not None:
            self.display_frame(frame, self.video_label)
        self.left_input.setText(self.face_detector.name)
        
    def display_frame(self, frame, label):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(p)

    def closeEvent(self, event):
        self.plate_detector.save_results()
        self.tello_controller.land()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoWindow()
    ex.show()
    sys.exit(app.exec_())
