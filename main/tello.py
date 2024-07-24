import threading
from djitellopy import Tello
import time
import cv2


import numpy as np
from djitellopy import Tello

import imutils
class TelloController:
    
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.frame_read = self.tello.get_frame_read()
        self.max_speed_limit = 40

   
        
        
    def send_rc_controler(self, left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity):
        self.tello.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
    
        

    
        
    def get_frame(self):
        frame = self.tello.get_frame_read().frame
        #frame = self.cap = cv2.VideoCapture("D:\GradProje2/video.mp4")
        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def takeoff(self):
        self.tello.takeoff()

    def land(self):

        self.tello.land()

    
    def move_forward(self):
        self.tello.send_rc_control(0, 20, 0, 0)
    def move_back(self):
        self.tello.send_rc_control(0, -20, 0, 0)
        
    def move_up(self):
        self.tello.move_up(20)

    def move_down(self):
        self.tello.move_down(20)

    def move_left(self):
        self.tello.rotate_clockwise(90)

    def move_right(self):
        self.tello.rotate_counter_clockwise(90)

    def hover(self):
        self.tello.send_rc_control(0, 0, 0, 0)

    def rotate_180(self):
        self.tello.rotate_clockwise(180)

    def move_up_left(self):
        self.tello.send_rc_control(-20, 20, 0, 0)

    def move_up_right(self):
        self.tello.send_rc_control(20, 20, 0, 0)

    def move_down_left(self):
        self.tello.send_rc_control(-20, -20, 0, 0)

    def move_down_right(self):
        self.tello.send_rc_control(20, -20, 0, 0)

    def start_path_tracking(self):
        
        self.tracking_active = True
        self.manual_mode = False


    def stop_path_tracking(self):
        self.tracking_active = False


    def _path_tracking(self):
        while self.tracking_active:
            for i in range(12):
                
                self.tello.move_forward(30)  # Move forward for each side
                time.sleep(2)
                
                self.tello.rotate_clockwise(30)  # Rotate 30 degrees for each step in a 12-sided path
                time.sleep(2)
                i =  i + 1

    def set_manual_mode(self):
        self.manual_mode = True
        self.stop_path_tracking()
        self.tracking_active = False
        
    def stop_manual_mode(self):
        self.manual_mode = False
 


