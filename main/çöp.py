import cv2
from djitellopy import Tello

def main():
    # Initialize the Tello drone
    tello = Tello()
    
    # Connect to the Tello drone
    tello.connect()
    
    # Start video stream
    tello.streamon()
    
    try:
        while True:
            # Get the frame from the Tello drone
            frame = tello.get_frame_read().frame
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the frame
            cv2.imshow('Tello Live Camera', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
