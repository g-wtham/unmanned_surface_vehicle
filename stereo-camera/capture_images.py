import numpy as np
import cv2
import time
import os
import threading

output_path = "./data/"
stereoL_path = os.path.join(output_path, "stereoL")
stereoR_path = os.path.join(output_path, "stereoR")
os.makedirs(stereoL_path, exist_ok=True)
os.makedirs(stereoR_path, exist_ok=True)

# Camera IDs
CamL_id = 0
CamR_id = 1

CamL = cv2.VideoCapture(CamL_id, cv2.CAP_DSHOW)
CamR = cv2.VideoCapture(CamR_id, cv2.CAP_DSHOW)

if not CamL.isOpened() or not CamR.isOpened():
    print("Error: One or both cameras could not be opened!")
    CamL.release()
    CamR.release()
    exit(-1)

# Set resolution
CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)  # Allow cameras to adjust

INITIAL_WAIT = 10  # Initial wait before capturing
CAPTURE_INTERVAL = 5  # Capture every 5 seconds
CHECKERBOARD_SIZE = (9, 6)  # Adjust if needed
count = 0

start_time = time.time()

def capture_frames(cam, cam_name, save_path, display_name):
    global start_time, count

    # **Initial 10-sec delay**
    while time.time() - start_time < INITIAL_WAIT:
        ret, frame = cam.read()
        if not ret:
            print(f"Error: Could not read frame from {cam_name}.")
            break

        remaining_time = int(INITIAL_WAIT - (time.time() - start_time))
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Starting in: {remaining_time}s", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(display_name, display_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit if ESC is pressed
            return

    print(f"Starting image capture for {cam_name}...")

    while True:
        ret, frame = cam.read()
        if not ret:
            print(f"Error: Could not read frame from {cam_name}.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

        elapsed_time = time.time() - start_time
        remaining_time = CAPTURE_INTERVAL - (elapsed_time % CAPTURE_INTERVAL)

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Next capture: {int(remaining_time)}s", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # **Only capture if checkerboard is detected**
        if ret_corners and int(elapsed_time) % CAPTURE_INTERVAL == 0:
            count += 1
            filename = f'img{count}.png'
            cv2.imwrite(os.path.join(save_path, filename), frame)
            print(f"Checkerboard detected! Image saved: {filename} in {save_path}")
            time.sleep(1)  # Small delay to avoid multiple captures in the same second

        cv2.imshow(display_name, display_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit if ESC is pressed
            break

# Start threads
threadL = threading.Thread(target=capture_frames, args=(CamL, "Left Camera", stereoL_path, "Left Camera"))
threadR = threading.Thread(target=capture_frames, args=(CamR, "Right Camera", stereoR_path, "Right Camera"))

threadL.start()
threadR.start()

threadL.join()
threadR.join()

CamL.release()
CamR.release()
cv2.destroyAllWindows()
