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
CamL_id = 1  
CamR_id = 2  

CamL = cv2.VideoCapture(CamL_id, cv2.CAP_DSHOW)
CamR = cv2.VideoCapture(CamR_id, cv2.CAP_DSHOW)

if not CamL.isOpened() or not CamR.isOpened():
    print("Error: One or both cameras could not be opened!")
    exit(-1)

# Set resolution
CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)  

T = 10  
count = 0  # to sync two cameras while taking imgs
start = time.time()  

def capture_frames(cam, cam_name, save_path, display_name, is_left_camera):
    global start, count
    while True:
        ret, frame = cam.read()
        if not ret:
            print(f"Error: Could not read frame from {cam_name}.")
            break

        timer = T - int(time.time() - start)

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Time: {timer}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(display_name, display_frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret_corners or timer <= 0:
            count += 1  
            filename = f'img{count}.png'
            cv2.imwrite(os.path.join(save_path, filename), frame)  
            print(f"Image saved: {filename} in {save_path}")

            if timer <= 0:  
                start = time.time()  # Reset timer only if it fully runs out

        if cv2.waitKey(1) & 0xFF == 27:
            break

# Start threads
threadL = threading.Thread(target=capture_frames, args=(CamL, "Left Camera", stereoL_path, "Left Camera", True))
threadR = threading.Thread(target=capture_frames, args=(CamR, "Right Camera", stereoR_path, "Right Camera", False))

threadL.start()
threadR.start()

threadL.join()
threadR.join()

CamL.release()
CamR.release()
cv2.destroyAllWindows()
