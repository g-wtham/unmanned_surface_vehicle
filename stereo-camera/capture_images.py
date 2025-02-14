import numpy as np
import cv2
import time
import os

# Create output directories if they don't exist
output_path = "./data/"
stereoL_path = os.path.join(output_path, "stereoL")
stereoR_path = os.path.join(output_path, "stereoR")
os.makedirs(stereoL_path, exist_ok=True)
os.makedirs(stereoR_path, exist_ok=True)

print("Checking the right and left camera IDs:")
print("Press (y) if IDs are correct and (n) to swap the IDs")
print("Press enter to start the process >> ")
input()

CamL_id = 0
CamR_id = 1

CamL = cv2.VideoCapture(CamL_id)
CamR = cv2.VideoCapture(CamR_id)

if not CamL.isOpened() or not CamR.isOpened():
    print("Error: One or both cameras could not be opened!")
    exit(-1)

for i in range(10):
    retL, frameL = CamL.read()
    retR, frameR = CamR.read()

    if not retL or not retR:
        print("Error: Could not read frames from cameras.")
        exit(-1)

cv2.imshow('imgL', frameL)
cv2.imshow('imgR', frameR)

key = cv2.waitKey(0) & 0xFF
if key == ord('y') or key == ord('Y'):
    print("Camera IDs maintained")
elif key == ord('n') or key == ord('N'):
    CamL_id, CamR_id = CamR_id, CamL_id
    print("Camera IDs swapped")
else:
    print("Wrong input response")
    exit(-1)

CamR.release()
CamL.release()

CamL = cv2.VideoCapture(CamL_id)
CamR = cv2.VideoCapture(CamR_id)

if not CamL.isOpened() or not CamR.isOpened():
    print("Error: One or both cameras could not be reopened after swapping.")
    exit(-1)

start = time.time()
T = 10  
count = 0

while True:
    timer = T - int(time.time() - start)
    
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    if not retR or not retL:
        print("Error: Unable to fetch frames from cameras.")
        break

    img1_temp = frameL.copy()
    cv2.putText(img1_temp, "%r" % timer, (50, 50), 1, 5, (55, 0, 0), 5)
    cv2.imshow('imgR', frameR)
    cv2.imshow('imgL', img1_temp)

    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    retCornersR, cornersR = cv2.findChessboardCorners(grayR, (9, 6), None)
    retCornersL, cornersL = cv2.findChessboardCorners(grayL, (9, 6), None)

    # Save images if chessboard corners are detected or timer hits 0
    if (retCornersR and retCornersL) or timer <= 0:
        count += 1
        cv2.imwrite(os.path.join(stereoR_path, f'img{count}.png'), frameR)
        cv2.imwrite(os.path.join(stereoL_path, f'img{count}.png'), frameL)
        print(f"Images saved: img{count}.png in stereoL and stereoR")

        start = time.time()  

    if cv2.waitKey(1) & 0xFF == 27:
        print("Closing the cameras!")
        break

CamR.release()
CamL.release()
cv2.destroyAllWindows()
