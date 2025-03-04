import os
import numpy as np
import cv2
from tqdm import tqdm

# Set the path to the images captured by the left and right cameras
pathL = "./data/stereoL/"
pathR = "./data/stereoR/"

print("Extracting image coordinates of respective 3D pattern ....\n")

# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

img_ptsL = []
img_ptsR = []
obj_pts = []

# Get sorted list of image files
filesL = sorted([f for f in os.listdir(pathL) if f.endswith(".png")])
filesR = sorted([f for f in os.listdir(pathR) if f.endswith(".png")])

# Ensure both directories have the same number of images
num_images = min(len(filesL), len(filesR))

for i in tqdm(range(num_images)):
    imgL = cv2.imread(os.path.join(pathL, filesL[i]))
    imgR = cv2.imread(os.path.join(pathR, filesR[i]))

    if imgL is None or imgR is None:
        print(f"Skipping {filesL[i]} or {filesR[i]} because it couldn't be loaded")
        continue

    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    outputL = imgL.copy()
    outputR = imgR.copy()

    retR, cornersR = cv2.findChessboardCorners(
        imgR_gray, (9, 6),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    retL, cornersL = cv2.findChessboardCorners(
        imgL_gray, (9, 6),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if retR and retL:
        obj_pts.append(objp)
        cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(outputR, (9, 6), cornersR, retR)
        cv2.drawChessboardCorners(outputL, (9, 6), cornersL, retL)
        cv2.imshow('cornersR', outputR)
        cv2.imshow('cornersL', outputL)
        cv2.waitKey(100)

        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)

        # Save the output images
        cv2.imwrite(f"./data/outputL/outputL_{i}.png", outputL)
        cv2.imwrite(f"./data/outputR/outputR_{i}.png", outputR)

cv2.destroyAllWindows()

if len(obj_pts) == 0:
    print("Error: No chessboard patterns detected in any images. Exiting...")
    exit()

print("Calculating left camera parameters ... ")
# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, imgL_gray.shape[::-1], None, None)
hL, wL = imgL_gray.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

print("Calculating right camera parameters ... ")
# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, imgR_gray.shape[::-1], None, None)
hR, wR = imgR_gray.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

print("Stereo calibration .....")
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
    obj_pts, img_ptsL, img_ptsR, new_mtxL, distL, new_mtxR, distR,
    imgL_gray.shape[::-1], criteria_stereo, flags)

rectify_scale = 1  # if 0 image cropped, if 1 image not cropped
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(
    new_mtxL, distL, new_mtxR, distR,
    imgL_gray.shape[::-1], Rot, Trns, rectify_scale, (0, 0))

Left_Stereo_Map = cv2.initUndistortRectifyMap(
    new_mtxL, distL, rect_l, proj_mat_l, imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(
    new_mtxR, distR, rect_r, proj_mat_r, imgR_gray.shape[::-1], cv2.CV_16SC2)

print("Saving parameters ......")
cv_file = cv2.FileStorage("data/params_py.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.write("Q", Q)
cv_file.release()

print("Calibration completed successfully!")
