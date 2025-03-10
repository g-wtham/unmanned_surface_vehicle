'''
1.  Memory Usage Tracking   
   - Uses `psutil` to check the process's memory usage in MB.  
   - Prints memory usage for each frame.

2.  YOLO Object Detection   
   - Loads a custom-trained `best.pt` model for detecting plastic bottles.  
   - Runs inference on each frame and retrieves bounding box coordinates, confidence, and class ID.

3.  Depth Estimation with MiDaS   
   - Loads a MiDaS depth estimation model (`model-small.onnx`).  
   - Converts the frame to RGB and prepares input for depth estimation.  
   - Normalizes and resizes the depth map to match the frame size.

4.  Depth-Based Distance Calculation   
   - Extracts depth at the center of detected objects.  
   - Uses a linear function to approximate real-world distance.  
   - Displays the estimated distance along with the object label.

5.  Visualization   
   - Draws bounding boxes and labels detected objects with their class and estimated distance.  
   - Displays FPS on the output frame.  
   - Shows both the detection output and the depth map in separate windows.

6.  Performance Optimization Considerations   
   - Uses `cv2.dnn.blobFromImage` to preprocess depth model input efficiently.  
   - Converts `results[0].boxes.data.cpu().numpy()` to avoid GPU memory leaks.  
   - Resizes output frames for better visualization performance.

7.  Termination and Cleanup   
   - Allows exiting with the ESC key.  
   - Releases video capture and destroys all OpenCV windows.

'''


import time
import cv2
import numpy as np
from ultralytics import YOLO
import psutil

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

yolo_model = YOLO("./plastic_detection_FloW.pt")
class_names = yolo_model.names
mono_model = cv2.dnn.readNet("./model-small.onnx")

def depth_to_distance(depth) -> float:
    return -2 * depth + 2

cap = cv2.VideoCapture('./test-videos/video5.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    print(f"Memory Usage: {get_memory_usage():.2f} MB")
    height, width, _ = frame.shape
    start_time = time.time()
    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(rgb_frame, 1/255., (256, 256), (123.675, 116.28, 103.53), True, False)
    mono_model.setInput(blob)
    depth_map = mono_model.forward()[0, :, :]
    depth_map = cv2.resize(depth_map, (width, height))
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        center_x = np.clip(center_x, 0, width - 1)
        center_y = np.clip(center_y, 0, height - 1)
        depth = depth_map[center_y, center_x]
        distance = depth_to_distance(depth)
        label = f"{class_names[cls_id]} {round(distance * 100, 2)} cm"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    scaled_frame = cv2.resize(frame, (1280, 720))
    scaled_depth_map = cv2.resize(depth_map, (640, 360))
    cv2.imshow("Object Detection & Distance Estimation", scaled_frame)
    cv2.imshow("Depth Map", scaled_depth_map)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()