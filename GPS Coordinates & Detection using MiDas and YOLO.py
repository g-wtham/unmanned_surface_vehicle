import time
import cv2
import numpy as np
from ultralytics import YOLO
import psutil
import math # Import math module for trigonometric functions

# --- Constants ---
CAMERA_HFOV = 55.0 # degrees (Logitech C270)
CAMERA_VFOV = 35.0 # degrees (Approximate for C270 - Check datasheet if precise vertical angle needed)

# Performance Optimization Settings
PROCESS_EVERY_N_FRAMES = 3 # Process 1 out of every 3 frames (adjust as needed)
PROCESSING_WIDTH = 640 # Process frames at this width (adjust based on Pi performance)
PROCESSING_HEIGHT = 480 # Process frames at this height

# Detection Settings
CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence to consider a detection

# --- Functions ---
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # Return in MB

def depth_to_distance(depth_value, method='linear_simple'):
    """
    Converts normalized depth value (0-1) to estimated distance in meters.
    NEEDS CALIBRATION BASED ON REAL-WORLD TESTS!
    Args:
        depth_value: Normalized depth value (closer typically closer to 1).
        method: The calibration method to use.
    Returns:
        Estimated distance in meters, or float('inf') if invalid.
    """
    if method == 'linear_simple':
        # Simple inverse linear relationship: depth=1 -> near, depth=0 -> far
        # Example: Max range 5m, min range 0.3m
        MAX_DIST = 5.0
        MIN_DIST = 0.3
        # Ensure depth_value is reasonable
        if 0 <= depth_value <= 1:
             # Map depth (0..1) to distance (MAX_DIST..MIN_DIST)
             distance = MAX_DIST - depth_value * (MAX_DIST - MIN_DIST)
             # Clamp distance to valid range
             return max(MIN_DIST, distance)
        else:
             return float('inf') # Invalid depth

    elif method == 'inverse_scaled':
         # Inverse relationship (common for depth): distance = k / depth_value
         # Needs calibration to find 'k'
         k = 0.5 # PLACEHOLDER - CALIBRATE THIS CONSTANT
         if depth_value > 1e-4: # Avoid division by small numbers/zero
             return k / depth_value
         else:
             return float('inf')

    # Add more calibration methods here if needed

    # Default if method unknown or calibration needed
    # Return raw depth for debugging if no calibration done
    # return depth_value # Uncomment for debugging raw depth
    print("WARNING: depth_to_distance needs calibration!")
    return -2 * depth_value + 2 # Keep previous example as a fallback placeholder


# --- Load Models ---
print("Loading YOLO model...")
try:
    yolo_model = YOLO("./plastic_detection_FloW.pt") # Use your trained model
    class_names = yolo_model.names
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

print("Loading MiDaS depth model...")
try:
    # Using ONNX for potentially better CPU performance
    mono_model = cv2.dnn.readNet("./model-small.onnx")
except Exception as e:
    print(f"Error loading MiDaS model: {e}")
    exit()
print("Models loaded.")

# --- Initialize Video Capture ---
capture_index = 0
cap = cv2.VideoCapture(capture_index)

if not cap.isOpened():
    print(f"Error: Could not open video source at index {capture_index}.")
    exit()

# Try setting capture resolution closer to processing resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESSING_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESSING_HEIGHT)
# Read back actual resolution
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera requested {PROCESSING_WIDTH}x{PROCESSING_HEIGHT}, got {actual_width}x{actual_height}")

# Use the actual dimensions obtained from the camera for calculations
WIDTH = actual_width
HEIGHT = actual_height
image_center_x = WIDTH / 2.0
image_center_y = HEIGHT / 2.0 # Needed if calculating vertical angle

print("Starting video capture...")
frame_count = 0
last_detection_data = None # Store data from the last processed frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to capture frame. Retrying...")
        time.sleep(0.1)
        continue

    frame_count += 1
    start_time = time.time() # Start timer for FPS calculation

    # --- Process Frame Conditionally ---
    current_detection_data = None # Reset for this loop iteration
    processed_this_frame = False

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        processed_this_frame = True
        # --- Preprocessing ---
        # Ensure frame is the correct size for processing
        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            process_frame = cv2.resize(frame, (WIDTH, HEIGHT))
        else:
            process_frame = frame

        # --- YOLO Detection ---
        results = yolo_model(process_frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detections = results[0].boxes.data.cpu().numpy()

        # --- Depth Estimation (only if detections exist) ---
        if len(detections) > 0:
            rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            blob = cv2.dnn.blobFromImage(rgb_frame, 1/255., (256, 256), (123.675, 116.28, 103.53), swapRB=True, crop=False)
            mono_model.setInput(blob)
            depth_map = mono_model.forward().squeeze()
            depth_map = cv2.resize(depth_map, (WIDTH, HEIGHT))
            depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            depth_map = None # No need for depth if no objects detected


        # --- Process Detections ---
        closest_detection = None
        min_distance = float('inf')

        if depth_map is not None: # Ensure depth map was calculated
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                # Confidence already filtered by yolo_model call, but double-check if needed

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cls_id = int(cls_id)
                try:
                    object_class = class_names[cls_id]
                except IndexError: continue # Skip if class ID is invalid

                object_center_x = (x1 + x2) / 2.0
                object_center_y = (y1 + y2) / 2.0

                bounded_center_x = int(np.clip(object_center_x, 0, WIDTH - 1))
                bounded_center_y = int(np.clip(object_center_y, 0, HEIGHT - 1))

                # --- Calculate Bearing Angle ---
                pixel_deviation_x = object_center_x - image_center_x
                bearing_angle = (pixel_deviation_x * CAMERA_HFOV) / WIDTH # Degrees

                # --- Calculate Distance ---
                depth_value = depth_map[bounded_center_y, bounded_center_x]
                distance = depth_to_distance(depth_value, method='linear_simple') # CALL YOUR CALIBRATED FUNCTION

                # --- Calculate Coordinates (X, Y relative to USV) ---
                if distance != float('inf'):
                    angle_radians = math.radians(bearing_angle)
                    coord_x = distance * math.sin(angle_radians) # X = d * sin(theta)
                    coord_y = distance * math.cos(angle_radians) # Y = d * cos(theta)
                else:
                    coord_x, coord_y = float('nan'), float('nan') # Indicate invalid coords

                # --- Store Data ---
                if distance < min_distance and distance != float('inf'):
                    min_distance = distance
                    closest_detection = {
                        'class': object_class,
                        'distance': distance,
                        'bearing': bearing_angle,
                        'coord_x': coord_x,
                        'coord_y': coord_y,
                        'bbox': (x1, y1, x2, y2)
                    }

                # --- Prepare Label for Visualization ---
                label = f"{object_class}"
                dist_str = f"{distance:.1f}m" if distance != float('inf') else "Dist?"
                bearing_str = f"{bearing_angle:.0f}d"
                coord_str = f"X:{coord_x:.1f} Y:{coord_y:.1f}" if not (math.isnan(coord_x) or math.isnan(coord_y)) else "XY?"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put text above box
                y_text = y1 - 10 if y1 > 25 else y1 + 15 # Adjust text position if near top
                cv2.putText(frame, f"{label} {bearing_str}", (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"{dist_str} {coord_str}", (x1, y_text+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


            # Store the best detection from this processed frame
            if closest_detection:
                 last_detection_data = closest_detection
                 current_detection_data = closest_detection # Use for this frame's display too

    # --- Display Info using last known good data if not processed this frame ---
    display_data = current_detection_data if processed_this_frame else last_detection_data

    if display_data:
        # Highlight the target based on stored/current data
        x1, y1, x2, y2 = display_data['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3) # Magenta highlight

        # Display overall target info at the bottom
        target_info = f"TARGET: {display_data['class']} @ {display_data['distance']:.1f}m, {display_data['bearing']:.0f}deg"
        target_coords = f" (X:{display_data['coord_x']:.1f}, Y:{display_data['coord_y']:.1f})" if not math.isnan(display_data['coord_x']) else ""
        cv2.putText(frame, target_info + target_coords, (15, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "TARGET: None", (15, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    # --- Display FPS and Memory ---
    end_time = time.time()
    fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
    cv2.putText(frame, f"FPS: {int(fps)} (Proc Freq: 1/{PROCESS_EVERY_N_FRAMES})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    mem_usage = get_memory_usage()
    cv2.putText(frame, f"Mem: {mem_usage:.1f}MB", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Display Frame ---
    cv2.imshow("Object Detection", frame)
    # Optional: Display depth map from last processed frame if needed for debug
    # if processed_this_frame and depth_map is not None:
    #    cv2.imshow("Depth Map", depth_map)


    # --- Exit Condition ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        print("Exit key pressed.")
        break

# --- Cleanup ---
print("Releasing video capture and closing windows.")
cap.release()
cv2.destroyAllWindows()

# --- Print final target info ---
if last_detection_data:
    print("\nLast targeted object:")
    print(f"  Class: {last_detection_data['class']}")
    print(f"  Distance: {last_detection_data['distance']:.2f} m (APPROXIMATE - CALIBRATE!)")
    print(f"  Bearing: {last_detection_data['bearing']:.2f} degrees")
    print(f"  Coordinates (X,Y): ({last_detection_data['coord_x']:.2f}m, {last_detection_data['coord_y']:.2f}m)")
else:
    print("\nNo objects targeted during run.")