import time
import cv2
import numpy as np
from ultralytics import YOLO
import psutil
import onnxruntime # Make sure onnxruntime is installed

# --- Constants ---
# Logitech C270 webcam HFOV
CAMERA_HFOV = 55.0 # degrees

# --- Optimization Settings ---
# 1. Reduce Processing Resolution (Significantly impacts performance)
#    Try different values. Lower values = Faster FPS, less detail/accuracy far away.
PROC_WIDTH = 480  # Processing width (e.g., 640, 480, 320)
PROC_HEIGHT = 360 # Processing height (e.g., 480, 360, 240)

# 2. Frame Skipping (Process every Nth frame)
#    1 = Process every frame, 3 = Process every 3rd frame, etc.
PROCESS_EVERY_N_FRAMES = 3 # Adjust based on desired reactivity vs performance

# --- Other Settings ---
CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence for YOLO detection
DEPTH_MODEL_PATH = "./model-small.onnx"
YOLO_MODEL_PATH = "./plastic_detection_FloW.pt"
# Use -1 for default camera index if unsure, otherwise 0, 1, ...
CAMERA_INDEX = 0 

# --- Helper Functions ---
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # Return in MB

def depth_to_distance(depth) -> float:
    # !!! CRITICAL: CALIBRATE THIS FUNCTION FOR YOUR SETUP !!!
    # This is just a placeholder. Measure real distances and corresponding
    # 'depth' values from the normalized depth map to create an accurate mapping.
    # Example: Maybe it's an inverse relationship? distance = a / (depth + b)
    # Example: Or polynomial? distance = a*depth^2 + b*depth + c
    # For now, using the previous linear example - REPLACE WITH YOUR CALIBRATION
    distance_m = -2.0 * depth + 2.0 # Example: maps depth 0 to 2m, depth 1 to 0m? Needs check.
    return max(0.0, distance_m) # Ensure distance is not negative


# --- Load Models ---
print("Loading YOLO model...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    class_names = yolo_model.names
except Exception as e:
    print(f"Error loading YOLO model from {YOLO_MODEL_PATH}: {e}")
    exit()

print("Loading MiDaS depth model...")
try:
    # Use ONNX Runtime for inference
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 1 # Optimize for single inference thread? Test this.
    mono_model_session = onnxruntime.InferenceSession(DEPTH_MODEL_PATH, sess_options=session_options, providers=['CPUExecutionProvider'])
    # Get model input details (usually name is 'input')
    model_input_name = mono_model_session.get_inputs()[0].name
    # Get expected input shape (e.g., [1, 3, 256, 256]) - we need the H, W
    model_input_shape = mono_model_session.get_inputs()[0].shape
    onnx_input_height = model_input_shape[2]
    onnx_input_width = model_input_shape[3]
    print(f"MiDaS ONNX model expects input shape like: {model_input_shape}")
except Exception as e:
    print(f"Error loading MiDaS ONNX model from {DEPTH_MODEL_PATH}: {e}")
    exit()

print("Models loaded successfully.")

# --- Initialize Video Capture ---
print(f"Attempting to open camera at index {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open video source at index {CAMERA_INDEX}.")
    exit()

# Set desired capture resolution (optional, but good practice)
# Note: This might differ from processing resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera opened successfully. Actual capture resolution: {int(actual_width)}x{int(actual_height)}")
print(f"Processing resolution set to: {PROC_WIDTH}x{PROC_HEIGHT}")
print(f"Processing every {PROCESS_EVERY_N_FRAMES} frames.")

# --- Main Loop ---
frame_count = 0
last_fps_time = time.time()
fps_counter = 0
display_fps = 0

# Store last known detection results to display smoothly during skipped frames
last_display_frame = None
last_closest_detection_info = None
last_depth_map_display = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to capture frame.")
        # Attempt to reconnect or break after multiple failures?
        time.sleep(0.5)
        continue

    frame_count += 1
    fps_counter += 1

    # --- Resize Frame for Processing ---
    # Do this early to reduce data size for all subsequent steps
    try:
        proc_frame = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Error resizing frame: {e}")
        continue

    closest_detection = None # Reset for current processing cycle
    min_distance = float('inf')
    current_detections_drawn = [] # Keep track of boxes drawn in this cycle

    # --- Frame Skipping Logic ---
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # --- Process This Frame ---
        start_time = time.time() # Time the processing part

        # --- YOLO Detection ---
        # Run on the resized processing frame
        results = yolo_model(proc_frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detections = results[0].boxes.data.cpu().numpy()

        # --- Depth Estimation ---
        # Prepare frame for ONNX MiDaS model
        rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        # Resize specifically for the ONNX model's expected input size
        onnx_input = cv2.resize(rgb_frame, (onnx_input_width, onnx_input_height)).astype(np.float32)
        # Normalize (example, check MiDaS preprocessing requirements if different)
        onnx_input = onnx_input / 255.0
        # Mean subtraction (example, check MiDaS preprocessing)
        onnx_input -= np.array([0.485, 0.456, 0.406]) # ImageNet mean
        onnx_input /= np.array([0.229, 0.224, 0.225]) # ImageNet std
        # Transpose to NCHW format (Batch, Channels, Height, Width)
        onnx_input = onnx_input.transpose(2, 0, 1)
        # Add batch dimension
        onnx_input = np.expand_dims(onnx_input, axis=0)

        # Run ONNX inference
        depth_output = mono_model_session.run(None, {model_input_name: onnx_input})[0]

        # Process depth map output
        depth_map = depth_output.squeeze() # Remove batch and channel dims if present
        # Resize depth map back to processing frame size
        depth_map_resized = cv2.resize(depth_map, (PROC_WIDTH, PROC_HEIGHT))
        # Normalize depth map (0=far, 1=near typically, but verify)
        depth_map_normalized = cv2.normalize(depth_map_resized, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        last_depth_map_display = depth_map_normalized # Store for display

        # --- Process Detections (using PROC_WIDTH/HEIGHT) ---
        image_center_x = PROC_WIDTH / 2.0

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            # Coordinates are relative to proc_frame

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_id = int(cls_id)
            try:
                object_class = class_names[cls_id]
            except IndexError:
                continue # Skip invalid class ID

            object_center_x = (x1 + x2) / 2.0
            object_center_y = (y1 + y2) / 2.0

            bounded_center_x = int(np.clip(object_center_x, 0, PROC_WIDTH - 1))
            bounded_center_y = int(np.clip(object_center_y, 0, PROC_HEIGHT - 1))

            # Calculate Bearing Angle (using PROC_WIDTH and HFOV)
            pixel_deviation_x = object_center_x - image_center_x
            bearing_angle = (pixel_deviation_x * CAMERA_HFOV) / PROC_WIDTH if PROC_WIDTH > 0 else 0

            # Calculate Distance from normalized depth map
            depth_value = depth_map_normalized[bounded_center_y, bounded_center_x]
            distance = depth_to_distance(depth_value) # Use calibrated function

            # Store detection info drawn on this frame
            current_detections_drawn.append({
                'bbox': (x1, y1, x2, y2),
                'label': f"{object_class} {distance:.1f}m {bearing_angle:.0f}deg",
                'color': (0, 255, 0) # Default Green
            })

            # Find the closest valid object
            if distance < min_distance and distance > 0:
                min_distance = distance
                closest_detection = {
                    'class': object_class,
                    'distance': distance,
                    'bearing': bearing_angle,
                    'bbox': (x1, y1, x2, y2)
                }
        
        # --- Prepare Display Frame for this cycle ---
        display_frame = proc_frame.copy() # Draw on a copy of the resized frame

        # Draw all detections from this cycle
        for det_info in current_detections_drawn:
             x1, y1, x2, y2 = det_info['bbox']
             label = det_info['label']
             color = det_info['color']
             cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
             cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) # Smaller font

        # Highlight closest detection from this cycle
        last_closest_detection_info = None # Clear previous cycle info
        if closest_detection:
            x1, y1, x2, y2 = closest_detection['bbox']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 3) # Magenta border
            last_closest_detection_info = f"TARGET: {closest_detection['class']} @ {closest_detection['distance']:.1f}m, {closest_detection['bearing']:.0f}deg"
            # print(last_closest_detection_info) # Optional console output

        # Store this fully drawn frame
        last_display_frame = display_frame
        # End timing for processing part
        end_time = time.time() 
        proc_time = end_time - start_time
        # print(f"Frame {frame_count} Process Time: {proc_time:.4f} sec") # Debugging

    # --- Display Logic (Runs Every Frame) ---
    # Calculate FPS based on display rate
    current_time = time.time()
    if current_time - last_fps_time >= 1.0:
        display_fps = fps_counter
        fps_counter = 0
        last_fps_time = current_time

    # Use the last processed frame for display if available
    if last_display_frame is not None:
        display_output = last_display_frame.copy() # Work on a copy

        # Add FPS and Memory info
        cv2.putText(display_output, f"FPS: {display_fps}", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        mem_usage = get_memory_usage()
        cv2.putText(display_output, f"Mem: {mem_usage:.1f}MB", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add info about the last targeted object (if any)
        if last_closest_detection_info:
             cv2.putText(display_output, last_closest_detection_info, (15, display_output.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Object Detection", display_output)
    else:
        # If no frame has been processed yet, show the raw resized frame
        cv2.putText(proc_frame, "Initializing...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Object Detection", proc_frame)

    # Show the last processed depth map (if available)
    if last_depth_map_display is not None:
         cv2.imshow("Depth Map", last_depth_map_display)


    # --- Exit Condition ---
    key = cv2.waitKey(1) & 0xFF # Crucial: waitKey(1) allows frame display
    if key == 27 or key == ord('q'):
        print("Exit key pressed. Cleaning up...")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Resources released.")