import time
import cv2
import numpy as np
from ultralytics import YOLO
import psutil
import math
from dronekit import connect, VehicleMode, LocationGlobalRelative # Import DroneKit components
import os # For checking file existence

# --- Constants ---
CAMERA_HFOV = 55.0 # degrees (Logitech C270)
PROCESS_EVERY_N_FRAMES = 1 # Process frames frequently
PROCESSING_WIDTH = 640 # Target processing width
PROCESSING_HEIGHT = 480 # Target processing height
CONFIDENCE_THRESHOLD = 0.5
EARTH_RADIUS_KM = 6371.0
YOLO_MODEL_PATH = "./plastic_detection_FloW.pt"
DEPTH_MODEL_PATH = "./model-small.onnx"
VIDEO_SOURCE = "./video5.mp4" # <<< CHANGE TO 0 or camera index for live deployment
# VIDEO_SOURCE = 0 # Example for using default live camera

# --- DroneKit Connection Settings ---
connection_string = "/dev/ttyACM0" # Adjust if your Pixhawk connects differently
baud_rate = 115200 # Match ArduPilot SERIAL port settings (e.g., SERIAL1_BAUD)

# --- Helper Functions ---
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # MB

def depth_to_distance(depth_value, method='calibrated_inverse_v1'):
    """
    Converts normalized depth value (0-1 from MiDaS) to estimated distance in meters.
    Uses 'calibrated_inverse_v1' based on previous notes.
    !!! REQUIRES MORE DATA POINTS FOR ACCURATE CALIBRATION !!!
    """
    if method == 'calibrated_inverse_v1':
        k = 0.188
        MIN_DIST_CLAMP = 0.15
        MAX_DIST_CLAMP = 3.0 # Max reliable distance (NEEDS TESTING!)
        if depth_value > 1e-5:
            calculated_distance = k / depth_value
            final_distance = max(MIN_DIST_CLAMP, min(calculated_distance, MAX_DIST_CLAMP))
            return final_distance
        else:
            return MAX_DIST_CLAMP
    else: # Fallback
        MAX_DIST_FALLBACK = 2.0
        MIN_DIST_FALLBACK = 0.2
        if 0 <= depth_value <= 1:
             distance = MAX_DIST_FALLBACK - depth_value * (MAX_DIST_FALLBACK - MIN_DIST_FALLBACK)
             return max(MIN_DIST_FALLBACK, distance)
        else:
             return float('inf')

def calculate_destination_gps(lat, lon, distance_m, bearing_deg):
    """
    Calculates the destination GPS coordinates given a starting point, distance, and bearing.
    Uses actual bearing (relative + vehicle heading).
    """
    if lat is None or lon is None or distance_m is None or bearing_deg is None or distance_m == float('inf'):
        print("Warning: Invalid input for GPS calculation.")
        return None, None

    distance_km = distance_m / 1000.0
    R = EARTH_RADIUS_KM
    try:
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing_deg)

        lat2_rad = math.asin(math.sin(lat_rad) * math.cos(distance_km / R) +
                             math.cos(lat_rad) * math.sin(distance_km / R) * math.cos(bearing_rad))

        lon2_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat_rad),
                                        math.cos(distance_km / R) - math.sin(lat_rad) * math.sin(lat2_rad))

        lat2_deg = math.degrees(lat2_rad)
        lon2_deg = math.degrees(lon2_rad)
        return lat2_deg, lon2_deg
    except (ValueError, TypeError) as e:
        print(f"Math/Type error during GPS calculation: {e}. Inputs: lat={lat}, lon={lon}, dist={distance_m}m, bearing={bearing_deg}deg")
        return None, None

# --- Load Models ---
print("Loading models...")
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"Error: YOLO model not found at {YOLO_MODEL_PATH}")
    exit()
if not os.path.exists(DEPTH_MODEL_PATH):
    print(f"Error: Depth model not found at {DEPTH_MODEL_PATH}")
    exit()

try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    class_names = yolo_model.names
    print("YOLO model loaded.")
    mono_model = cv2.dnn.readNet(DEPTH_MODEL_PATH)
    print("MiDaS depth model loaded.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# --- Initialize Video Capture ---
print(f"Initializing video source: {VIDEO_SOURCE}")
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_SOURCE}")
    # If using camera index, check permissions and if camera is connected
    exit()

# Attempt to set resolution (camera might ignore this or provide closest match)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESSING_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESSING_HEIGHT)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video source opened. Requested {PROCESSING_WIDTH}x{PROCESSING_HEIGHT}, Actual {actual_width}x{actual_height}")

WIDTH = actual_width
HEIGHT = actual_height
image_center_x = WIDTH / 2.0

# --- Connect to Pixhawk ---
vehicle = None
try:
    print(f"Connecting to vehicle on: {connection_string} at {baud_rate} baud")
    vehicle = connect(connection_string, wait_ready=False, baud=baud_rate, timeout=60) # wait_ready=False for faster startup, check later
    vehicle.wait_ready(True, timeout=60) # Now wait for parameters, etc.
    print("Pixhawk connected.")
    print(f" Initial Vehicle State:")
    print(f"  Mode: {vehicle.mode.name}")
    print(f"  Armed: {vehicle.armed}")
    print(f"  GPS: {vehicle.gps_0.fix_type}")
    print(f"  Location: {vehicle.location.global_relative_frame}")
    print(f"  Heading: {vehicle.heading}")

except Exception as e:
    print(f"Error connecting to Pixhawk: {e}")
    # Decide if script should exit or continue without Pixhawk for testing?
    print("Continuing without Pixhawk connection (CV only).")
    # vehicle remains None

# --- Main Loop ---
frame_count = 0
last_detection_data = None # Stores data about the last valid target
print("\n--- Starting Main Loop ---")
print("Press 'q' or ESC to Quit")

try:
    while True:
        start_frame_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of video file or cannot read frame.")
            # If using video file, break. If live camera, maybe retry/wait.
            if isinstance(VIDEO_SOURCE, str): # If source is a file path
                break
            else:
                time.sleep(0.5)
                continue

        # Resize frame immediately if actual capture differs from processing size
        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
             frame = cv2.resize(frame, (WIDTH, HEIGHT))

        frame_count += 1
        processed_this_frame = False
        current_detection_data = None
        depth_map_display = None # For visualizing depth map

        # --- Get Real Vehicle State (if connected) ---
        current_mode = "N/A"
        is_armed = False
        usv_lat = None
        usv_lon = None
        usv_heading = None
        gps_fix = 0

        if vehicle and vehicle.is_connected:
             current_mode = vehicle.mode.name
             is_armed = vehicle.armed
             # Ensure location data is available
             if vehicle.location and vehicle.location.global_relative_frame:
                 usv_lat = vehicle.location.global_relative_frame.lat
                 usv_lon = vehicle.location.global_relative_frame.lon
             # Ensure heading data is available
             if vehicle.heading is not None:
                 usv_heading = vehicle.heading
             if vehicle.gps_0:
                 gps_fix = vehicle.gps_0.fix_type


        # --- Conditional Processing: Only run CV if GUIDED and ARMED ---
        run_cv = vehicle and vehicle.is_connected and current_mode == "GUIDED" and is_armed

        if run_cv:
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                processed_this_frame = True
                process_frame = frame # Use the (potentially resized) frame

                # --- YOLO Detection ---
                results = yolo_model(process_frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
                detections = results[0].boxes.data.cpu().numpy()

                # --- Depth Estimation (only if detections exist) ---
                depth_map = None # Reset depth map
                if len(detections) > 0:
                    try:
                         rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                         blob = cv2.dnn.blobFromImage(rgb_frame, 1/255., (256, 256), (123.675, 116.28, 103.53), swapRB=True, crop=False)
                         mono_model.setInput(blob)
                         depth_map_raw = mono_model.forward().squeeze()
                         depth_map = cv2.resize(depth_map_raw, (WIDTH, HEIGHT))
                         depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                         depth_map_display = depth_map.copy() # Keep for display
                    except Exception as e:
                         print(f"Error during depth estimation: {e}")
                         depth_map = None # Ensure it's None if error occurs

                # --- Process Detections ---
                closest_detection = None
                min_distance = float('inf')
                detections_info_for_drawing = []

                if depth_map is not None and usv_lat is not None and usv_lon is not None and usv_heading is not None:
                    for det in detections:
                        x1, y1, x2, y2, conf, cls_id = det
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cls_id = int(cls_id)
                        try: object_class = class_names[cls_id]
                        except IndexError: continue

                        object_center_x = (x1 + x2) / 2.0
                        object_center_y = (y1 + y2) / 2.0
                        bounded_center_x = int(np.clip(object_center_x, 0, WIDTH - 1))
                        bounded_center_y = int(np.clip(object_center_y, 0, HEIGHT - 1))

                        # Calculate Relative Bearing
                        pixel_deviation_x = object_center_x - image_center_x
                        relative_bearing_deg = (pixel_deviation_x * CAMERA_HFOV) / WIDTH

                        # Calculate Distance
                        depth_value = depth_map[bounded_center_y, bounded_center_x]
                        distance_m = depth_to_distance(depth_value, method='calibrated_inverse_v1')

                        # Calculate Absolute Bearing (using REAL heading)
                        absolute_bearing_deg = (usv_heading + relative_bearing_deg) % 360

                        # Calculate Target GPS (using REAL location & calculated bearing/distance)
                        target_lat, target_lon = calculate_destination_gps(usv_lat, usv_lon, distance_m, absolute_bearing_deg)

                        det_info = {
                            'bbox': (x1, y1, x2, y2),
                            'class': object_class,
                            'distance': distance_m,
                            'rel_bearing': relative_bearing_deg,
                            'abs_bearing': absolute_bearing_deg,
                            'target_lat': target_lat,
                            'target_lon': target_lon
                        }
                        detections_info_for_drawing.append(det_info)

                        if distance_m < min_distance and distance_m != float('inf'):
                            min_distance = distance_m
                            closest_detection = det_info # Store full data for closest

                    # Update last known target if found this frame
                    if closest_detection:
                         last_detection_data = closest_detection
                         current_detection_data = closest_detection
                         # --- ACTION: Send Waypoint (Example) ---
                         # if target_lat is not None and target_lon is not None:
                         #    target_location = LocationGlobalRelative(target_lat, target_lon, altitude) # Define altitude
                         #    print(f"COMMANDING: Move to {target_lat:.6f}, {target_lon:.6f}")
                         #    vehicle.simple_goto(target_location)
                         #    # Need logic here to wait for arrival, trigger collection, etc. (from pseudo code)

                else: # No depth map or missing vehicle state
                     if not (usv_lat is not None and usv_lon is not None and usv_heading is not None):
                           print("Waiting for valid GPS fix and heading...")


        # --- Display Logic (Runs every loop) ---
        display_frame = frame.copy()

        # Use latest data if processed, otherwise show last known target
        display_target_data = current_detection_data if processed_this_frame else last_detection_data

        # Draw detection boxes from the last processed frame
        if processed_this_frame and len(detections_info_for_drawing) > 0:
             for info in detections_info_for_drawing:
                x1, y1, x2, y2 = info['bbox']
                dist_str = f"{info['distance']:.1f}m" if info['distance'] != float('inf') else "Dist?"
                rel_bearing_str = f"{info['rel_bearing']:.0f}d"
                label = f"{info['class']} {dist_str} {rel_bearing_str}"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Highlight the main target
        if display_target_data:
            x1, y1, x2, y2 = display_target_data['bbox']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2) # Magenta highlight

            target_dist = f"{display_target_data['distance']:.2f}m" if display_target_data['distance'] != float('inf') else "Dist?"
            target_info = f"TARGET: {display_target_data['class']} @ {target_dist}"
            if display_target_data['target_lat'] is not None:
                target_gps = f" GPS: {display_target_data['target_lat']:.6f}, {display_target_data['target_lon']:.6f}"
            else:
                target_gps = " (GPS Calc Fail)"
            cv2.putText(display_frame, target_info + target_gps, (15, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        elif run_cv: # Only show "TARGET: None" if we are actively looking
            cv2.putText(display_frame, "TARGET: None", (15, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


        # --- Display Vehicle Status ---
        elapsed_time = time.time() - start_frame_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        status_text = f"FPS:{int(fps):>3} Mem:{get_memory_usage():.0f}MB"
        if vehicle and vehicle.is_connected:
             mode_str = f" Mode:{current_mode}"
             arm_str = " ARMED" if is_armed else " DISARMED"
             gps_str = f" GPS:{gps_fix}"
             head_str = f" Hdg:{usv_heading if usv_heading is not None else '---':.1f}"
             status_text += mode_str + arm_str + gps_str + head_str
             if not run_cv and current_mode != "GUIDED":
                 status_text += f" (CV Idle - Mode:{current_mode})"
             elif not run_cv and not is_armed:
                  status_text += f" (CV Idle - Disarmed)"
             elif not (usv_lat is not None and usv_lon is not None and usv_heading is not None):
                  status_text += " (Waiting GPS/Heading)"

        else:
             status_text += " (Pixhawk Disconnected)"

        cv2.putText(display_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) # Cyan status

        # --- Show Frames ---
        cv2.imshow("Object Detection", display_frame)
        if depth_map_display is not None and processed_this_frame:
             cv2.imshow("Depth Map", depth_map_display)


        # --- Exit Condition ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("Exit key pressed.")
            break

# --- Cleanup ---
except KeyboardInterrupt:
    print("Keyboard interrupt detected. Cleaning up...")
finally:
    print("Releasing resources...")
    if vehicle and vehicle.is_connected:
        # Set mode to HOLD or LOITER maybe? Or just disconnect?
        # vehicle.mode = VehicleMode("HOLD")
        vehicle.close()
        print("Pixhawk disconnected.")
    cap.release()
    print("Video capture released.")
    cv2.destroyAllWindows()
    print("Display windows closed.")

    # Print final target info if available
    if last_detection_data:
        print("\nLast targeted object:")
        # (Print details like in the simulation script)
        print(f"  Class: {last_detection_data['class']}")
        #... etc ...
    else:
        print("\nNo objects targeted.")