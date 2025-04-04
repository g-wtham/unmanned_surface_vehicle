# #############################################################################
# ####### Full Integrated USV Autonomous Garbage Detection & Targeting ########
# #############################################################################
# Combines:
# - YOLO Object Detection (Ultralytics)
# - MiDaS Depth Estimation (ONNX Runtime) - Optimized for CPU
# - Relative Bearing Calculation
# - DroneKit Connection to Pixhawk (GPS, Heading, Mode, Battery)
# - Absolute GPS Target Calculation (geopy)
# - Placeholder for Pixhawk Navigation Command
# #############################################################################

import time
import cv2
import numpy as np
from ultralytics import YOLO
import psutil
import onnxruntime # For MiDaS ONNX model
import sys
import threading # Potentially useful later for non-blocking tasks
from math import atan2, degrees

# --- DroneKit Imports ---
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil # Needed for command message definitions (though simple_goto often sufficient)

# --- Geodetic Calculations ---
from geopy import Point
from geopy.distance import geodesic

# #######################
# ## --- CONSTANTS & CONFIGURATION --- ##
# #######################

# --- Model Paths ---
YOLO_MODEL_PATH = "./plastic_detection_FloW.pt" # Your trained YOLO model
DEPTH_MODEL_PATH = "./model-small.onnx"     # MiDaS small ONNX model

# --- Camera Settings ---
CAMERA_INDEX = 0       # 0 is usually the default USB webcam
CAMERA_HFOV = 55.0     # Horizontal Field of View for Logitech C270 (degrees)

# --- Optimization Settings ---
PROC_WIDTH = 480       # Processing width (lower = faster) e.g., 320, 480, 640
PROC_HEIGHT = 360      # Processing height (lower = faster) e.g., 240, 360, 480
PROCESS_EVERY_N_FRAMES = 3 # Process every 3rd frame (1 = process all)

# --- Detection Settings ---
CONFIDENCE_THRESHOLD = 0.5 # Minimum YOLO confidence score

# --- Pixhawk Connection Settings (CHANGE THESE) ---
CONNECTION_STRING = '/dev/ttyACM0' # Serial port for Pixhawk (check /dev/tty*)
BAUD_RATE = 57600                  # Baud rate (match Pixhawk SERIALx_BAUD)

# --- Navigation/Control Parameters (TUNE THESE) ---
TARGET_STOP_DISTANCE = 0.8 # Meters: How close to get before stopping/collecting
WAYPOINT_HOLD_TIME_S = 5   # Seconds: How long to wait at waypoint (for collection)

# #############################
# ## --- HELPER FUNCTIONS --- ##
# #############################

def get_memory_usage():
    """Returns current memory usage of the script in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

def depth_to_distance(depth_normalized) -> float:
    """
    ### CRITICAL: CALIBRATE THIS FUNCTION!!! ###
    Maps the normalized depth value (0=far, 1=near - check normalization!) from MiDaS
    to an estimated distance in meters. This requires real-world measurements.

    Args:
        depth_normalized (float): Normalized depth value (typically 0.0 to 1.0).

    Returns:
        float: Estimated distance in meters, or negative/infinity if invalid.
    """
    # --- Placeholder - Replace with your calibration curve ---
    # Example 1: Simple inverse (distance = k / depth_value) - Tune 'k'
    # k = 1.0 # Example constant
    # if depth_normalized > 1e-4:
    #      distance_m = k / depth_normalized
    # else:
    #      distance_m = float('inf') # Very far / invalid

    # Example 2: Linear (distance = slope * depth + intercept) - Tune slope/intercept
    slope = -3.0 # Example
    intercept = 3.0 # Example
    distance_m = slope * depth_normalized + intercept

    # Ensure non-negative distance
    return max(0.0, distance_m)
    # ----------------------------------------------------------


def calculate_target_gps(start_lat, start_lon, bearing_deg, distance_m):
    """Calculates destination GPS coordinates given start, bearing, and distance."""
    try:
        start_point = Point(latitude=start_lat, longitude=start_lon)
        destination = geodesic(meters=distance_m).destination(point=start_point, bearing=bearing_deg)
        return destination.latitude, destination.longitude
    except Exception as e:
        print(f"[Error] GPS Calculation failed: {e}")
        return None, None

# ###################################
# ## --- INITIALIZATION STEPS --- ###
# ###################################

# --- 1. Load Models ---
print("[Info] Loading YOLO model...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    class_names = yolo_model.names
    print(f"[Info] YOLO classes: {class_names}")
except Exception as e:
    print(f"[Error] Failed to load YOLO model from {YOLO_MODEL_PATH}: {e}")
    sys.exit(1)

print("[Info] Loading MiDaS Depth model (ONNX Runtime)...")
try:
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 1 # Can potentially tune thread settings
    mono_model_session = onnxruntime.InferenceSession(DEPTH_MODEL_PATH, sess_options=session_options, providers=['CPUExecutionProvider'])
    model_input_name = mono_model_session.get_inputs()[0].name
    model_input_shape = mono_model_session.get_inputs()[0].shape
    onnx_input_height = model_input_shape[2]
    onnx_input_width = model_input_shape[3]
    print(f"[Info] MiDaS ONNX expects input shape like: {model_input_shape}")
except Exception as e:
    print(f"[Error] Failed to load MiDaS ONNX model from {DEPTH_MODEL_PATH}: {e}")
    sys.exit(1)

print("[Info] Models loaded successfully.")

# --- 2. Connect to Pixhawk (DroneKit) ---
vehicle = None
print(f"[Info] Attempting to connect to Pixhawk on {CONNECTION_STRING} at {BAUD_RATE} baud...")
try:
    vehicle = connect(CONNECTION_STRING, wait_ready=True, baud=BAUD_RATE, timeout=60)
    print("[Success] Pixhawk Connected!")
    print(f"  Firmware: {vehicle.version}")
    print(f"  Mode: {vehicle.mode.name}")
    print(f"  GPS: {vehicle.gps_0}")
    print(f"  Battery: {vehicle.battery}")
    print(f"  Is Armable? {vehicle.is_armable}") # Good status check

    # Optional: Set initial parameters if needed (e.g., Fence, Geofence - Advanced)

except Exception as e:
    print(f"[Critical Error] Failed to connect to Pixhawk: {e}")
    print("  Check: \n  - Pixhawk powered & booted?\n  - Correct port & baud rate?\n  - Connection wires (TX->RX)?\n  - RPi user permissions ('dialout' group)?")
    sys.exit(1)


# --- 3. Initialize Video Capture ---
print(f"[Info] Opening camera at index {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"[Critical Error] Cannot open camera at index {CAMERA_INDEX}.")
    if vehicle: vehicle.close() # Close connection if camera fails
    sys.exit(1)

# Optional: Set camera capture resolution (check `v4l2-ctl --list-formats-ext` for options)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"[Info] Camera opened. Capture res: {int(actual_width)}x{int(actual_height)}. Processing res: {PROC_WIDTH}x{PROC_HEIGHT}")


# #############################
# ## --- MAIN CONTROL LOOP --- ##
# #############################
frame_count = 0
last_fps_time = time.time()
fps_counter = 0
display_fps = 0

# State variables for smoother display/logic across skipped frames
last_display_frame = None
last_closest_detection_info_str = "Status: Initializing..."
last_target_gps_str = ""
last_depth_map_display = np.zeros((PROC_HEIGHT, PROC_WIDTH), dtype=np.float32) # Placeholder

# Simple state machine could be useful: "SEARCHING", "MOVING_TO_TARGET", "COLLECTING", "RETURNING"
current_state = "SEARCHING"

print("\n[Info] Starting main loop...")
try:
    while True:
        # --- Read Frame ---
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to capture frame. Skipping...")
            time.sleep(0.1)
            continue

        frame_count += 1
        fps_counter += 1

        # --- Resize for Processing ---
        proc_frame = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT), interpolation=cv2.INTER_AREA)

        # --- Variables for current cycle ---
        closest_detection_this_cycle = None
        min_distance_this_cycle = float('inf')
        current_detections_drawn = [] # List of dicts for drawing boxes/labels
        processed_this_frame = False # Flag if processing happened


        # === FRAME PROCESSING BLOCK (Based on Frame Skipping) ===
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            processed_this_frame = True
            start_proc_time = time.time()

            # 1. --- YOLO Detection ---
            results = yolo_model(proc_frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            detections = results[0].boxes.data.cpu().numpy()

            # 2. --- Depth Estimation ---
            #    a. Preprocess for ONNX MiDaS
            rgb_frame_onnx = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            onnx_input = cv2.resize(rgb_frame_onnx, (onnx_input_width, onnx_input_height)).astype(np.float32)
            onnx_input = onnx_input / 255.0
            onnx_input -= np.array([0.485, 0.456, 0.406]) # ImageNet Mean
            onnx_input /= np.array([0.229, 0.224, 0.225]) # ImageNet Std
            onnx_input = onnx_input.transpose(2, 0, 1) # HWC to CHW
            onnx_input = np.expand_dims(onnx_input, axis=0) # Add batch dim

            #    b. Run ONNX Inference
            depth_output = mono_model_session.run(None, {model_input_name: onnx_input})[0]

            #    c. Postprocess Depth Map
            depth_map = depth_output.squeeze()
            depth_map_resized = cv2.resize(depth_map, (PROC_WIDTH, PROC_HEIGHT))
            depth_map_normalized = cv2.normalize(depth_map_resized, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            last_depth_map_display = depth_map_normalized # Update display version

            # 3. --- Process Detections & Calculate Target ---
            image_center_x = PROC_WIDTH / 2.0
            last_target_gps_str = "" # Clear target from previous cycle

            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cls_id = int(cls_id)
                try:
                    object_class = class_names[cls_id]
                except IndexError: continue # Skip if class ID invalid

                #   a. Calculate Object Center & Relative Bearing
                object_center_x = (x1 + x2) / 2.0
                object_center_y = (y1 + y2) / 2.0
                pixel_deviation_x = object_center_x - image_center_x
                bearing_relative_deg = (pixel_deviation_x * CAMERA_HFOV) / PROC_WIDTH if PROC_WIDTH > 0 else 0

                #   b. Estimate Distance
                bounded_center_x = int(np.clip(object_center_x, 0, PROC_WIDTH - 1))
                bounded_center_y = int(np.clip(object_center_y, 0, PROC_HEIGHT - 1))
                depth_value = depth_map_normalized[bounded_center_y, bounded_center_x]
                distance_m = depth_to_distance(depth_value) # !!! USES CALIBRATED FUNCTION !!!

                # Prepare info for drawing later
                label = f"{object_class} {distance_m:.1f}m {bearing_relative_deg:.0f}d"
                current_detections_drawn.append({'bbox': (x1, y1, x2, y2), 'label': label, 'color': (0, 255, 0)})

                #   c. Find Closest *Valid* Target
                if 0 < distance_m < min_distance_this_cycle: # Must be positive distance
                     min_distance_this_cycle = distance_m
                     closest_detection_this_cycle = {
                         'class': object_class,
                         'distance': distance_m,
                         'bearing': bearing_relative_deg,
                         'bbox': (x1, y1, x2, y2)
                     }

            # 4. --- Calculate Absolute GPS of Closest Target (if found & GPS good) ---
            target_lat, target_lon = None, None
            if closest_detection_this_cycle and vehicle.location.global_relative_frame.lat is not None and vehicle.heading is not None:
                # Get current USV state
                lat_usv = vehicle.location.global_relative_frame.lat
                lon_usv = vehicle.location.global_relative_frame.lon
                alt_usv = vehicle.location.global_relative_frame.alt # Relative altitude
                heading_usv = vehicle.heading

                # Get closest target details
                dist_m = closest_detection_this_cycle['distance']
                rel_bearing_deg = closest_detection_this_cycle['bearing']

                # Calculate absolute bearing
                absolute_bearing_deg = (heading_usv + rel_bearing_deg + 360) % 360

                # Calculate target GPS
                target_lat, target_lon = calculate_target_gps(lat_usv, lon_usv, absolute_bearing_deg, dist_m)

                # Update status strings
                if target_lat is not None:
                    last_closest_detection_info_str = (f"Target: {closest_detection_this_cycle['class']} "
                                                       f"@ {dist_m:.1f}m, {rel_bearing_deg:.0f}deg rel "
                                                       f"-> {absolute_bearing_deg:.0f}deg abs")
                    last_target_gps_str = f"Target GPS: ({target_lat:.6f}, {target_lon:.6f})"
                    print(f"[Target Found] {last_closest_detection_info_str} | {last_target_gps_str}") # Log target
                else:
                    last_closest_detection_info_str = "Status: GPS Calc Error"
                    last_target_gps_str = ""

            elif closest_detection_this_cycle:
                # Target detected but USV GPS/Heading unavailable
                last_closest_detection_info_str = "Status: Target Visible, Waiting GPS/Heading..."
                last_target_gps_str = ""
            else:
                # No target detected this cycle
                 last_closest_detection_info_str = "Status: Searching..."
                 last_target_gps_str = ""

            # === END OF FRAME PROCESSING BLOCK ===


        # === PIXHAWK CONTROL LOGIC (PLACEHOLDER) ===
        # This section would typically run regardless of frame processing,
        # acting based on the *last known target* and *current state*.
        # For now, we just use the info calculated in the processing block.

        if processed_this_frame and target_lat is not None and current_state == "SEARCHING":
            print("[Control] Target acquired. Switching to GUIDED mode (if not already).")

            if vehicle.mode.name != "GUIDED":
                vehicle.mode = VehicleMode("GUIDED")
                vehicle.flush() # Send command immediately
                time.sleep(1) # Wait for mode change

            if vehicle.mode.name == "GUIDED":
                print(f"[Control] Commanding USV to target: Lat={target_lat:.6f}, Lon={target_lon:.6f}")
                target_location = LocationGlobalRelative(target_lat, target_lon, alt_usv) # Use current relative altitude

                # --- >>> ACTUAL PIXHAWK COMMAND <<< ---
                # This sends the command to the Pixhawk
                vehicle.simple_goto(target_location)
                print("[Control] simple_goto command sent.")
                # -----------------------------------------

                current_state = "MOVING_TO_TARGET" # Update state
                last_closest_detection_info_str = "Status: Moving to Target" # Update display status
            else:
                 print("[Warning] Could not switch to GUIDED mode.")
                 last_closest_detection_info_str = "Status: Mode Change Failed"


        # !!! Additional logic needed here: !!!
        # - Check if USV has reached the target (distance_to_waypoint < TARGET_STOP_DISTANCE)
        # - If reached: Stop motors, trigger relay for collection, wait, stop relay (State: COLLECTING)
        # - After collection: Return to SEARCHING state or RTL if bin full.
        # - Handle Bin Full: Check capacity (needs separate logic/sensor/camera), if full set state to RETURNING, command RTL.
        # - Handle loss of target: Revert to SEARCHING if target lost while MOVING_TO_TARGET?


        # === DISPLAY & VISUALIZATION ===
        display_output = None
        if processed_this_frame:
            # Draw on the processed frame
            display_frame_this_cycle = proc_frame.copy()
            # Draw all detections from this cycle
            for det_info in current_detections_drawn:
                x1, y1, x2, y2 = det_info['bbox']
                cv2.rectangle(display_frame_this_cycle, (x1, y1), (x2, y2), det_info['color'], 2)
                cv2.putText(display_frame_this_cycle, det_info['label'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Highlight the closest detection (if any from this cycle)
            if closest_detection_this_cycle:
                x1, y1, x2, y2 = closest_detection_this_cycle['bbox']
                cv2.rectangle(display_frame_this_cycle, (x1, y1), (x2, y2), (255, 0, 255), 3) # Magenta

            last_display_frame = display_frame_this_cycle # Store for display

        # Update FPS counter (independent of processing rate)
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            display_fps = fps_counter
            fps_counter = 0
            last_fps_time = current_time

        # Draw overlays on the latest available processed frame
        if last_display_frame is not None:
             display_output = last_display_frame.copy()
             
             # System Info
             mem_usage = get_memory_usage()
             cv2.putText(display_output, f"FPS: {display_fps}", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
             cv2.putText(display_output, f"Mem: {mem_usage:.1f}MB", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
             
             # Vehicle Status
             mode_str = f"Mode: {vehicle.mode.name}"
             batt_str = f"Batt: {vehicle.battery.voltage:.1f}V" if vehicle.battery.voltage else "Batt: N/A"
             gps_str = f"GPS: {vehicle.gps_0.fix_type} ({vehicle.gps_0.satellites_visible} sats)" if vehicle.gps_0 else "GPS: N/A"
             hdg_str = f"Hdg: {vehicle.heading}" if vehicle.heading is not None else "Hdg: N/A"

             cv2.putText(display_output, mode_str, (PROC_WIDTH - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
             cv2.putText(display_output, batt_str, (PROC_WIDTH - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
             cv2.putText(display_output, gps_str, (PROC_WIDTH - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
             cv2.putText(display_output, hdg_str, (PROC_WIDTH - 150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

             # Detection / Targeting Status
             cv2.putText(display_output, last_closest_detection_info_str, (15, PROC_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 255), 1)
             cv2.putText(display_output, last_target_gps_str, (15, PROC_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 255), 1)
             cv2.putText(display_output, f"State: {current_state}", (PROC_WIDTH - 150, PROC_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)


        else:
            # Fallback if no frame processed yet
            display_output = proc_frame.copy()
            cv2.putText(display_output, "Initializing...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frames
        cv2.imshow("Object Detection", display_output)
        cv2.imshow("Depth Map (Normalized)", last_depth_map_display)


        # === Exit Condition ===
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): # ESC or 'q'
            print("[Info] Exit key pressed.")
            break

except KeyboardInterrupt:
    print("[Info] Keyboard Interrupt detected.")
except Exception as e:
    print(f"[Critical Error] An unexpected error occurred in the main loop: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback for debugging
finally:
    # ########################
    # ## --- CLEANUP --- ###
    # ########################
    print("[Info] Cleaning up resources...")
    # Release camera
    if 'cap' in locals() and cap.isOpened():
        cap.release()
        print("[Info] Camera released.")
    # Destroy OpenCV windows
    cv2.destroyAllWindows()
    print("[Info] OpenCV windows closed.")
    # Close DroneKit connection
    if vehicle is not None and not vehicle.closed:
        print("[Info] Returning to HOLD/LOITER mode (if possible).")
        # Attempt to put into a safe holding mode before disconnecting
        try:
            if vehicle.mode.name not in ["HOLD", "LOITER"]:
                if "HOLD" in vehicle.mode.available_modes: # Check available modes
                    vehicle.mode = VehicleMode("HOLD")
                elif "LOITER" in vehicle.mode.available_modes:
                    vehicle.mode = VehicleMode("LOITER")
                vehicle.flush()
                time.sleep(1)
        except Exception as mode_err:
             print(f"[Warning] Could not set final mode: {mode_err}")
             
        print("[Info] Closing Pixhawk connection...")
        vehicle.close()
        print("[Info] DroneKit connection closed.")

    print("[Info] Script finished.")