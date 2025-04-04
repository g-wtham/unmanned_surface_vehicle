import time
from dronekit import connect, VehicleMode, LocationGlobalRelative
import sys

# --- Connection Settings ---
# IMPORTANT: Make sure these match your Pi-to-Pixhawk connection
# Option 1: Connected via USB-to-Serial adapter
# Likely ports: /dev/ttyACM0, /dev/ttyUSB0 (Check using 'ls /dev/tty*' command)
connection_string = '/dev/ttyACM0' 

# Option 2: Connected via RPi's GPIO serial pins (requires RPi serial config)
# connection_string = '/dev/serial0' 

# Standard baud rates for telemetry ports (check ArduPilot SERIALx_BAUD param in Mission Planner/QGroundControl)
baud_rate = 57600 
# baud_rate = 115200 

# --- Time between updates ---
UPDATE_RATE_HZ = 2  # Get data 2 times per second (adjust as needed)
sleep_duration = 1.0 / UPDATE_RATE_HZ

# --- DroneKit Connection ---
vehicle = None  # Initialize vehicle variable
print(f"Attempting to connect to vehicle on: {connection_string} at {baud_rate} baud")
try:
    # wait_ready=True ensures essential parameters and messages (like GPS, attitude) are loaded.
    # Timeout prevents the script from hanging indefinitely if connection fails.
    vehicle = connect(connection_string, wait_ready=True, baud=baud_rate, timeout=60) 
    print("Pixhawk connected!")
    print(f"Firmware Version: {vehicle.version}") # Good check to see communication details

except Exception as e:
    print(f"Error connecting to Pixhawk: {e}")
    print("Please check:")
    print("- Is Pixhawk powered on and booted?")
    print(f"- Is the serial device '{connection_string}' correct?")
    print(f"- Is the baud rate '{baud_rate}' correct (matches ArduPilot SERIALx_BAUD)?")
    print("- Are the TX/RX wires connected correctly (TX->RX, RX->TX)?")
    print("- Does the RPi user have permission for the serial port (member of 'dialout' group)?")
    sys.exit(1) # Exit if connection fails


# --- Main Loop to Fetch Data ---
print("\nStarting data polling loop (Press Ctrl+C to exit)...")
try:
    while True:
        # --- Get Vehicle Location (Fused estimate using GPS and potentially other sensors) ---
        # global_relative_frame: Lat, Lon, Alt relative to Home altitude
        # global_frame: Lat, Lon, Alt AMSL (Above Mean Sea Level)
        current_location = vehicle.location.global_relative_frame
        lat = current_location.lat
        lon = current_location.lon
        alt = current_location.alt # Altitude relative to home

        # --- Get Raw GPS Info (from the first detected GPS - likely your M8N) ---
        gps_info = vehicle.gps_0 # gps_0 is the typical primary GPS instance
        gps_fix = gps_info.fix_type 
        num_sats = gps_info.satellites_visible

        # Fix type descriptions
        # 0-1: No fix, 2: 2D fix, 3: 3D fix, 4: DGPS, 5: RTK Float, 6: RTK Fixed
        fix_desc = {0: "No GPS", 1: "No Fix", 2: "2D Fix", 3: "3D Fix", 
                    4: "DGPS", 5: "RTK Float", 6: "RTK Fixed"}
        fix_status_str = fix_desc.get(gps_fix, f"Unknown ({gps_fix})")

        # --- Get Heading (from Compass/Magnetometer primarily) ---
        # Note: Heading relies on the compass calibration & magnetic environment
        current_heading = vehicle.heading

        # --- Get Battery Status (Requires setup in ArduPilot) ---
        battery_voltage = vehicle.battery.voltage
        battery_level = vehicle.battery.level # Percentage (can be None)

        # --- Get Current Flight Mode ---
        current_mode = vehicle.mode.name

        # --- Print the collected data ---
        print("-" * 30)
        print(f"Mode: {current_mode} | Heading: {current_heading}°")
        
        if lat is not None and lon is not None:
            print(f"Loc: {lat:.6f}, {lon:.6f} | Alt: {alt:.2f}m (rel home)")
        else:
            print("Loc: Data not available")
            
        if num_sats is not None:
             print(f"GPS: {fix_status_str} | Sats: {num_sats}")
        else:
             print("GPS: Status not available")
             
        if battery_voltage is not None:
            batt_str = f"Batt: {battery_voltage:.2f}V"
            if battery_level is not None:
                batt_str += f" ({battery_level}%)"
            print(batt_str)
        else:
            print("Batt: Data not available")

        # --- Wait before next update ---
        time.sleep(sleep_duration)

except KeyboardInterrupt:
    print("\nKeyboard interrupt detected. Exiting...")
except Exception as e:
    print(f"\nAn error occurred during the loop: {e}")
finally:
    # --- Cleanly close the connection ---
    if vehicle is not None and not vehicle.closed: # Check if vehicle exists and connection is open
         print("Closing Pixhawk connection...")
         vehicle.close()

print("Script finished.")