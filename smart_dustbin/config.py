"""Settings for the trash-catching bin.

The values are the ones from the original script. The only changes: the Bluetooth address and serial port can be set
from environment variables (the original had the author's device address written into the source).
"""
import os

import numpy as np

# --- camera ---
VIDEO_SRC = 0  # camera index, or a video file path

# --- Bluetooth link to the ESP32 (only used with the real robot) ---
ESP_MAC = os.environ.get("ESP_MAC", "")  # e.g. export ESP_MAC=AA:BB:CC:DD:EE:FF
RFCOMM_PORT = os.environ.get("RFCOMM_PORT", "/dev/rfcomm0")
BAUD_RATE = 115200

# --- how long each movement runs (seconds). Open loop: there is no sensor feedback ---
FORWARD_TIME = 1.0
BACKWARD_TIME = 1.0
TURN_90_TIME = 0.5
SIDE_FORWARD_TIME = 1.0

# --- red detection, in OpenCV HSV (hue 0-179). Red wraps around the hue axis, so two ranges are needed ---
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

MIN_RED_AREA = 500  # smallest blob (pixels squared) that counts as trash

ACTION_COOLDOWN = 3.0  # seconds between two movements

DRY_RUN_IF_NO_SERIAL = True  # without a robot, print the commands instead of sending them

# --- the car's placement box in the camera view (pixels), measured from the frame size ---
CAR_ZONE_SIZE = 60  # half the side of the white box the car is placed in
CAR_Y_RATIO = 0.85  # the box centre sits this far down the frame
