# AI-Powered Smart Trash Bin

An autonomous robotic waste management system that uses computer vision to detect thrown trash and moves a motorized dustbin to intercept it.

---

## Overview

The system watches a live camera feed, identifies red-colored trash objects, determines which zone they land in, and drives the dustbin autonomously to that position. Communication between the host computer and the robot is handled over Bluetooth, with an ESP32 microcontroller controlling the DC motors.

---

## Features

- Real-time red object detection using OpenCV and HSV color space filtering
- 4-zone spatial awareness (Front, Back, Left, Right)
- Autonomous movement: forward, backward, and 90° turns
- Bluetooth RFCOMM communication with an ESP32 microcontroller
- Dry-run mode for testing without hardware connected
- Manual keyboard override for direct motor control
- Action cooldown to prevent repeated triggering
- On-screen status overlay and zone visualization

---

## Hardware Requirements

- ESP32 microcontroller
- L298N motor driver
- 2× DC motors (for differential drive)
- Webcam or smartphone camera (USB tethering supported)
- Bluetooth-capable host computer (Linux recommended)
- Power supply for motors

---

## Software Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- PySerial (`serial`) — optional; system runs in dry-run mode without it

Install dependencies:

```bash
pip install opencv-python numpy pyserial
```

---

## Configuration

Edit the constants at the top of the script before running:

| Variable | Default | Description |
|---|---|---|
| `VIDEO_SRC` | `0` | Camera index |
| `ESP_MAC` | `F8:B3:B7:21:F2:62` | Bluetooth MAC address of the ESP32 |
| `RFCOMM_PORT` | `/dev/rfcomm0` | Serial port for Bluetooth |
| `FORWARD_TIME` | `1.0` s | Duration for forward movement |
| `BACKWARD_TIME` | `1.0` s | Duration for backward movement |
| `TURN_90_TIME` | `0.5` s | Duration for a 90° turn |
| `SIDE_FORWARD_TIME` | `1.0` s | Forward duration after a side turn |
| `MIN_RED_AREA` | `500` px² | Minimum contour area to count as detection |
| `ACTION_COOLDOWN` | `3.0` s | Minimum time between actions |
| `DRY_RUN_IF_NO_SERIAL` | `True` | Simulate commands if no serial connection |

---

## Running the System

```bash
sudo python3 esp32_autocar_tracker.py
```

> `sudo` is required for Bluetooth RFCOMM binding on Linux.

### Setup Steps

1. Place the robot in the white box shown at the bottom center of the camera feed.
2. Orient it so the front faces upward (toward the green arrow on screen).
3. Press **Spacebar** or type **`G`** in the terminal to enable detection.
4. Throw a red object into one of the four colored zones.
5. The robot will execute the corresponding movement sequence.
6. After the action completes, return the robot to the starting box and re-enable detection.

---

## Zone Layout

The camera frame is divided into four detection zones:

```
        ┌──────────────────────────┐
        │         FRONT            │  ← green
        ├──────┬───────────┬───────┤
        │      │           │       │
  LEFT  │      │  (center) │       │  RIGHT
 (cyan) │      │           │       │  (yellow)
        │      │  [CAR]    │       │
        ├──────┴───────────┴───────┤
        │         BACK             │  ← blue
        └──────────────────────────┘
```

| Zone | Movement |
|------|----------|
| FRONT | Move forward |
| BACK | Move backward |
| LEFT | Turn 90° left, then move forward |
| RIGHT | Turn 90° right, then move forward |

---

## Motor Commands (ESP32)

Single-character commands are sent over serial:

| Command | Action |
|---------|--------|
| `U` | Forward |
| `D` | Backward |
| `L` | Turn left |
| `R` | Turn right |
| `S` | Stop |

---

## Manual Controls (Keyboard)

| Key | Action |
|-----|--------|
| `G` / Space | Enable detection |
| `W` | Forward |
| `S` | Backward |
| `A` | Left |
| `D` | Right |
| `X` | Stop |
| `1` | Test FRONT zone action |
| `2` | Test BACK zone action |
| `3` | Test LEFT zone action |
| `4` | Test RIGHT zone action |
| `Q` | Quit |

---

## System Architecture

```
Camera Feed
    │
    ▼
RedTrashDetector (OpenCV / HSV masking)
    │
    ▼
ZoneManager (centroid → zone mapping)
    │
    ▼
ActionExecutor (hardcoded movement sequence)
    │
    ▼
ESP32AutoConnect (Bluetooth RFCOMM)
    │
    ▼
ESP32 → L298N Motor Driver → DC Motors
```

---

## Troubleshooting

**No serial connection found**
The system falls back to dry-run mode automatically. Commands are printed to the console instead of sent over Bluetooth.

**Bluetooth bind fails**
Make sure the ESP32 is powered and paired with the host machine before running. Try releasing any existing binding manually:
```bash
sudo rfcomm release /dev/rfcomm0
```

**Camera not opening**
Change `VIDEO_SRC` to the correct index (try `1`, `2`, etc.) or provide a video file path.

**False detections**
Increase `MIN_RED_AREA` to filter out small red patches, or adjust `RED_LOWER1`/`RED_UPPER1`/`RED_LOWER2`/`RED_UPPER2` to better match your lighting conditions.

---

## Future Improvements

- Machine learning-based object recognition to support multiple trash types and colors
- Predictive movement using trajectory estimation
- Additional sensors (ultrasonic, IR) for obstacle avoidance
- Enhanced Bluetooth protocol for higher reliability and feedback
- Web or mobile dashboard for remote monitoring

---

## License

This project was developed for academic purposes. No license is currently specified.
