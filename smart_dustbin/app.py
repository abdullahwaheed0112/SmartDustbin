"""The live program: a camera watches the floor, and when red trash lands in a zone the bin moves towards that zone.

    python -m smart_dustbin live

Press SPACE or G in the terminal to arm detection. Without a Bluetooth address (ESP_MAC) or without the robot the
commands are printed instead of sent ("dry run"), so the whole loop can be tried with just a webcam.
"""
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2

from . import config
from .actions import ActionExecutor
from .transport import BluetoothTransport, DryRunTransport, ESP32AutoConnect
from .vision import RedTrashDetector, ZoneManager

WINDOW_TITLE = "ESP32 4-Zone Trash Collector"

# keyboard: manual driving and zone tests
MANUAL_MOVES = {'W': 'U', 'A': 'L', 'S': 'D', 'D': 'R', 'X': 'S'}
ZONE_TEST_KEYS = {'1': 'FRONT', '2': 'BACK', '3': 'LEFT', '4': 'RIGHT'}


@dataclass
class LiveState:
    running: bool = True
    detection_enabled: bool = False  # armed with SPACE / G


def keyboard_thread(state, commands):
    """Manual keyboard control for testing: reads lines from the terminal and queues the commands."""
    print("\n=== Manual Control (Testing) ===")
    print("SPACEBAR or G = Enable detection and movement")
    print("W=Forward, S=Backward, A=Left, D=Right, X=Stop, Q=Quit")
    print("1=Test FRONT, 2=Test BACK, 3=Test LEFT, 4=Test RIGHT")

    try:
        while state.running:
            line = input().strip()
            if not line:
                continue
            cmd = line[0].upper()

            if cmd == 'Q':
                commands.append('Q')
                break
            elif cmd in MANUAL_MOVES or cmd in ZONE_TEST_KEYS:
                commands.append(cmd)
            elif cmd == 'G' or cmd == ' ':
                commands.append('G')
    except Exception:
        pass
    finally:
        state.running = False


def show_in_window(frame):
    """Show a frame in an OpenCV window. Returns True when the user pressed q to quit."""
    cv2.imshow(WINDOW_TITLE, frame)
    return (cv2.waitKey(1) & 0xFF) == ord('q')


def run_live(capture, transport, executor=None, commands=None, state=None, display=show_in_window,
             max_frames=None, threaded=True, log=print):
    """The main loop. Everything it talks to is passed in, so tests can use a fake camera and no window.

    `capture` needs a read() returning (ok, frame). `display(frame)` returns True to quit. `max_frames` stops the loop
    after that many frames (for tests). With `threaded=False` the movement runs inline instead of in a thread.
    """
    state = state or LiveState()
    commands = commands if commands is not None else deque()
    executor = executor or ActionExecutor(transport, log=log)

    ok, frame = capture.read()
    if not ok:
        log("Error: Cannot read from camera")
        return state
    h, w = frame.shape[:2]

    zone_manager = ZoneManager(w, h)
    detector = RedTrashDetector()

    log("\n" + "=" * 60)
    log("SETUP INSTRUCTIONS:")
    log("=" * 60)
    log("1. Place your ESP32 car in the WHITE BOX at bottom center")
    log("2. Orient car so FRONT faces UP (green arrow)")
    log("3. Press SPACEBAR or 'G' in terminal to enable detection")
    log("4. Throw red trash into one of the 4 colored zones")
    log("5. Car will execute movement for that zone")
    log("6. Detection auto-disables after movement completes")
    log("7. Return car to white box and press SPACEBAR/'G' again")
    log("=" * 60 + "\n")

    frames = 0
    while state.running:
        if max_frames is not None and frames >= max_frames:
            break
        ok, frame = capture.read()
        frames += 1
        if not ok:
            time.sleep(0.03)
            continue

        zone_manager.draw_zones(frame)

        if state.detection_enabled:
            red_pos = detector.detect_red(frame)
            if red_pos:
                rx, ry = red_pos
                cv2.circle(frame, (rx, ry), 15, (0, 0, 255), 3)
                cv2.circle(frame, (rx, ry), 5, (0, 0, 255), -1)

                zone = zone_manager.get_zone_for_position(rx, ry)
                if zone:
                    cv2.putText(frame, f"DETECTED: {zone}", (rx + 20, ry),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if executor.can_execute():
                        state.detection_enabled = False  # one throw, one movement
                        log("\n>>> Detection DISABLED - Movement starting...")
                        if threaded:
                            threading.Thread(target=executor.execute_zone_action, args=(zone,), daemon=True).start()
                        else:
                            executor.execute_zone_action(zone)

        if executor.executing:
            status, status_color = "EXECUTING...", (0, 165, 255)
        elif state.detection_enabled:
            status, status_color = "DETECTING (Active)", (0, 255, 0)
        else:
            status, status_color = "STANDBY (Press SPACE/G)", (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        if commands:
            cmd = commands.popleft()
            if cmd == 'Q':
                log("Quit requested")
                state.running = False
            elif cmd == 'G':
                state.detection_enabled = True
                log("\n>>> Detection ENABLED - Place red object in zone...")
            elif cmd in ZONE_TEST_KEYS:
                executor.execute_zone_action(ZONE_TEST_KEYS[cmd])
            elif cmd in MANUAL_MOVES:
                transport.send(MANUAL_MOVES[cmd])

        if display(frame):
            state.running = False

    return state


def main():
    print("\n" + "=" * 60)
    print("ESP32 4-ZONE HARDCODED TRASH COLLECTOR")
    print("=" * 60)

    # connect to the robot, if an address is configured
    car, connected = None, False
    if config.ESP_MAC:
        car = ESP32AutoConnect(config.ESP_MAC, config.RFCOMM_PORT)
        try:
            connected = car.connect()
        except Exception as e:
            print(f"Connection error: {e}")
    else:
        print("ESP_MAC is not set.")
    transport = BluetoothTransport(car) if connected else DryRunTransport()
    if not connected:
        print(" No serial - DRY-RUN mode")

    cap = cv2.VideoCapture(config.VIDEO_SRC)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {config.VIDEO_SRC}")
        return

    state, commands = LiveState(), deque()
    threading.Thread(target=keyboard_thread, args=(state, commands), daemon=True).start()

    try:
        run_live(cap, transport, commands=commands, state=state)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.2)
        if car:
            car.cleanup()
        print("Exited")
