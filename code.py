#!/usr/bin/env python3
"""
Run with: sudo python3 esp32_autocar_tracker.py
"""

import cv2
import numpy as np
import time
from collections import deque
import threading
import sys
import os
import subprocess

#optional
try:
    import serial
except Exception:
    serial = None

#config
VIDEO_SRC = 0                    #camera
ESP_MAC = "F8:B3:B7:21:F2:62"    #mac
RFCOMM_PORT = "/dev/rfcomm0"     #rfcomm

#timing
FORWARD_TIME = 1.0               #forward
BACKWARD_TIME = 1.0              #backward
TURN_90_TIME = 0.5               #turn
SIDE_FORWARD_TIME = 1.0          #sideforward

#hsv
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

#minarea
MIN_RED_AREA = 500               #minarea

#cooldown
ACTION_COOLDOWN = 3.0            #cooldownsec

#dryrun
DRY_RUN_IF_NO_SERIAL = True

#bluetooth
class ESP32AutoConnect:
    """Bluetooth connection handler"""
    
    def __init__(self, mac_address, port):
        self.mac = mac_address
        self.port = port
        self.ser = None

    def setup_connection(self):
        """Setup rfcomm binding"""
        try:
            print("Releasing existing bindings...")
            subprocess.run(['sudo', 'rfcomm', 'release', self.port],
                           stderr=subprocess.DEVNULL)
            time.sleep(1)
            
            print(f"Binding {self.mac} to {self.port}...")
            result = subprocess.run(
                ['sudo', 'rfcomm', 'bind', self.port, self.mac, '1'],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"Bind failed: {result.stderr}")
                return False
            print(" Bound successfully")
            time.sleep(1.2)
            
            subprocess.run(['sudo', 'chmod', '666', self.port],
                           stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            print(f"Setup error: {e}")
            return False

    def connect(self):
        """Open serial connection"""
        if not self.setup_connection():
            return False
        if not os.path.exists(self.port):
            print(f" Port not created")
            return False
        if serial is None:
            print("pyserial not installed")
            return False
        try:
            print(f"Opening serial on {self.port}...")
            self.ser = serial.Serial(port=self.port, baudrate=115200, timeout=2)
            time.sleep(1.0)
            self.send_command('S')
            print("Connected!")
            return True
        except Exception as e:
            print(f" Connection failed: {e}")
            return False

    def send_command(self, cmd):
        """Send single character command"""
        if not self.ser or not getattr(self.ser, "is_open", False):
            return False
        try:
            self.ser.write(cmd.encode())
            self.ser.flush()
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        if self.ser and getattr(self.ser, "is_open", False):
            try:
                self.send_command('S')
                time.sleep(0.1)
                self.ser.close()
            except:
                pass
        try:
            subprocess.run(['sudo', 'rfcomm', 'release', self.port], 
                         stderr=subprocess.DEVNULL)
        except:
            pass
        print(" Cleanup done")

#zonedefinitions
class ZoneManager:
    """
    Manages 4 hardcoded zones and car placement area.
    Car is always at BOTTOM CENTER.
    """
    
    def __init__(self, frame_width, frame_height):
        self.w = frame_width
        self.h = frame_height
        
        #placement
        self.car_x = frame_width // 2
        self.car_y = int(frame_height * 0.85)  #placement
        self.car_zone_size = 60
        
        #zones
        zone_top = int(frame_height * 0.3)
        zone_bottom = int(frame_height * 0.7)
        zone_left = int(frame_width * 0.25)
        zone_right = int(frame_width * 0.75)
        
        #front
        self.zones = {
            'FRONT': {
                'x1': int(frame_width * 0.35),
                'y1': 0,
                'x2': int(frame_width * 0.65),
                'y2': zone_top,
                'color': (0, 255, 0),      #green
                'name': 'FRONT'
            },
            'BACK': {
                'x1': int(frame_width * 0.35),
                'y1': zone_bottom,
                'x2': int(frame_width * 0.65),
                'y2': frame_height,
                'color': (255, 0, 0),      #blue
                'name': 'BACK'
            },
            'LEFT': {
                'x1': 0,
                'y1': zone_top,
                'x2': zone_left,
                'y2': zone_bottom,
                'color': (255, 255, 0),    #cyan
                'name': 'LEFT'
            },
            'RIGHT': {
                'x1': zone_right,
                'y1': zone_top,
                'x2': frame_width,
                'y2': zone_bottom,
                'color': (0, 255, 255),    #yellow
                'name': 'RIGHT'
            }
        }
    
    def get_zone_for_position(self, x, y):
        """Determine which zone the position falls into"""
        for zone_name, zone in self.zones.items():
            if (zone['x1'] <= x <= zone['x2'] and 
                zone['y1'] <= y <= zone['y2']):
                return zone_name
        return None
    
    def draw_zones(self, frame):
        """Draw all zones and car placement on frame"""
        #draw
        for zone in self.zones.values():
            cv2.rectangle(frame, 
                         (zone['x1'], zone['y1']),
                         (zone['x2'], zone['y2']),
                         zone['color'], 2)
            
            #label
            label_x = (zone['x1'] + zone['x2']) // 2 - 30
            label_y = (zone['y1'] + zone['y2']) // 2
            cv2.putText(frame, zone['name'], (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, zone['color'], 3)
        
        #placementbox
        cv2.rectangle(frame,
                     (self.car_x - self.car_zone_size, 
                      self.car_y - self.car_zone_size),
                     (self.car_x + self.car_zone_size, 
                      self.car_y + self.car_zone_size),
                     (255, 255, 255), 3)
        
        #orientation
        #front
        front_y = self.car_y - self.car_zone_size - 20
        cv2.arrowedLine(frame, 
                       (self.car_x, self.car_y - self.car_zone_size),
                       (self.car_x, front_y),
                       (0, 255, 0), 3)
        cv2.putText(frame, "FRONT", (self.car_x + 10, front_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        #back
        back_y = self.car_y + self.car_zone_size + 30
        cv2.arrowedLine(frame,
                       (self.car_x, self.car_y + self.car_zone_size),
                       (self.car_x, back_y),
                       (255, 0, 0), 3)
        cv2.putText(frame, "BACK", (self.car_x + 10, back_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        #placetext
        cv2.putText(frame, "PLACE CAR HERE", 
                   (self.car_x - 80, self.car_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#detector
class RedTrashDetector:
    """Detects red trash objects"""
    
    def __init__(self):
        self.last_detection = None
        
    def detect_red(self, frame):
        """Detect red object and return centroid"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #mask
        mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        red_mask = mask1 | mask2
        
        #cleanup
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        #contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        #largest
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < MIN_RED_AREA:
            return None
        
        #centroid
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        self.last_detection = (cx, cy)
        return (cx, cy)

#executor
class ActionExecutor:
    """
    Executes hardcoded movement sequences for each zone.
    """
    
    def __init__(self, car_controller):
        self.car = car_controller
        self.executing = False
        self.last_action_time = 0
        
    def can_execute(self):
        """Check if enough time passed since last action"""
        if self.executing:
            return False
        
        now = time.time()
        if now - self.last_action_time < ACTION_COOLDOWN:
            return False
        
        return True
    
    def send_and_wait(self, cmd, duration, desc):
        """Send command and wait for duration"""
        print(f"   {cmd} for {duration:.1f}s ({desc})")
        
        if self.car and getattr(self.car, "ser", None):
            self.car.send_command(cmd)
        elif DRY_RUN_IF_NO_SERIAL:
            print(f"  [DRY] {cmd}")
        
        time.sleep(duration)
        
        #stop
        if self.car and getattr(self.car, "ser", None):
            self.car.send_command('S')
        elif DRY_RUN_IF_NO_SERIAL:
            print(f"  [DRY] S (stop)")
        
        time.sleep(0.2)  #pause
    
    def execute_zone_action(self, zone_name):
        """Execute movement for specific zone"""
        if not self.can_execute():
            return
        
        self.executing = True
        print(f"\n{'='*50}")
        print(f"EXECUTING ACTION: {zone_name}")
        print(f"{'='*50}")
        
        try:
            if zone_name == 'FRONT':
                #forward
                self.send_and_wait('U', FORWARD_TIME, "Move forward to FRONT zone")
                
            elif zone_name == 'BACK':
                #backward
                self.send_and_wait('D', BACKWARD_TIME, "Move backward to BACK zone")
                
            elif zone_name == 'LEFT':
                #leftturn
                self.send_and_wait('L', TURN_90_TIME, "Turn 90° LEFT")
                self.send_and_wait('U', SIDE_FORWARD_TIME, "Move forward to LEFT zone")
                
            elif zone_name == 'RIGHT':
                #rightturn
                self.send_and_wait('R', TURN_90_TIME, "Turn 90° RIGHT")
                self.send_and_wait('U', SIDE_FORWARD_TIME, "Move forward to RIGHT zone")
            
            print(f"ACTION COMPLETE: {zone_name}")
            print(f" Return car to starting position")
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self.executing = False
            self.last_action_time = time.time()

#state
running = True
manual_cmd_queue = deque()
connected = False
car = None
detection_enabled = False  #state

#input
def manual_input_thread():
    """Manual keyboard control for testing"""
    global running, detection_enabled
    print("\n=== Manual Control (Testing) ===")
    print("SPACEBAR or G = Enable detection and movement")
    print("W=Forward, S=Backward, A=Left, D=Right, X=Stop, Q=Quit")
    print("1=Test FRONT, 2=Test BACK, 3=Test LEFT, 4=Test RIGHT")
    
    try:
        while running:
            line = input().strip()
            if not line:
                continue
            cmd = line[0].upper()
            
            if cmd == 'Q':
                manual_cmd_queue.append('Q')
                break
            elif cmd in ['W', 'A', 'S', 'D', 'X', '1', '2', '3', '4']:
                manual_cmd_queue.append(cmd)
            elif cmd == 'G' or cmd == ' ':
                #enable
                manual_cmd_queue.append('G')
                
    except Exception:
        pass
    finally:
        running = False

#send
def send_command(cmd):
    """Send command"""
    global connected, car
    
    if connected and car and getattr(car, "ser", None):
        car.send_command(cmd)
    elif DRY_RUN_IF_NO_SERIAL:
        print(f"[DRY] {cmd}")

#main
def main():
    global running, connected, car, detection_enabled

    print("\n" + "="*60)
    print("ESP32 4-ZONE HARDCODED TRASH COLLECTOR")
    print("="*60)

    #connect
    car = ESP32AutoConnect(ESP_MAC, RFCOMM_PORT)
    try:
        connected = car.connect()
    except Exception as e:
        connected = False
        print(f"Connection error: {e}")

    if not connected:
        print(" No serial - DRY-RUN mode")

    #thread
    threading.Thread(target=manual_input_thread, daemon=True).start()

    #camera
    cap = cv2.VideoCapture(VIDEO_SRC)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {VIDEO_SRC}")
        return

    #framesize
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from camera")
        return
    h, w = frame.shape[:2]

    #init
    zone_manager = ZoneManager(w, h)
    detector = RedTrashDetector()
    executor = ActionExecutor(car)
    
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS:")
    print("="*60)
    print("1. Place your ESP32 car in the WHITE BOX at bottom center")
    print("2. Orient car so FRONT faces UP (green arrow)")
    print("3. Press SPACEBAR or 'G' in terminal to enable detection")
    print("4. Throw red trash into one of the 4 colored zones")
    print("5. Car will execute movement for that zone")
    print("6. Detection auto-disables after movement completes")
    print("7. Return car to white box and press SPACEBAR/'G' again")
    print("="*60 + "\n")

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            #draw
            zone_manager.draw_zones(frame)
            
            #detect
            if detection_enabled:
                #detectred
                red_pos = detector.detect_red(frame)
                
                if red_pos:
                    rx, ry = red_pos
                    
                    #drawred
                    cv2.circle(frame, (rx, ry), 15, (0, 0, 255), 3)
                    cv2.circle(frame, (rx, ry), 5, (0, 0, 255), -1)
                    
                    #zone
                    zone = zone_manager.get_zone_for_position(rx, ry)
                    
                    if zone:
                        cv2.putText(frame, f"DETECTED: {zone}", (rx + 20, ry),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        #exec
                        if executor.can_execute():
                            #disable
                            detection_enabled = False
                            print("\n>>> Detection DISABLED - Movement starting...")
                            
                            #threadexec
                            threading.Thread(
                                target=executor.execute_zone_action,
                                args=(zone,),
                                daemon=True
                            ).start()
            
            #status
            if executor.executing:
                status = "EXECUTING..."
                status_color = (0, 165, 255)
            elif detection_enabled:
                status = "DETECTING (Active)"
                status_color = (0, 255, 0)
            else:
                status = "STANDBY (Press SPACE/G)"
                status_color = (0, 0, 255)
            
            cv2.putText(frame, f"Status: {status}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
            
            #manual
            if manual_cmd_queue:
                cmd = manual_cmd_queue.popleft()
                if cmd == 'Q':
                    print("Quit requested")
                    running = False
                elif cmd == 'G':
                    #toggle
                    detection_enabled = True
                    print("\n>>> Detection ENABLED - Place red object in zone...")
                elif cmd == '1':
                    executor.execute_zone_action('FRONT')
                elif cmd == '2':
                    executor.execute_zone_action('BACK')
                elif cmd == '3':
                    executor.execute_zone_action('LEFT')
                elif cmd == '4':
                    executor.execute_zone_action('RIGHT')
                elif cmd in ['W', 'A', 'S', 'D', 'X']:
                    #manualcontrol
                    cmd_map = {'W': 'U', 'A': 'L', 'S': 'D', 'D': 'R', 'X': 'S'}
                    send_command(cmd_map.get(cmd, 'S'))

            cv2.imshow("ESP32 4-Zone Trash Collector", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

    except KeyboardInterrupt:
        print("\nInterrupted")
        running = False
    finally:
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.2)
        if car:
            car.cleanup()
        print("Exited")

if __name__ == "__main__":
    main()
