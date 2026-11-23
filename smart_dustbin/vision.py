"""Seeing: where the camera frame is divided into zones, and how a red object is found in it.

This is the original logic, moved here without changing what it does, so it can be tested and simulated without a camera.
Two small additions, both off by default: `ZoneManager.car_box()` to read the car's placement box, and an optional
keep-out rectangle for the detector (see `RedTrashDetector`).
"""
import cv2
import numpy as np

from . import config


class ZoneManager:
    """Four hard-coded zones (FRONT, BACK, LEFT, RIGHT) in image space, and the car's placement box.

    The car is placed at the BOTTOM CENTRE of the frame, facing up. Zones are rectangles measured from the frame size.
    """

    def __init__(self, frame_width, frame_height):
        self.w = frame_width
        self.h = frame_height

        # placement
        self.car_x = frame_width // 2
        self.car_y = int(frame_height * config.CAR_Y_RATIO)
        self.car_zone_size = config.CAR_ZONE_SIZE

        # zones
        zone_top = int(frame_height * 0.3)
        zone_bottom = int(frame_height * 0.7)
        zone_left = int(frame_width * 0.25)
        zone_right = int(frame_width * 0.75)

        self.zones = {
            'FRONT': {
                'x1': int(frame_width * 0.35), 'y1': 0,
                'x2': int(frame_width * 0.65), 'y2': zone_top,
                'color': (0, 255, 0),      # green
                'name': 'FRONT'
            },
            'BACK': {
                'x1': int(frame_width * 0.35), 'y1': zone_bottom,
                'x2': int(frame_width * 0.65), 'y2': frame_height,
                'color': (255, 0, 0),      # blue
                'name': 'BACK'
            },
            'LEFT': {
                'x1': 0, 'y1': zone_top,
                'x2': zone_left, 'y2': zone_bottom,
                'color': (255, 255, 0),    # cyan
                'name': 'LEFT'
            },
            'RIGHT': {
                'x1': zone_right, 'y1': zone_top,
                'x2': frame_width, 'y2': zone_bottom,
                'color': (0, 255, 255),    # yellow
                'name': 'RIGHT'
            }
        }

    def get_zone_for_position(self, x, y):
        """Which zone a point falls into, or None if it is in a gap between zones."""
        for zone_name, zone in self.zones.items():
            if (zone['x1'] <= x <= zone['x2'] and
                    zone['y1'] <= y <= zone['y2']):
                return zone_name
        return None

    def car_box(self):
        """The car's placement box as (x1, y1, x2, y2)."""
        s = self.car_zone_size
        return (self.car_x - s, self.car_y - s, self.car_x + s, self.car_y + s)

    def draw_zones(self, frame):
        """Draw all zones and the car placement box on the frame."""
        for zone in self.zones.values():
            cv2.rectangle(frame, (zone['x1'], zone['y1']), (zone['x2'], zone['y2']), zone['color'], 2)

            label_x = (zone['x1'] + zone['x2']) // 2 - 30
            label_y = (zone['y1'] + zone['y2']) // 2
            cv2.putText(frame, zone['name'], (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, zone['color'], 3)

        # placement box
        cv2.rectangle(frame,
                      (self.car_x - self.car_zone_size, self.car_y - self.car_zone_size),
                      (self.car_x + self.car_zone_size, self.car_y + self.car_zone_size),
                      (255, 255, 255), 3)

        # orientation arrows
        front_y = self.car_y - self.car_zone_size - 20
        cv2.arrowedLine(frame, (self.car_x, self.car_y - self.car_zone_size), (self.car_x, front_y), (0, 255, 0), 3)
        cv2.putText(frame, "FRONT", (self.car_x + 10, front_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        back_y = self.car_y + self.car_zone_size + 30
        cv2.arrowedLine(frame, (self.car_x, self.car_y + self.car_zone_size), (self.car_x, back_y), (255, 0, 0), 3)
        cv2.putText(frame, "BACK", (self.car_x + 10, back_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(frame, "PLACE CAR HERE", (self.car_x - 80, self.car_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


class RedTrashDetector:
    """Finds the largest red blob in a frame and returns its centre.

    Classical computer vision, no machine learning: threshold in HSV (two hue ranges, because red wraps around),
    clean the mask with a morphological open and close, take the largest contour, and ignore it if it is smaller than
    `min_area` pixels squared.

    `ignore_rect=(x1, y1, x2, y2)` is an optional keep-out region (for example the car's own placement box) whose pixels
    are never counted. It is None by default, which is the original behaviour.
    """

    def __init__(self, min_area=None, ignore_rect=None):
        self.min_area = config.MIN_RED_AREA if min_area is None else min_area
        self.ignore_rect = ignore_rect
        self.last_detection = None
        self.last_area = 0

    def red_mask(self, frame):
        """The cleaned-up binary mask of red pixels."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, config.RED_LOWER1, config.RED_UPPER1)
        mask2 = cv2.inRange(hsv, config.RED_LOWER2, config.RED_UPPER2)
        red_mask = mask1 | mask2

        if self.ignore_rect is not None:
            x1, y1, x2, y2 = self.ignore_rect
            red_mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        return red_mask

    def detect_red(self, frame):
        """Return the (x, y) centre of the largest red object, or None."""
        red_mask = self.red_mask(frame)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area:
            return None

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        self.last_detection = (cx, cy)
        self.last_area = area
        return (cx, cy)
