"""Synthetic camera scenes: a stand-in for the real camera, so the detection code can be exercised without hardware.

A scene is a top-down view of a floor with the car's placement box at the bottom centre, red "trash" blobs, and optional
look-alike distractors and camera problems (dim light, glare, noise, motion blur). Everything is random but seeded, so a
run can be repeated exactly.

IMPORTANT: this is a model, not the real camera. Results measured on these scenes say how the detection code behaves on
the situations built here, not how accurate it is on a real floor under real lighting.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .vision import ZoneManager

FRAME_W, FRAME_H = 640, 480

# Blob colours as (hue 0-179, saturation, value) in OpenCV's HSV. The detector accepts hue 0-10 and 170-180,
# saturation >= 120 and value >= 70.
TRASH_RED = "red trash"
DISTRACTORS = {
    "orange": ((12, 22), (190, 255), (180, 255)),         # hue clearly outside the red range
    "brown": ((12, 20), (150, 200), (80, 130)),
    "skin tone": ((5, 12), (60, 100), (170, 230)),        # right hue, too little saturation
    "pink": ((168, 176), (60, 105), (200, 255)),          # right hue, too little saturation
    "dark red": ((0, 4), (200, 255), (35, 60)),           # right hue, too dark
    "orange-red": ((6, 14), (190, 255), (180, 255)),      # right at the edge of the accepted hue range
}


@dataclass
class Blob:
    """A filled disc in the scene."""
    x: int
    y: int
    radius: int
    hsv: tuple
    label: str = ""

    @property
    def area(self):
        return float(np.pi * self.radius ** 2)


def hsv_to_bgr(h, s, v):
    pixel = np.uint8([[[int(h), int(s), int(v)]]])
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def red_hsv(rng):
    """A random red inside the range the detector accepts, either side of the hue wrap-around."""
    hue = int(rng.integers(0, 7)) if rng.random() < 0.5 else int(rng.integers(174, 180))
    return hue, int(rng.integers(170, 256)), int(rng.integers(130, 231))


def distractor_hsv(rng, kind):
    (h0, h1), (s0, s1), (v0, v1) = DISTRACTORS[kind]
    return int(rng.integers(h0, h1 + 1)), int(rng.integers(s0, s1 + 1)), int(rng.integers(v0, v1 + 1))


def floor_background(rng, w=FRAME_W, h=FRAME_H):
    """A plain, slightly uneven grey floor with a soft lighting gradient."""
    base = rng.uniform(135, 165)
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = 1 + 0.12 * (xx / w - 0.5) + 0.10 * (yy / h - 0.5)
    blotches = cv2.resize(cv2.GaussianBlur(rng.normal(0, 1, (h // 8, w // 8)).astype(np.float32), (0, 0), 3),
                          (w, h), interpolation=cv2.INTER_CUBIC) * 14
    gray = np.clip(base * gradient + blotches, 0, 255)
    return np.dstack([gray * 0.95, gray * 0.97, gray]).astype(np.uint8)


def draw_car(frame, zone_manager, red_patch=0):
    """The robot parked in its placement box: a dark body with a light front stripe. `red_patch` is the side of a red
    square stuck on it (for example a red tape or LED cover), in pixels."""
    cx, cy = zone_manager.car_x, zone_manager.car_y
    cv2.rectangle(frame, (cx - 36, cy - 50), (cx + 36, cy + 50), (50, 50, 55), -1)
    cv2.rectangle(frame, (cx - 36, cy - 50), (cx + 36, cy - 38), (150, 150, 155), -1)
    if red_patch:
        half = red_patch // 2
        cv2.rectangle(frame, (cx - half, cy - half + 8), (cx + half, cy + half + 8), hsv_to_bgr(2, 230, 200), -1)


def draw_blob(frame, blob):
    color = hsv_to_bgr(*blob.hsv)
    cv2.circle(frame, (blob.x, blob.y), blob.radius, color, -1, cv2.LINE_AA)
    # a soft highlight so it looks like a lit object, not a flat disc
    lighter = hsv_to_bgr(blob.hsv[0], max(blob.hsv[1] - 25, 0), min(blob.hsv[2] + 20, 255))
    cv2.circle(frame, (blob.x - blob.radius // 4, blob.y - blob.radius // 4), max(blob.radius // 2, 1), lighter, -1, cv2.LINE_AA)


def render_scene(rng, blobs=(), *, brightness=1.0, washout=0.0, noise_sigma=0.0, salt_pepper=0.0,
                 motion_blur=0, car=True, car_red_patch=0, background=None):
    """Draw a scene.

    brightness: multiplies every pixel (below 1 is dim light). washout: 0-1 blend towards white (glare, overexposure).
    noise_sigma: Gaussian sensor noise. salt_pepper: fraction of pixels turned black or white.
    motion_blur: length in pixels of a horizontal smear (a thrown object, or camera shake).
    background: reuse this floor image instead of drawing a new one (a real camera sees the same floor every frame).
    """
    zone_manager = ZoneManager(FRAME_W, FRAME_H)
    frame = background.copy() if background is not None else floor_background(rng)
    if car:
        draw_car(frame, zone_manager, car_red_patch)
    for blob in blobs:
        draw_blob(frame, blob)

    if motion_blur > 1:
        kernel = np.zeros((motion_blur, motion_blur), np.float32)
        kernel[motion_blur // 2, :] = 1.0 / motion_blur
        frame = cv2.filter2D(frame, -1, kernel)

    out = frame.astype(np.float32) * brightness
    if washout:
        out = out * (1 - washout) + 255 * washout
    if noise_sigma:
        out += rng.normal(0, noise_sigma, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if salt_pepper:
        mask = rng.random(out.shape[:2])
        out[mask < salt_pepper / 2] = 0
        out[mask > 1 - salt_pepper / 2] = 255
    return out


# ---------------------------------------------------------------------------------------------------- sampling helpers
def random_point_in_zone(rng, zone_manager, zone_name, radius, avoid_car=True):
    """A point inside a zone, far enough from the frame edge for a blob of `radius` to stay fully visible,
    and (by default) not on top of the car."""
    zone = zone_manager.zones[zone_name]
    cx1, cy1, cx2, cy2 = zone_manager.car_box()
    for _ in range(1000):
        x = int(rng.integers(max(zone['x1'], radius + 2), min(zone['x2'], FRAME_W - radius - 2) + 1))
        y = int(rng.integers(max(zone['y1'], radius + 2), min(zone['y2'], FRAME_H - radius - 2) + 1))
        if avoid_car and cx1 - radius <= x <= cx2 + radius and cy1 - radius <= y <= cy2 + radius:
            continue
        return x, y
    raise RuntimeError(f"could not place a blob in zone {zone_name}")


def random_point_in_gap(rng, zone_manager, radius):
    """A point that belongs to no zone and is not on the car (for example the middle of the floor)."""
    cx1, cy1, cx2, cy2 = zone_manager.car_box()
    for _ in range(5000):
        x = int(rng.integers(radius + 2, FRAME_W - radius - 2))
        y = int(rng.integers(radius + 2, FRAME_H - radius - 2))
        if zone_manager.get_zone_for_position(x, y) is None and not (cx1 - radius <= x <= cx2 + radius and cy1 - radius <= y <= cy2 + radius):
            return x, y
    raise RuntimeError("could not find a gap position")


class SyntheticCamera:
    """Plays a list of frames through the same read() interface as cv2.VideoCapture, then repeats the last one."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0

    def read(self):
        frame = self.frames[min(self.index, len(self.frames) - 1)].copy()
        self.index += 1
        return True, frame

    def release(self):
        pass
