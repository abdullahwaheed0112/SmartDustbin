"""Run the real program against a software camera and a software robot, and animate what would happen.

What is real here: the detector, the zone logic, the movement plans and the live loop (`run_live`) are the program's own code.
What is modelled: the camera frames (see scenes.py) and the robot's motion.

The motion model is deliberately simple and says so on the animation: the robot drives straight for as long as a forward or
backward command is held and turns at a fixed rate while a turn command is held. The turn rate comes from the original
author's own timing (a 90 degree turn takes 0.5 s, so 180 degrees per second). The forward speed is NOT known: the real
robot's speed depends on its motors, battery and floor, so it is a parameter with an illustrative default.
"""
import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from . import config
from .actions import COMMAND_MEANING, ActionExecutor, plan_for
from .app import run_live
from .scenes import FRAME_H, FRAME_W, Blob, SyntheticCamera, floor_background, random_point_in_zone, red_hsv, render_scene
from .transport import RecordingTransport
from .vision import RedTrashDetector, ZoneManager, locate_zone

DEFAULT_SPEED_PX_S = 200.0  # illustrative only: how many image pixels per second the bin covers while moving


class VirtualClock:
    """A clock whose sleep() just moves time forward, so a one-second movement costs no real time."""

    def __init__(self, start=1000.0):
        self.t = start

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def turn_rate_deg_s():
    """Degrees per second while a turn command is held, from the original timing: 90 degrees in TURN_90_TIME seconds."""
    return 90.0 / config.TURN_90_TIME


@dataclass
class Pose:
    x: float
    y: float
    heading: float  # degrees clockwise from "up" in the image (up is where the car's front points at the start)


def integrate_path(events, start, speed, turn_rate, dt=0.05):
    """Follow a list of (time, command) events and return [(time, Pose)] samples.

    A command stays active until the next event: U forward, D backward, L turn left, R turn right, S stop.
    """
    pose = Pose(start.x, start.y, start.heading)
    samples = []
    if not events:
        return samples
    t = events[0][0]
    samples.append((t, Pose(pose.x, pose.y, pose.heading)))
    for (t0, command), (t1, _) in zip(events, events[1:]):
        t = t0
        while t < t1 - 1e-9:
            step = min(dt, t1 - t)
            rad = math.radians(pose.heading)
            if command == 'U':
                pose.x += speed * math.sin(rad) * step
                pose.y -= speed * math.cos(rad) * step
            elif command == 'D':
                pose.x -= speed * math.sin(rad) * step
                pose.y += speed * math.cos(rad) * step
            elif command == 'L':
                pose.heading -= turn_rate * step
            elif command == 'R':
                pose.heading += turn_rate * step
            t += step
            samples.append((t, Pose(pose.x, pose.y, pose.heading)))
    return samples


def draw_car_pose(frame, pose):
    """The robot as a rotated rectangle with a light front stripe."""
    rad = math.radians(pose.heading)
    cos, sin = math.cos(rad), math.sin(rad)

    def to_image(dx, dy):  # dx to the car's right, dy towards its back
        return [int(round(pose.x + dx * cos - dy * sin)), int(round(pose.y + dx * sin + dy * cos))]

    body = np.array([to_image(-36, -50), to_image(36, -50), to_image(36, 50), to_image(-36, 50)], np.int32)
    front = np.array([to_image(-36, -50), to_image(36, -50), to_image(36, -38), to_image(-36, -38)], np.int32)
    cv2.fillPoly(frame, [body], (50, 50, 55))
    cv2.fillPoly(frame, [front], (150, 150, 155))


@dataclass
class DemoResult:
    zone: str
    object_position: tuple
    detected: tuple
    detected_zone: str
    commands: list
    final_pose: Pose
    offset_px: float
    log: list = field(default_factory=list)
    frames: list = field(default_factory=list)   # BGR frames for the animation
    durations_ms: list = field(default_factory=list)


CAPTION_HEIGHT = 62


def with_caption(frame, lines, color=(255, 255, 255)):
    """The frame with a dark strip underneath carrying up to three lines of text (so nothing in the image is covered)."""
    bar = np.full((CAPTION_HEIGHT, frame.shape[1], 3), 28, np.uint8)
    for i, text in enumerate(lines[:3]):
        cv2.putText(bar, text, (10, 20 + i * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return np.vstack([frame, bar])


def run_demo(zone, seed=7, speed=DEFAULT_SPEED_PX_S, radius=18):
    """Throw a red object into `zone` of a synthetic scene, run the real program on it, and animate the result."""
    rng = np.random.default_rng(seed)
    zone_manager = ZoneManager(FRAME_W, FRAME_H)
    background = floor_background(rng)
    x, y = random_point_in_zone(rng, zone_manager, zone, radius)
    blob = Blob(x, y, radius, (0, 235, 215), "red trash")  # a clear, bright red so the demo reads well

    empty = render_scene(rng, background=background, noise_sigma=3)
    thrown = render_scene(rng, [blob], background=background, noise_sigma=3)
    camera = SyntheticCamera([empty] * 8 + [thrown] * 4)

    # the real live loop, on a synthetic camera, with a recording robot and a virtual clock
    clock = VirtualClock()
    transport = RecordingTransport(clock.now)
    log = []
    executor = ActionExecutor(transport, clock=clock.now, sleep=clock.sleep, log=log.append)
    shown = []
    run_live(camera, transport, executor=executor, commands=deque(['G']), display=lambda f: shown.append(f.copy()) or False,
             max_frames=11, threaded=False, log=log.append)

    detected, detected_zone = locate_zone(thrown, RedTrashDetector(), zone_manager)
    events = [(t - transport.events[0][0], c) for t, c in transport.events]
    start = Pose(zone_manager.car_x, zone_manager.car_y, 0.0)
    path = integrate_path([(t, c) for t, c in events], start, speed, turn_rate_deg_s())
    final = path[-1][1] if path else start
    offset = math.hypot(final.x - x, final.y - y)

    # ---- the animation: what the camera showed, then the robot driving
    frames, durations = [], []
    for i, f in enumerate(shown):
        frames.append(with_caption(f, ["SIMULATION: a synthetic camera feeds the program's own detection, zone and",
                                       "movement code. Nothing here was recorded from the real robot."]))
        durations.append(140)
    durations[-1] = 1400

    plan = plan_for(detected_zone) or []
    plan_text = " then ".join(f"{s.command} ({COMMAND_MEANING[s.command]}) {s.duration:.1f}s" for s in plan)
    base = render_scene(rng, [blob], background=background, noise_sigma=3, car=False)
    trail = []
    command_at = lambda t: next((c for tt, c in reversed(events) if tt <= t + 1e-9), 'S')
    for t, pose in path[::2]:
        frame = base.copy()
        zone_manager.draw_zones(frame)
        trail.append((int(pose.x), int(pose.y)))
        if len(trail) > 1:
            cv2.polylines(frame, [np.array(trail, np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
        draw_car_pose(frame, pose)
        cmd = command_at(t)
        cv2.putText(frame, f"Status: EXECUTING  {cmd} ({COMMAND_MEANING[cmd]})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        frames.append(with_caption(frame, [f"Plan for {detected_zone}: {plan_text}",
                                           f"Illustrative motion: assumed {speed:.0f} px/s forward, {turn_rate_deg_s():.0f} deg/s turning",
                                           "(the real speed is not known)"]))
        durations.append(100)

    final_frame = base.copy()
    zone_manager.draw_zones(final_frame)
    cv2.polylines(final_frame, [np.array(trail, np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA) if len(trail) > 1 else None
    draw_car_pose(final_frame, final)
    cv2.putText(final_frame, "Status: ACTION COMPLETE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    frames.append(with_caption(final_frame, [f"Bin ends {offset:.0f} px from the object (open loop: it cannot steer)",
                                             f"Illustrative motion: assumed {speed:.0f} px/s, {turn_rate_deg_s():.0f} deg/s turning",
                                             "(the real speed is not known)"]))
    durations.append(2200)

    return DemoResult(zone=zone, object_position=(x, y), detected=detected, detected_zone=detected_zone,
                      commands=events, final_pose=final, offset_px=offset, log=log, frames=frames, durations_ms=durations)


def save_gif(frames, durations_ms, path, scale=0.7):
    """Write BGR frames as an animated GIF. Each frame gets its own 256-colour palette from octree quantization, which keeps
    small but distinct colours (the red object, the coloured zone outlines) true instead of averaging them into the grey."""
    images = []
    for frame in frames:
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rgb = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        images.append(rgb.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE))
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations_ms, loop=0, optimize=True)
    return path
