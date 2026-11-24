"""Measure the detection and zone logic on hundreds of synthetic scenes, and analyse the zone layout geometrically.

Every number produced here comes from SIMULATED scenes (see scenes.py). It shows how the code behaves in the situations that
were built, including some it is known to handle badly. It is not the accuracy of the real robot on a real floor.

The one question asked of every scene: does the program make the right decision? The right decision is the zone the red
trash is in, or "do nothing" when there is no trash in a zone (nothing red, a look-alike, a red mark on the car, or trash in
a gap between zones).
"""
import json
from dataclasses import dataclass

import cv2
import numpy as np

from .scenes import (DISTRACTORS, FRAME_H, FRAME_W, Blob, distractor_hsv, random_point_in_gap, random_point_in_zone,
                     red_hsv, render_scene)
from .vision import RedTrashDetector, ZoneManager, locate_zone

ZONES = ["FRONT", "BACK", "LEFT", "RIGHT"]
LOOKALIKES = ["orange", "brown", "skin tone", "pink", "dark red"]


@dataclass
class Scene:
    frame: np.ndarray
    expected: str  # the zone the program should act on, or None for "do nothing"
    target: tuple = None  # (x, y) of the trash, if any


# ------------------------------------------------------------------------------------------------ scene makers
def _trash(rng, zm, zone=None, radius=None):
    zone = zone or ZONES[int(rng.integers(0, 4))]
    radius = radius or int(rng.integers(14, 25))
    x, y = random_point_in_zone(rng, zm, zone, radius)
    return zone, Blob(x, y, radius, red_hsv(rng), "red trash")


def _positive(**camera):
    """Red trash somewhere in a random zone, under the given camera conditions (each value is a function of rng)."""
    def make(rng, zm):
        zone, blob = _trash(rng, zm)
        kwargs = {k: (v(rng) if callable(v) else v) for k, v in camera.items()}
        return Scene(render_scene(rng, [blob], **kwargs), zone, (blob.x, blob.y))
    return make


def _lookalikes_only(kinds):
    def make(rng, zm):
        blobs = []
        for _ in range(int(rng.integers(1, 4))):
            kind = kinds[int(rng.integers(0, len(kinds)))]
            r = int(rng.integers(15, 31))
            blobs.append(Blob(int(rng.integers(r + 2, FRAME_W - r - 2)), int(rng.integers(r + 2, FRAME_H - r - 2)), r,
                              distractor_hsv(rng, kind), kind))
        return Scene(render_scene(rng, blobs), None)
    return make


def _trash_among_lookalikes(rng, zm):
    zone, blob = _trash(rng, zm)
    blobs = [blob]
    for _ in range(int(rng.integers(1, 3))):
        kind = LOOKALIKES[int(rng.integers(0, len(LOOKALIKES)))]
        r = int(rng.integers(15, 31))
        blobs.append(Blob(int(rng.integers(r + 2, FRAME_W - r - 2)), int(rng.integers(r + 2, FRAME_H - r - 2)), r,
                          distractor_hsv(rng, kind), kind))
    blobs.sort(key=lambda b: b.label != "red trash")  # trash drawn first so look-alikes may partly cover it
    return Scene(render_scene(rng, blobs), zone, (blob.x, blob.y))


def _two_red_objects(rng, zm):
    zone_a, big = _trash(rng, zm, radius=int(rng.integers(20, 27)))
    other_zone = [z for z in ZONES if z != zone_a][int(rng.integers(0, 3))]
    _, small = _trash(rng, zm, zone=other_zone, radius=int(rng.integers(13, 17)))
    return Scene(render_scene(rng, [big, small]), zone_a, (big.x, big.y))  # the larger one should win


def _small_trash(rng, zm):
    zone, blob = _trash(rng, zm, radius=int(rng.integers(8, 12)))  # 200 to 380 square pixels: below the 500 limit
    return Scene(render_scene(rng, [blob]), zone, (blob.x, blob.y))


def _trash_in_a_gap(rng, zm):
    r = int(rng.integers(14, 25))
    x, y = random_point_in_gap(rng, zm, r)
    return Scene(render_scene(rng, [Blob(x, y, r, red_hsv(rng), "red trash")]), None, (x, y))


def _red_mark_on_car(rng, zm):
    return Scene(render_scene(rng, [], car_red_patch=30), None)


CONDITIONS = [
    ("clean scene", "red trash in a random zone, good light", _positive()),
    ("dim light", "image 45-80% as bright", _positive(brightness=lambda r: r.uniform(0.45, 0.80))),
    ("glare", "10-35% washed out towards white", _positive(washout=lambda r: r.uniform(0.10, 0.35))),
    ("sensor noise", "Gaussian noise (sigma 8-20) plus 0.5% salt-and-pepper", _positive(noise_sigma=lambda r: r.uniform(8, 20), salt_pepper=0.005)),
    ("motion blur", "9-21 pixel horizontal smear", _positive(motion_blur=lambda r: int(r.integers(9, 22)))),
    ("trash among look-alikes", "red trash plus orange, brown, skin-tone, pink or dark-red objects", _trash_among_lookalikes),
    ("two red objects", "the larger one should win", _two_red_objects),
    ("look-alikes only", "no trash: should do nothing", _lookalikes_only(LOOKALIKES)),
    ("orange-red look-alikes", "no trash; hue right at the edge of the accepted range", _lookalikes_only(["orange-red"])),
    ("small trash", "area 200-380 px^2, below the 500 px^2 minimum", _small_trash),
    ("trash outside every zone", "should do nothing", _trash_in_a_gap),
    ("red mark on the car", "30 x 30 px red patch on the robot, no trash: should do nothing", _red_mark_on_car),
]


# ------------------------------------------------------------------------------------------------ running them
def evaluate(n=100, seed=42):
    """Run every condition on `n` scenes with the original detector and with a keep-out mask over the car's own box."""
    zm = ZoneManager(FRAME_W, FRAME_H)
    detectors = {"original": RedTrashDetector(), "keepout": RedTrashDetector(ignore_rect=zm.car_box())}
    rows = []
    for index, (name, description, make) in enumerate(CONDITIONS):
        rng = np.random.default_rng(seed + index)
        stats = {key: {"correct": 0, "triggered": 0, "errors": []} for key in detectors}
        for _ in range(n):
            scene = make(rng, zm)
            for key, detector in detectors.items():
                position, zone = locate_zone(scene.frame, detector, zm)
                stats[key]["correct"] += zone == scene.expected
                stats[key]["triggered"] += zone is not None
                if scene.target and position and zone == scene.expected:
                    stats[key]["errors"].append(float(np.hypot(position[0] - scene.target[0], position[1] - scene.target[1])))
        row = {"condition": name, "description": description, "n": n}
        for key in detectors:
            errors = stats[key]["errors"]
            row[key] = {"accuracy": stats[key]["correct"] / n, "trigger_rate": stats[key]["triggered"] / n,
                        "mean_centroid_error_px": float(np.mean(errors)) if errors else None}
        rows.append(row)
    total = {key: sum(r[key]["accuracy"] for r in rows) / len(rows) for key in detectors}
    return {"seed": seed, "scenes_per_condition": n, "frame": [FRAME_W, FRAME_H], "conditions": rows, "mean_accuracy": total}


def format_markdown(results):
    """The result table as GitHub markdown."""
    lines = ["| Condition | What it tests | Right decision | With keep-out mask | Centre error |",
             "|---|---|---|---|---|"]
    for r in results["conditions"]:
        err = r["original"]["mean_centroid_error_px"]
        lines.append(f"| {r['condition']} | {r['description']} | {r['original']['accuracy']:.0%} | "
                     f"{r['keepout']['accuracy']:.0%} | {'' if err is None else f'{err:.1f} px'} |")
    return "\n".join(lines)


def conditions_montage(path, seed=3):
    """One example frame per condition, marked with the expected zone, the detection and the verdict."""
    zm = ZoneManager(FRAME_W, FRAME_H)
    detector = RedTrashDetector()
    tiles = []
    for index, (name, _, make) in enumerate(CONDITIONS):
        rng = np.random.default_rng(seed + index)
        scene = make(rng, zm)
        position, zone = locate_zone(scene.frame, detector, zm)
        frame = scene.frame.copy()
        zm.draw_zones(frame)
        if position:
            cv2.circle(frame, position, 20, (0, 0, 255), 3)
        ok = zone == scene.expected
        verdict = ("OK: " if ok else "WRONG: ") + f"expected {scene.expected or 'nothing'}, got {zone or 'nothing'}"
        image = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
        header = np.full((52, 480, 3), 30, np.uint8)  # a strip above the image, so nothing in the frame is covered
        cv2.putText(header, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(header, verdict, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 90) if ok else (80, 80, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([header, image]))
    columns = 3
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(path), np.vstack(rows))
    return path


# ------------------------------------------------------------------------------------------------ geometry
def layout_report(w=FRAME_W, h=FRAME_H, samples=40000, seed=0):
    """What the zone layout allows, from geometry alone (no images).

    Assumes a top-down camera and a bin about as wide as the car's placement box (the white box on the floor). The robot only
    ever drives along one straight line from its start (forward, backward, or after a 90 degree turn to either side), so the
    distance by which trash misses that line is fixed by where the trash is, whatever the timing.
    """
    zm = ZoneManager(w, h)
    cx1, cy1, cx2, cy2 = zm.car_box()
    rng = np.random.default_rng(seed)
    report = {"frame": [w, h], "car_box": [cx1, cy1, cx2, cy2], "zones": {}}
    frame_area = w * h
    covered = 0
    for name, z in zm.zones.items():
        area = (z["x2"] - z["x1"]) * (z["y2"] - z["y1"])
        covered += area
        ox = max(0, min(z["x2"], cx2) - max(z["x1"], cx1))
        oy = max(0, min(z["y2"], cy2) - max(z["y1"], cy1))
        xs = rng.integers(z["x1"], z["x2"] + 1, samples)
        ys = rng.integers(z["y1"], z["y2"] + 1, samples)
        on_car = (xs >= cx1) & (xs <= cx2) & (ys >= cy1) & (ys <= cy2)
        xs, ys = xs[~on_car], ys[~on_car]
        # distance from the straight line the robot drives along
        miss = np.abs(xs - zm.car_x) if name in ("FRONT", "BACK") else np.abs(ys - zm.car_y)
        # distance the robot has to drive to be level with the trash
        along = np.abs(ys - zm.car_y) if name in ("FRONT", "BACK") else np.abs(xs - zm.car_x)
        report["zones"][name] = {
            "rect": [z["x1"], z["y1"], z["x2"], z["y2"]],
            "area_px": area,
            "overlap_with_car_box_px": ox * oy,
            "overlap_with_car_box_pct": ox * oy / area,
            "median_miss_px": float(np.median(miss)),
            "p90_miss_px": float(np.percentile(miss, 90)),
            "within_car_half_width_pct": float(np.mean(miss <= zm.car_zone_size)),
            "drive_distance_min_px": float(along.min()),
            "drive_distance_median_px": float(np.median(along)),
            "drive_distance_max_px": float(along.max()),
        }
    report["frame_covered_by_zones_pct"] = covered / frame_area
    return report


def format_layout(report):
    lines = [f"Frame {report['frame'][0]}x{report['frame'][1]}, car placement box {tuple(report['car_box'])}, "
             f"zones cover {report['frame_covered_by_zones_pct']:.0%} of the frame", "",
             "| Zone | Zone area | Inside the car's own box | Median miss | 90th percentile miss | Within the box's half-width | Drive distance needed (min / median / max) |",
             "|---|---|---|---|---|---|---|"]
    for name, z in report["zones"].items():
        lines.append(f"| {name} | {z['area_px']:,} px² | {z['overlap_with_car_box_pct']:.0%} | {z['median_miss_px']:.0f} px | "
                     f"{z['p90_miss_px']:.0f} px | {z['within_car_half_width_pct']:.0%} | "
                     f"{z['drive_distance_min_px']:.0f} / {z['drive_distance_median_px']:.0f} / {z['drive_distance_max_px']:.0f} px |")
    return "\n".join(lines)


def save_json(results, path):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
