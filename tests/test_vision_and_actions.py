"""Zones, the detector, the movement plans and the executor. No camera or robot needed."""
import cv2
import numpy as np
import pytest

from smart_dustbin import config
from smart_dustbin.actions import ActionExecutor, plan_for
from smart_dustbin.scenes import render_scene
from smart_dustbin.simulation import VirtualClock
from smart_dustbin.transport import DryRunTransport, RecordingTransport
from smart_dustbin.vision import RedTrashDetector, ZoneManager, locate_zone

W, H = 640, 480


def blank():
    return np.full((H, W, 3), 140, np.uint8)


def with_disc(x, y, radius=20, bgr=(0, 0, 220)):
    frame = blank()
    cv2.circle(frame, (x, y), radius, bgr, -1)
    return frame


# ------------------------------------------------------------------------------------------------ zones
class TestZones:
    def test_geometry_at_640_by_480(self):
        zm = ZoneManager(W, H)
        assert {n: (z['x1'], z['y1'], z['x2'], z['y2']) for n, z in zm.zones.items()} == {
            'FRONT': (224, 0, 416, 144), 'BACK': (224, 336, 416, 480),
            'LEFT': (0, 144, 160, 336), 'RIGHT': (480, 144, 640, 336)}
        assert zm.car_box() == (260, 348, 380, 468)

    @pytest.mark.parametrize("point,zone", [((320, 50), 'FRONT'), ((320, 440), 'BACK'), ((60, 240), 'LEFT'), ((580, 240), 'RIGHT')])
    def test_points_in_each_zone(self, point, zone):
        assert ZoneManager(W, H).get_zone_for_position(*point) == zone

    @pytest.mark.parametrize("point", [(320, 240), (10, 10), (630, 470), (200, 100), (150, 400)])
    def test_gaps_belong_to_no_zone(self, point):
        assert ZoneManager(W, H).get_zone_for_position(*point) is None

    def test_zones_scale_with_the_frame(self):
        zm = ZoneManager(1280, 720)
        assert zm.zones['LEFT']['x2'] == 320 and zm.zones['FRONT']['y2'] == 216

    def test_draw_zones_marks_the_frame(self):
        frame = blank()
        before = frame.copy()
        ZoneManager(W, H).draw_zones(frame)
        assert not np.array_equal(frame, before)


# ------------------------------------------------------------------------------------------------ detector
class TestDetector:
    def test_finds_a_red_disc_and_its_centre(self):
        assert RedTrashDetector().detect_red(with_disc(110, 240)) == (110, 240)

    def test_both_sides_of_the_hue_wrap_are_red(self):
        low = (30, 30, 220)    # BGR with hue near 0
        high = (60, 0, 220)    # BGR with hue near 175
        for color in (low, high):
            assert RedTrashDetector().detect_red(with_disc(300, 200, bgr=color)) is not None

    @pytest.mark.parametrize("bgr", [(0, 140, 255), (0, 215, 255), (255, 0, 0), (0, 200, 0), (130, 130, 130)])
    def test_other_colours_are_ignored(self, bgr):  # orange, yellow, blue, green, grey
        assert RedTrashDetector().detect_red(with_disc(300, 200, bgr=bgr)) is None

    def test_empty_scene(self):
        assert RedTrashDetector().detect_red(blank()) is None

    def test_objects_below_the_minimum_area_are_ignored(self):
        assert RedTrashDetector().detect_red(with_disc(300, 200, radius=9)) is None      # about 254 px^2
        assert RedTrashDetector().detect_red(with_disc(300, 200, radius=14)) is not None  # about 615 px^2

    def test_the_largest_object_wins(self):
        frame = with_disc(100, 100, radius=15)
        cv2.circle(frame, (500, 300), 28, (0, 0, 220), -1)
        assert RedTrashDetector().detect_red(frame) == (500, 300)

    def test_keep_out_rectangle_hides_the_cars_own_box(self):
        zm = ZoneManager(W, H)
        frame = with_disc(zm.car_x, zm.car_y, radius=18)
        assert RedTrashDetector().detect_red(frame) is not None
        assert RedTrashDetector(ignore_rect=zm.car_box()).detect_red(frame) is None

    def test_keep_out_leaves_the_rest_of_the_frame_alone(self):
        zm = ZoneManager(W, H)
        assert RedTrashDetector(ignore_rect=zm.car_box()).detect_red(with_disc(110, 240)) == (110, 240)


class TestLocateZone:
    @pytest.mark.parametrize("point,zone", [((320, 50), 'FRONT'), ((60, 240), 'LEFT'), ((580, 240), 'RIGHT')])
    def test_position_and_zone(self, point, zone):
        position, found = locate_zone(with_disc(*point), RedTrashDetector(), ZoneManager(W, H))
        assert position == point and found == zone

    def test_trash_in_a_gap_has_a_position_but_no_zone(self):
        position, zone = locate_zone(with_disc(320, 240), RedTrashDetector(), ZoneManager(W, H))
        assert position == (320, 240) and zone is None

    def test_nothing_red(self):
        assert locate_zone(blank(), RedTrashDetector(), ZoneManager(W, H)) == (None, None)

    def test_a_red_mark_on_the_car_triggers_BACK(self):
        """A known weakness of the original design, pinned down: the car's own box lies inside the BACK zone."""
        frame = render_scene(np.random.default_rng(0), [], car_red_patch=30)
        assert locate_zone(frame, RedTrashDetector(), ZoneManager(W, H))[1] == 'BACK'
        assert locate_zone(frame, RedTrashDetector(ignore_rect=ZoneManager(W, H).car_box()), ZoneManager(W, H))[1] is None


# ------------------------------------------------------------------------------------------------ movement plans
class TestPlans:
    @pytest.mark.parametrize("zone,commands", [('FRONT', 'U'), ('BACK', 'D'), ('LEFT', 'LU'), ('RIGHT', 'RU')])
    def test_each_zone_has_its_movement(self, zone, commands):
        assert ''.join(s.command for s in plan_for(zone)) == commands

    def test_durations_come_from_the_configuration(self):
        assert [s.duration for s in plan_for('LEFT')] == [config.TURN_90_TIME, config.SIDE_FORWARD_TIME]

    def test_unknown_zone(self):
        assert plan_for('MIDDLE') is None


def make_executor(**kwargs):
    clock = VirtualClock()
    transport = RecordingTransport(clock.now)
    logs = []
    executor = ActionExecutor(transport, clock=clock.now, sleep=clock.sleep, log=logs.append, **kwargs)
    return executor, transport, clock, logs


class TestExecutor:
    @pytest.mark.parametrize("zone,sent", [('FRONT', 'US'), ('BACK', 'DS'), ('LEFT', 'LSUS'), ('RIGHT', 'RSUS')])
    def test_command_sequence_with_a_stop_after_every_movement(self, zone, sent):
        executor, transport, _, _ = make_executor()
        executor.execute_zone_action(zone)
        assert ''.join(c for _, c in transport.events) == sent

    def test_timing_is_the_configured_duration_plus_the_pause(self):
        executor, transport, _, _ = make_executor()
        executor.execute_zone_action('LEFT')
        t = [round(t - transport.events[0][0], 2) for t, _ in transport.events]
        assert t == [0.0, config.TURN_90_TIME, config.TURN_90_TIME + 0.2, config.TURN_90_TIME + 0.2 + config.SIDE_FORWARD_TIME]

    def test_nothing_runs_during_the_cooldown(self):
        executor, transport, clock, _ = make_executor()
        executor.execute_zone_action('FRONT')
        sent = len(transport.events)
        executor.execute_zone_action('FRONT')
        assert len(transport.events) == sent and not executor.can_execute()
        clock.sleep(config.ACTION_COOLDOWN)
        assert executor.can_execute()

    def test_it_is_ready_again_after_the_cooldown(self):
        executor, transport, clock, _ = make_executor()
        executor.execute_zone_action('FRONT')
        clock.sleep(config.ACTION_COOLDOWN + 1)
        executor.execute_zone_action('BACK')
        assert ''.join(c for _, c in transport.events) == 'USDS'

    def test_unknown_zone_does_nothing_and_does_not_start_a_cooldown(self):
        executor, transport, _, logs = make_executor()
        executor.execute_zone_action('MIDDLE')
        assert transport.events == [] and executor.can_execute() and any('MIDDLE' in line for line in logs)

    def test_a_failing_transport_is_logged_and_the_executor_recovers(self):
        class Broken:
            def send(self, cmd):
                raise OSError("link lost")
        clock = VirtualClock()
        logs = []
        executor = ActionExecutor(Broken(), clock=clock.now, sleep=clock.sleep, log=logs.append)
        executor.execute_zone_action('FRONT')
        assert any('link lost' in line for line in logs) and executor.executing is False


class TestTransports:
    def test_dry_run_prints_instead_of_sending(self):
        lines = []
        transport = DryRunTransport(log=lines.append)
        transport.send('U')
        transport.send('S')
        assert lines == ["  [DRY] U", "  [DRY] S (stop)"]

    def test_recording_transport_keeps_the_time_of_each_command(self):
        clock = VirtualClock(10.0)
        transport = RecordingTransport(clock.now)
        transport.send('U')
        clock.sleep(1.5)
        transport.send('S')
        assert transport.events == [(10.0, 'U'), (11.5, 'S')]


def test_the_bluetooth_address_is_not_hard_coded(monkeypatch):
    monkeypatch.delenv("ESP_MAC", raising=False)
    import importlib
    reloaded = importlib.reload(config)
    assert reloaded.ESP_MAC == ""
    monkeypatch.setenv("ESP_MAC", "AA:BB:CC:DD:EE:FF")
    assert importlib.reload(config).ESP_MAC == "AA:BB:CC:DD:EE:FF"
    monkeypatch.delenv("ESP_MAC")
    importlib.reload(config)
