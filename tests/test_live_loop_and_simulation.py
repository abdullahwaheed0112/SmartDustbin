"""The live loop on a fake camera, the motion model, the animation, the evaluation and the command line."""
import json
from collections import deque

import numpy as np
import pytest

from smart_dustbin import evaluation, simulation
from smart_dustbin.__main__ import main as cli
from smart_dustbin.actions import ActionExecutor
from smart_dustbin.app import LiveState, run_live
from smart_dustbin.scenes import Blob, SyntheticCamera, render_scene
from smart_dustbin.simulation import Pose, VirtualClock, integrate_path, turn_rate_deg_s
from smart_dustbin.transport import RecordingTransport

RED = (0, 235, 215)


def scene_frames(x, y, empty=4, thrown=6, seed=0):
    rng = np.random.default_rng(seed)
    empty_frame = render_scene(rng)
    thrown_frame = render_scene(rng, [Blob(x, y, 18, RED)], background=empty_frame)  # same floor, plus the object
    return [empty_frame] * empty + [thrown_frame] * thrown


def run(frames, commands, max_frames=None, state=None):
    clock = VirtualClock()
    transport = RecordingTransport(clock.now)
    executor = ActionExecutor(transport, clock=clock.now, sleep=clock.sleep, log=lambda *_: None)
    state = run_live(SyntheticCamera(frames), transport, executor=executor, commands=deque(commands), state=state,
                     display=lambda frame: False, max_frames=max_frames if max_frames is not None else len(frames) - 1,
                     threaded=False, log=lambda *_: None)
    return ''.join(c for _, c in transport.events), state, executor


# ------------------------------------------------------------------------------------------------ live loop
class TestLiveLoop:
    @pytest.mark.parametrize("point,sent", [((320, 50), 'US'), ((60, 240), 'LSUS'), ((580, 240), 'RSUS'), ((240, 400), 'DS')])
    def test_armed_detection_triggers_the_right_movement(self, point, sent):
        commands, state, _ = run(scene_frames(*point), ['G'])
        assert commands == sent and state.detection_enabled is False  # one throw, one movement

    def test_nothing_moves_until_detection_is_armed(self):
        commands, state, _ = run(scene_frames(60, 240), [])
        assert commands == '' and state.detection_enabled is False

    def test_trash_in_a_gap_is_ignored(self):
        commands, state, _ = run(scene_frames(320, 240), ['G'])
        assert commands == '' and state.detection_enabled is True  # still armed, nothing in a zone

    def test_a_single_throw_causes_a_single_movement(self):
        commands, _, _ = run(scene_frames(60, 240, thrown=12), ['G'])
        assert commands == 'LSUS'

    def test_zone_test_keys_run_a_movement_without_any_detection(self):
        commands, _, _ = run(scene_frames(320, 240, thrown=0), ['3'])
        assert commands == 'LSUS'

    def test_manual_driving_keys(self):
        commands, _, _ = run(scene_frames(320, 240, thrown=0, empty=6), ['W'])
        assert commands == 'U'

    def test_quit_stops_the_loop(self):
        _, state, _ = run(scene_frames(320, 240), ['Q'], max_frames=50)
        assert state.running is False

    def test_the_display_can_stop_the_loop(self):
        clock = VirtualClock()
        transport = RecordingTransport(clock.now)
        state = run_live(SyntheticCamera(scene_frames(60, 240)), transport, display=lambda f: True, threaded=False, log=lambda *_: None)
        assert state.running is False

    def test_an_unreadable_camera_ends_cleanly(self):
        class Dead:
            def read(self):
                return False, None
        logs = []
        state = run_live(Dead(), RecordingTransport(lambda: 0), display=lambda f: False, log=logs.append)
        assert any("Cannot read" in line for line in logs) and isinstance(state, LiveState)


# ------------------------------------------------------------------------------------------------ motion model
class TestMotionModel:
    def test_forward_moves_up_the_image(self):
        path = integrate_path([(0, 'U'), (1, 'S')], Pose(320, 400, 0), speed=100, turn_rate=180)
        assert path[-1][1].x == pytest.approx(320) and path[-1][1].y == pytest.approx(300, abs=0.5)

    def test_backward_moves_down(self):
        path = integrate_path([(0, 'D'), (1, 'S')], Pose(320, 100, 0), speed=100, turn_rate=180)
        assert path[-1][1].y == pytest.approx(200, abs=0.5)

    def test_the_original_timing_is_a_quarter_turn(self):
        assert turn_rate_deg_s() * 0.5 == pytest.approx(90)
        path = integrate_path([(0, 'L'), (0.5, 'S')], Pose(0, 0, 0), speed=0, turn_rate=turn_rate_deg_s())
        assert path[-1][1].heading == pytest.approx(-90, abs=0.5)

    def test_turn_left_then_forward_goes_left(self):
        path = integrate_path([(0, 'L'), (0.5, 'S'), (0.7, 'U'), (1.7, 'S')], Pose(320, 400, 0), speed=100, turn_rate=180)
        end = path[-1][1]
        assert end.x == pytest.approx(220, abs=1.0) and end.y == pytest.approx(400, abs=1.0)

    def test_turn_right_then_forward_goes_right(self):
        path = integrate_path([(0, 'R'), (0.5, 'S'), (0.7, 'U'), (1.7, 'S')], Pose(320, 400, 0), speed=100, turn_rate=180)
        assert path[-1][1].x == pytest.approx(420, abs=1.0)

    def test_no_events_no_path(self):
        assert integrate_path([], Pose(0, 0, 0), 100, 180) == []


# ------------------------------------------------------------------------------------------------ the demo
class TestDemo:
    @pytest.mark.parametrize("zone,sent", [('FRONT', 'US'), ('BACK', 'DS'), ('LEFT', 'LSUS'), ('RIGHT', 'RSUS')])
    def test_the_real_program_handles_each_zone(self, zone, sent):
        result = simulation.run_demo(zone)
        assert result.detected_zone == zone
        assert ''.join(c for _, c in result.commands) == sent
        assert len(result.frames) == len(result.durations_ms) > 10

    def test_the_same_seed_gives_the_same_scene(self):
        a, b = simulation.run_demo('LEFT', seed=3), simulation.run_demo('LEFT', seed=3)
        assert a.object_position == b.object_position and a.offset_px == b.offset_px

    def test_a_different_seed_moves_the_object(self):
        assert simulation.run_demo('LEFT', seed=1).object_position != simulation.run_demo('LEFT', seed=2).object_position

    def test_the_gif_is_written_and_has_all_frames(self, tmp_path):
        from PIL import Image
        result = simulation.run_demo('FRONT')
        path = simulation.save_gif(result.frames, result.durations_ms, tmp_path / "front.gif")
        gif = Image.open(path)
        assert gif.n_frames > 5 and path.stat().st_size < 3_000_000

    def test_every_frame_says_it_is_a_simulation(self):
        """The animation must never pass for a recording of the real robot."""
        result = simulation.run_demo('FRONT')
        assert all(f.shape == result.frames[0].shape for f in result.frames)
        assert result.frames[0].shape[0] > 480  # includes the caption strip below the camera image


# ------------------------------------------------------------------------------------------------ evaluation
class TestEvaluation:
    def test_results_are_repeatable(self):
        a = evaluation.evaluate(n=6, seed=1)
        b = evaluation.evaluate(n=6, seed=1)
        assert a == b

    def test_structure_and_totals(self):
        results = evaluation.evaluate(n=6, seed=1)
        assert len(results["conditions"]) == len(evaluation.CONDITIONS) == 12
        for row in results["conditions"]:
            assert row["n"] == 6 and 0 <= row["original"]["accuracy"] <= 1 and 0 <= row["keepout"]["accuracy"] <= 1

    def test_known_behaviours(self):
        rows = {r["condition"]: r for r in evaluation.evaluate(n=25, seed=42)["conditions"]}
        assert rows["clean scene"]["original"]["accuracy"] >= 0.9
        assert rows["look-alikes only"]["original"]["accuracy"] == 1.0
        assert rows["small trash"]["original"]["accuracy"] == 0.0               # below the 500 px^2 minimum, by design
        assert rows["red mark on the car"]["original"]["accuracy"] == 0.0       # the car's own box is inside BACK
        assert rows["red mark on the car"]["keepout"]["accuracy"] == 1.0        # ...and the keep-out mask fixes it
        assert rows["clean scene"]["original"]["mean_centroid_error_px"] < 2

    def test_markdown_table(self):
        table = evaluation.format_markdown(evaluation.evaluate(n=3, seed=1))
        assert table.count("\n") == 13 and table.startswith("| Condition")

    def test_json_round_trip(self, tmp_path):
        results = evaluation.evaluate(n=3, seed=1)
        evaluation.save_json(results, tmp_path / "r.json")
        assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8")) == results

    def test_montage_is_written(self, tmp_path):
        path = evaluation.conditions_montage(tmp_path / "m.png")
        assert path.exists() and path.stat().st_size > 10_000


class TestLayoutGeometry:
    report = evaluation.layout_report()

    def test_zone_areas_and_coverage(self):
        assert self.report["zones"]["FRONT"]["area_px"] == 192 * 144
        assert self.report["frame_covered_by_zones_pct"] == pytest.approx(0.38, abs=0.005)

    def test_the_cars_own_box_fills_half_of_the_BACK_zone(self):
        back = self.report["zones"]["BACK"]
        assert back["overlap_with_car_box_px"] == 120 * 120
        assert back["overlap_with_car_box_pct"] == pytest.approx(14400 / 27648)
        assert all(self.report["zones"][z]["overlap_with_car_box_px"] == 0 for z in ("FRONT", "LEFT", "RIGHT"))

    def test_identical_one_second_moves_for_very_different_distances(self):
        zones = self.report["zones"]
        assert zones["FRONT"]["drive_distance_min_px"] == 264 and zones["FRONT"]["drive_distance_max_px"] == 408
        assert zones["BACK"]["drive_distance_max_px"] == 72

    def test_side_zones_are_never_on_the_line_the_robot_drives_along(self):
        for zone in ("LEFT", "RIGHT"):
            assert self.report["zones"][zone]["within_car_half_width_pct"] == 0.0
            assert self.report["zones"][zone]["median_miss_px"] > 150


# ------------------------------------------------------------------------------------------------ command line
class TestCommandLine:
    def test_layout(self, capsys):
        assert cli(["layout"]) == 0
        assert "BACK" in capsys.readouterr().out

    def test_simulate_prints_the_commands(self, capsys):
        assert cli(["simulate", "--zone", "LEFT", "--no-gif"]) == 0
        out = capsys.readouterr().out
        assert "detected" in out and "L" in out and "bin ends" in out

    def test_simulate_writes_a_gif(self, tmp_path, capsys):
        assert cli(["simulate", "--zone", "FRONT", "--out", str(tmp_path)]) == 0
        assert (tmp_path / "demo-front.gif").exists()

    def test_evaluate_with_json(self, tmp_path, capsys):
        assert cli(["evaluate", "--n", "3", "--json", str(tmp_path / "r.json")]) == 0
        out = capsys.readouterr().out
        assert "synthetic" in out and (tmp_path / "r.json").exists()

    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            cli([])
