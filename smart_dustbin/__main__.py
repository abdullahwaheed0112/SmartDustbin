"""Command line entry point.

    python -m smart_dustbin simulate --zone LEFT     run the program on a synthetic camera and animate the result
    python -m smart_dustbin evaluate                 measure detection and zone decisions on hundreds of synthetic scenes
    python -m smart_dustbin layout                   analyse the zone layout from geometry alone
    python -m smart_dustbin live                     run with a real camera (and the robot, if ESP_MAC is set)

The first three need no camera, no robot and no Bluetooth.
"""
import argparse
import sys
from pathlib import Path

ZONES = ["FRONT", "BACK", "LEFT", "RIGHT"]


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m smart_dustbin", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run the real program on a synthetic camera and animate what would happen")
    sim.add_argument("--zone", choices=ZONES + ["all"], default="all", help="zone to throw the red object into")
    sim.add_argument("--seed", type=int, default=7, help="random seed (the same seed repeats the same scene)")
    sim.add_argument("--speed", type=float, default=None, help="assumed forward speed in image pixels per second (default 200)")
    sim.add_argument("--out", default="docs/demo", help="folder for the animated GIFs (default docs/demo); use --no-gif to skip")
    sim.add_argument("--no-gif", action="store_true", help="only print the commands, write no files")

    ev = sub.add_parser("evaluate", help="measure detection and zone decisions on synthetic scenes")
    ev.add_argument("--n", type=int, default=100, help="scenes per condition (default 100)")
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--json", help="also write the full results to this JSON file")
    ev.add_argument("--montage", help="also write an example image per condition to this PNG file")

    sub.add_parser("layout", help="analyse the zone layout from geometry alone")
    sub.add_parser("live", help="run with a camera (and the robot, if ESP_MAC is set)")
    return parser


def _simulate(args):
    from . import simulation

    speed = args.speed if args.speed is not None else simulation.DEFAULT_SPEED_PX_S
    zones = ZONES if args.zone == "all" else [args.zone]
    out = Path(args.out)
    if not args.no_gif:
        out.mkdir(parents=True, exist_ok=True)
    for zone in zones:
        result = simulation.run_demo(zone, seed=args.seed, speed=speed)
        commands = " ".join(f"{t:4.1f}s {c}" for t, c in result.commands)
        print(f"{zone:5s} object at {result.object_position} -> detected {result.detected} in zone {result.detected_zone}")
        print(f"      commands sent: {commands}")
        print(f"      bin ends {result.offset_px:.0f} px from the object (assumed speed {speed:.0f} px/s; open loop, no steering)")
        if not args.no_gif:
            path = simulation.save_gif(result.frames, result.durations_ms, out / f"demo-{zone.lower()}.gif")
            print(f"      animation: {path}")
    return 0


def _evaluate(args):
    from . import evaluation

    results = evaluation.evaluate(n=args.n, seed=args.seed)
    print(evaluation.format_markdown(results))
    print(f"\nMean right-decision rate over {len(results['conditions'])} conditions: "
          f"{results['mean_accuracy']['original']:.1%} (original) / {results['mean_accuracy']['keepout']:.1%} (keep-out mask)")
    print("All scenes are synthetic: this is how the code behaves on the situations built here, not real-world accuracy.")
    if args.json:
        evaluation.save_json(results, args.json)
        print(f"results written to {args.json}")
    if args.montage:
        evaluation.conditions_montage(args.montage)
        print(f"example images written to {args.montage}")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        return _simulate(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "layout":
        from . import evaluation
        print(evaluation.format_layout(evaluation.layout_report()))
        return 0
    if args.command == "live":
        from .app import main as run_live
        run_live()
    return 0


if __name__ == "__main__":
    sys.exit(main())
