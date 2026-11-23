"""Command line entry point: python -m smart_dustbin <live|simulate|evaluate|layout>."""
import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m smart_dustbin", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("live", help="run with a camera (and the robot, if ESP_MAC is set)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "live":
        from .app import main as run_live
        run_live()
    return 0


if __name__ == "__main__":
    sys.exit(main())
