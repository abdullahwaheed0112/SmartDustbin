"""Deciding and moving: which movement each zone triggers, and the executor that carries it out.

The robot is driven open loop: each movement is a command character held for a fixed time, then a stop. There is no
sensor on the robot, so how far it actually goes depends on its battery, floor and motors. The durations live in config.py.
"""
import time
from dataclasses import dataclass

from . import config

# What the ESP32 firmware is told (one character each)
COMMAND_MEANING = {
    'U': 'forward',
    'D': 'backward',
    'L': 'turn left',
    'R': 'turn right',
    'S': 'stop',
}


@dataclass(frozen=True)
class Step:
    """One movement: a command character, how long to hold it, and a description for the log."""
    command: str
    duration: float
    description: str


def plan_for(zone_name):
    """The sequence of steps for a zone, or None for an unknown zone. Durations are read from config at call time."""
    plans = {
        'FRONT': [Step('U', config.FORWARD_TIME, "Move forward to FRONT zone")],
        'BACK': [Step('D', config.BACKWARD_TIME, "Move backward to BACK zone")],
        'LEFT': [Step('L', config.TURN_90_TIME, "Turn 90° LEFT"),
                 Step('U', config.SIDE_FORWARD_TIME, "Move forward to LEFT zone")],
        'RIGHT': [Step('R', config.TURN_90_TIME, "Turn 90° RIGHT"),
                  Step('U', config.SIDE_FORWARD_TIME, "Move forward to RIGHT zone")],
    }
    return plans.get(zone_name)


class ActionExecutor:
    """Runs the movement plan for a zone, with a cooldown so one throw does not trigger two movements.

    `transport` receives the command characters. `clock`, `sleep` and `log` default to the real ones; the simulation and
    the tests pass a virtual clock so a one-second movement costs no real time.
    """

    def __init__(self, transport, clock=time.time, sleep=time.sleep, log=print, cooldown=None):
        self.transport = transport
        self.clock = clock
        self.sleep = sleep
        self.log = log
        self.cooldown = config.ACTION_COOLDOWN if cooldown is None else cooldown
        self.executing = False
        self.last_action_time = 0

    def can_execute(self):
        """True if nothing is running and enough time has passed since the last movement."""
        if self.executing:
            return False
        if self.clock() - self.last_action_time < self.cooldown:
            return False
        return True

    def send_and_wait(self, cmd, duration, desc):
        """Send a command, hold it for `duration` seconds, then stop."""
        self.log(f"   {cmd} for {duration:.1f}s ({desc})")
        self.transport.send(cmd)
        self.sleep(duration)
        self.transport.send('S')
        self.sleep(0.2)  # pause

    def execute_zone_action(self, zone_name):
        """Carry out the movement for `zone_name` (nothing happens if one is already running or cooling down)."""
        if not self.can_execute():
            return
        steps = plan_for(zone_name)
        if steps is None:
            self.log(f"No movement is defined for zone {zone_name!r}")
            return

        self.executing = True
        self.log(f"\n{'=' * 50}")
        self.log(f"EXECUTING ACTION: {zone_name}")
        self.log(f"{'=' * 50}")

        try:
            for step in steps:
                self.send_and_wait(step.command, step.duration, step.description)

            self.log(f"ACTION COMPLETE: {zone_name}")
            self.log(" Return car to starting position")
            self.log(f"{'=' * 50}\n")

        except Exception as e:
            self.log(f"Error during execution: {e}")
        finally:
            self.executing = False
            self.last_action_time = self.clock()
