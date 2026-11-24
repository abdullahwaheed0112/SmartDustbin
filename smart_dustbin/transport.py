"""Getting commands to the robot (or pretending to).

- `BluetoothTransport` and `ESP32AutoConnect`: the real link to the ESP32 over Bluetooth RFCOMM. Linux only, needs
  `sudo`, and was written by the original author for the real robot. It has not been run in this repository.
- `DryRunTransport`: prints the commands instead of sending them (what the original did without a serial port).
- `RecordingTransport`: remembers every command with the time it was sent, which is what the simulation animates.
"""
import os
import subprocess
import time

from . import config

try:  # optional: only needed for the real robot
    import serial
except Exception:
    serial = None


class ESP32AutoConnect:
    """Bluetooth connection handler (the original author's code, with the address and port taken from config)."""

    def __init__(self, mac_address, port):
        self.mac = mac_address
        self.port = port
        self.ser = None

    def setup_connection(self):
        """Setup rfcomm binding"""
        try:
            print("Releasing existing bindings...")
            subprocess.run(['sudo', 'rfcomm', 'release', self.port], stderr=subprocess.DEVNULL)
            time.sleep(1)

            print(f"Binding {self.mac} to {self.port}...")
            result = subprocess.run(['sudo', 'rfcomm', 'bind', self.port, self.mac, '1'],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Bind failed: {result.stderr}")
                return False
            print(" Bound successfully")
            time.sleep(1.2)

            subprocess.run(['sudo', 'chmod', '666', self.port], stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            print(f"Setup error: {e}")
            return False

    def connect(self):
        """Open serial connection"""
        if not self.setup_connection():
            return False
        if not os.path.exists(self.port):
            print(" Port not created")
            return False
        if serial is None:
            print("pyserial not installed")
            return False
        try:
            print(f"Opening serial on {self.port}...")
            self.ser = serial.Serial(port=self.port, baudrate=config.BAUD_RATE, timeout=2)
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
            except Exception:
                pass
        try:
            subprocess.run(['sudo', 'rfcomm', 'release', self.port], stderr=subprocess.DEVNULL)
        except Exception:
            pass
        print(" Cleanup done")


class BluetoothTransport:
    """Sends commands over an open ESP32AutoConnect."""

    def __init__(self, car):
        self.car = car

    def send(self, cmd):
        self.car.send_command(cmd)


class DryRunTransport:
    """Prints each command instead of sending it. Used when there is no robot."""

    def __init__(self, log=print):
        self.log = log

    def send(self, cmd):
        self.log(f"  [DRY] {cmd}" + (" (stop)" if cmd == 'S' else ""))


class RecordingTransport:
    """Remembers every command as (time, command). `clock` is the clock the executor uses, so times line up."""

    def __init__(self, clock):
        self.clock = clock
        self.events = []

    def send(self, cmd):
        self.events.append((self.clock(), cmd))
