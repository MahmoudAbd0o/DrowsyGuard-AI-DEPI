"""Bench test for the Arduino / car link - no camera needed.

Usage:
    python tools/test_arduino.py            # uses config.SERIAL_PORT
    python tools/test_arduino.py COM7       # explicit port

It sends 0 -> 1 -> 3 -> 2 -> N -> 0 every 2 seconds so the LEDs and the
buzzer can be checked before running the full dashboard. While the level is
"2" the car must brake, so keep it on blocks or lift the wheels.
"""
from __future__ import annotations

import sys
import time

import config

try:
    import serial
    from serial.tools import list_ports
except ImportError:                                  # pragma: no cover
    print("pyserial is missing:  pip install pyserial")
    raise SystemExit(1)

SEQUENCE = [
    ("0", "safe - green LED"),
    ("1", "low risk - white LED + soft beep"),
    ("3", "distracted - faster beep"),
    ("2", "DROWSY - red LED + loud buzzer + BRAKE"),
    ("N", "no face - white LED, silent"),
    ("0", "safe again"),
]

STEP_SEC = 2.0


def main() -> int:
    port = sys.argv[1] if len(sys.argv) > 1 else config.SERIAL_PORT

    print("Ports found:", [p.device for p in list_ports.comports()] or "none")
    print(f"Opening {port} at {config.SERIAL_BAUD} ...")
    try:
        link = serial.Serial(port, config.SERIAL_BAUD, timeout=1)
    except Exception as exc:
        print(f"Cannot open {port}: {exc}")
        print("Close the Arduino IDE Serial Monitor and check the COM number.")
        return 1

    time.sleep(2)                                    # the board resets on open
    try:
        for level, label in SEQUENCE:
            link.write(level.encode())
            link.write(level.encode())               # twice: first byte is lost
            print(f"  -> {level}  {label}")
            time.sleep(STEP_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            link.write(b"N")
            link.close()
        except Exception:
            pass

    print("Done. Nothing alarming now (level N).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
