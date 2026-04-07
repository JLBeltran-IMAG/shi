#!/usr/bin/env python3

import time
import serial
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate trigger pulses via FTDI RTS")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("--fps", type=float, default=20.0, help="Trigger frequency in Hz")
    parser.add_argument("--npulses", type=int, default=100, help="Number of pulses to send")
    parser.add_argument("--pulse_ms", type=float, default=2.0, help="Pulse width in ms")
    args = parser.parse_args()

    period = 1.0 / args.fps
    pulse_s = args.pulse_ms / 1000.0

    print(f"Trigger test:")
    print(f"  Port      : {args.port}")
    print(f"  FPS       : {args.fps}")
    print(f"  Pulses    : {args.npulses}")
    print(f"  Pulse ms  : {args.pulse_ms}")
    print(f"  Period ms : {period*1000:.2f}")
    print("Press Ctrl+C to abort\n")

    ser = serial.Serial(args.port)
    ser.rts = False   # idle HIGH (because logic is inverted)

    try:
        for i in range(args.npulses):
            t0 = time.perf_counter()

            # --- trigger pulse ---
            ser.rts = True    # LOW
            time.sleep(pulse_s)
            ser.rts = False   # HIGH  -> rising edge

            print(f"Pulse {i+1}")

            # --- wait rest of period ---
            dt = time.perf_counter() - t0
            sleep_time = period - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nAborted by user")

    finally:
        ser.rts = False
        ser.close()
        print("Done")


if __name__ == "__main__":
    main()
