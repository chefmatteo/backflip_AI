"""Kinematic playback of the live baked_motion.csv through mma1 (no policy).

Usage (mma1 already running):
  cd src && python ../helpers/demo_ref.py

Keys (terminal): q quit, space pause, . / , step while paused.
"""
from __future__ import annotations

import select
import socket
import struct
import sys
import termios
import tty
from time import perf_counter, sleep

import numpy as np

FPS = 30.0
DT = 1.0 / FPS


class UDP:
    def __init__(self):
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = ("127.0.0.1", 5005)

    def reset(self, phase: float):
        # same -69 reset protocol as train/test
        self.send_sock.sendto(struct.pack("3f", -69.0, 0.0, float(phase)), self.send_addr)


def main():
    path = "baked_motion.csv"
    rows = np.loadtxt(path, delimiter=",")
    n = len(rows)
    print(f"demo: {path}  {n} frames ({n / FPS:.2f}s)  kinematic loop")
    print("  q=quit  space=pause  ,/.=step")
    udp = UDP()

    fd = sys.stdin.fileno()
    old = None
    try:
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except termios.error:
        old = None

    phase = 0.0
    paused = False
    next_t = perf_counter()
    try:
        while True:
            if old is not None and select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch in ("q", "Q"):
                    break
                if ch == " ":
                    paused = not paused
                    print("paused" if paused else "play", flush=True)
                    next_t = perf_counter()
                if paused and ch == ".":
                    phase = min(1.0, phase + 1.0 / max(n - 1, 1))
                    udp.reset(phase)
                if paused and ch == ",":
                    phase = max(0.0, phase - 1.0 / max(n - 1, 1))
                    udp.reset(phase)

            if paused:
                sleep(0.02)
                continue

            udp.reset(phase)
            phase += 1.0 / max(n - 1, 1)
            if phase >= 1.0:
                phase = 0.0
                print("loop", flush=True)

            next_t += DT
            delay = next_t - perf_counter()
            if delay > 0:
                sleep(delay)
            else:
                next_t = perf_counter()
    finally:
        if old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
