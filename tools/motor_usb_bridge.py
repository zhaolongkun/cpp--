#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bridge process for dual-axis USB motor controller.

Reads commands from stdin lines: "<x_speed> <y_speed>"
and sends them to two USB devices using the same frame format
as the legacy Python project.
"""

import argparse
import sys
import time
import traceback


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


class MotorController(object):
    def __init__(self, vid, pid, debug=False):
        self.debug = debug
        self.device = None
        self.crc16 = None
        self._init_deps()
        self._open(vid, pid)

    def _init_deps(self):
        try:
            import usb.core  # noqa: F401
            import crcmod.predefined  # noqa: F401
        except Exception as exc:
            raise RuntimeError("missing dependency pyusb/crcmod: %s" % exc)

    def _open(self, vid, pid):
        import usb.core
        import crcmod.predefined

        self.device = usb.core.find(idVendor=vid, idProduct=pid)
        if self.device is None:
            raise RuntimeError("usb device not found vid=%s pid=%s" % (vid, pid))
        try:
            self.device.set_configuration()
        except Exception as exc:
            # Some controllers are already configured by system driver. Keep going.
            if self.debug:
                eprint("[bridge] set_configuration warning:", repr(exc))
        self.crc16 = crcmod.predefined.Crc("modbus")
        if self.debug:
            eprint("[bridge] open usb %s:%s ok" % (hex(vid), hex(pid)))

    def _send_raw(self, payload):
        import usb.core

        if self.device is None:
            return
        self.crc16 = self.crc16.new()
        self.crc16.update(bytes(payload))
        crc = self.crc16.crcValue
        send_buf = list(payload)
        send_buf.append(crc & 0xFF)
        send_buf.append((crc >> 8) & 0xFF)
        self.device.write(0x01, send_buf, 200)
        try:
            self.device.read(0x81, 64, 100)
        except usb.core.USBTimeoutError:
            pass

    def go_speed(self, motor1=0):
        m = int(motor1)
        payload = [
            0x01,
            0x10,
            0x00,
            0x00,
            0x00,
            0x10,
            0x20,
            0x00,
            0x04,
            0x00,
            0x00,
            (m >> 8) & 0xFF,
            m & 0xFF,
            (m >> 24) & 0xFF,
            (m >> 16) & 0xFF,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
        ]
        self._send_raw(payload)
        if self.debug:
            eprint("[bridge] speed=%d" % m)

    def close(self):
        try:
            if self.device is not None:
                self.device.reset()
        except Exception:
            pass
        self.device = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--x_vid", type=int, required=True)
    p.add_argument("--x_pid", type=int, required=True)
    p.add_argument("--y_vid", type=int, required=True)
    p.add_argument("--y_pid", type=int, required=True)
    p.add_argument("--abs_limit", type=int, default=80)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    abs_limit = max(1, int(args.abs_limit))
    try:
        x = MotorController(args.x_vid, args.x_pid, debug=args.debug)
        y = MotorController(args.y_vid, args.y_pid, debug=args.debug)
    except Exception as exc:
        eprint("[bridge] init failed:", repr(exc))
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 1

    eprint("[bridge] ready")
    rc = 0
    last_send_time = 0.0
    MIN_SEND_INTERVAL = 0.033  # 最快 30Hz
    try:
        for line in sys.stdin:
            s = line.strip()
            if not s:
                continue
            if s.lower() in ("quit", "exit", "q"):
                break
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                xs = int(float(parts[0]))
                ys = int(float(parts[1]))
            except Exception:
                continue
            xs0, ys0 = xs, ys
            if xs > abs_limit:
                xs = abs_limit
            elif xs < -abs_limit:
                xs = -abs_limit
            if ys > abs_limit:
                ys = abs_limit
            elif ys < -abs_limit:
                ys = -abs_limit
            if args.debug and (xs != xs0 or ys != ys0):
                eprint("[bridge] clamp (%d,%d)->(%d,%d) lim=%d" % (xs0, ys0, xs, ys, abs_limit))
            try:
                now = time.monotonic()
                if now - last_send_time < MIN_SEND_INTERVAL:
                    continue
                x.go_speed(xs)
                y.go_speed(ys)
                last_send_time = time.monotonic()
                if args.debug:
                    eprint("[bridge] x=%d y=%d" % (xs, ys))
            except Exception as exc:
                rc = 2
                eprint("[bridge] send failed:", repr(exc))
                if args.debug:
                    traceback.print_exc(file=sys.stderr)
                time.sleep(0.02)
    finally:
        try:
            x.go_speed(0)
            y.go_speed(0)
        except Exception:
            pass
        x.close()
        y.close()
        eprint("[bridge] closed")

    return rc


if __name__ == "__main__":
    sys.exit(main())
