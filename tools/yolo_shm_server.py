"""
YOLO GPU inference server — writes detections to Windows shared memory.
Shared memory layout (little-endian):
  [int32 seq] [int32 count] [count × (float32 x1,y1,x2,y2,conf)]
Max 64 detections per frame.
"""
import argparse
import ctypes
import struct
import sys
import time

import numpy as np

MAX_DETS = 64
SHM_NAME = "yolo_det_shm"
FRAME_W = 640
FRAME_H = 480
FRAME_BYTES = FRAME_W * FRAME_H * 3
# seq(4) + count(4) + MAX_DETS*(5*4) + frame(FRAME_BYTES)
SHM_SIZE = 4 + 4 + MAX_DETS * 5 * 4 + FRAME_BYTES


def open_shm():
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileMappingW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_wchar_p,
    ]
    kernel32.CreateFileMappingW.restype = ctypes.c_void_p
    kernel32.MapViewOfFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_size_t,
    ]
    kernel32.MapViewOfFile.restype = ctypes.c_void_p
    handle = kernel32.CreateFileMappingW(
        ctypes.c_void_p(-1).value, None, 0x04, 0, SHM_SIZE, SHM_NAME
    )
    if not handle:
        raise RuntimeError("CreateFileMapping failed")
    ptr = kernel32.MapViewOfFile(handle, 0xF001F, 0, 0, SHM_SIZE)
    if not ptr:
        raise RuntimeError("MapViewOfFile failed")
    return kernel32, handle, ptr


def write_shm(ptr, seq, dets, frame):
    """dets: list of (x1,y1,x2,y2,conf), frame: BGR numpy array"""
    count = min(len(dets), MAX_DETS)
    data = struct.pack("<ii", seq, count)
    for i in range(count):
        data += struct.pack("<fffff", *dets[i])
    # pad det section
    det_size = 4 + 4 + MAX_DETS * 5 * 4
    data = data.ljust(det_size, b'\x00')
    # append frame
    import cv2
    resized = cv2.resize(frame, (FRAME_W, FRAME_H))
    data += resized.tobytes()
    ctypes.memmove(ptr, data, len(data))


def backend_candidates(cv2, backend_name):
    name = (backend_name or "dshow").strip().lower()
    if name == "msmf":
        return [("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY), ("dshow", cv2.CAP_DSHOW)]
    if name == "any":
        return [("any", cv2.CAP_ANY), ("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)]
    return [("dshow", cv2.CAP_DSHOW), ("any", cv2.CAP_ANY), ("msmf", cv2.CAP_MSMF)]


def warmup_capture(cap, attempts=20, delay_s=0.05):
    last_frame = None
    for _ in range(attempts):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size:
            return True, frame
        last_frame = frame
        time.sleep(delay_s)
    return False, last_frame


def open_camera(cv2, camera_index, width, height, backend_name):
    errors = []
    for backend_label, backend_api in backend_candidates(cv2, backend_name):
        if backend_api == cv2.CAP_ANY:
            cap = cv2.VideoCapture(camera_index)
        else:
            cap = cv2.VideoCapture(camera_index, backend_api)
        if not cap.isOpened():
            errors.append(f"{backend_label}: open failed")
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = warmup_capture(cap)
        if ok:
            return cap, backend_label, frame

        errors.append(f"{backend_label}: opened but no frame")
        cap.release()

    raise RuntimeError(
        f"failed to open camera index={camera_index} via backends "
        f"{', '.join(label for label, _ in backend_candidates(cv2, backend_name))}; "
        f"details: {'; '.join(errors)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="D:/kun-data/kun-code-data/run/yolo12n/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.65)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--backend", default="dshow")
    args = parser.parse_args()

    from ultralytics import YOLO
    import cv2

    kernel32, handle, shm_ptr = open_shm()
    model = YOLO(args.model)
    try:
        cap, backend_used, warmup_frame = open_camera(cv2, args.camera, args.width, args.height, args.backend)
    except RuntimeError as exc:
        print(f"[yolo_shm_server] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1

    print(
        f"[yolo_shm_server] started, model={args.model}, camera={args.camera}, backend={backend_used}",
        flush=True,
    )

    seq = 0
    frame = warmup_frame
    read_fail_streak = 0
    write_shm(shm_ptr, seq, [], frame)
    seq += 1
    while True:
        if frame is None:
            ret, frame = cap.read()
        else:
            ret = True
        if not ret or frame is None or frame.size == 0:
            read_fail_streak += 1
            if read_fail_streak == 1 or read_fail_streak % 100 == 0:
                print(
                    f"[yolo_shm_server] warning: camera read failed (streak={read_fail_streak})",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(0.01)
            frame = None
            continue
        if read_fail_streak:
            print(
                f"[yolo_shm_server] camera recovered after {read_fail_streak} failed reads",
                flush=True,
            )
            read_fail_streak = 0
        results = model.predict(frame, conf=args.conf, device=0, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                dets.append((x1, y1, x2, y2, conf))
        write_shm(shm_ptr, seq, dets, frame)
        seq += 1
        frame = None


if __name__ == "__main__":
    sys.exit(main())
