"""
Pipeline latency benchmark: Camera → YOLO → Kalman → LPF → Features → DSCGNet(GPU) → Motor
200 warmup + 500 measured frames. Prints mean/std/min/max per stage.
"""
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2

# ── Minimal DSCGNet matching checkpoint architecture ──────────────────────────
class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=k - 1)
        self._trim = k - 1
    def forward(self, x):
        return self.conv(x)[..., :x.shape[-1]] if self._trim else self.conv(x)

class _MotionEncoder(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.conv1 = _CausalConv1d(3, h, 3)
        self.conv2 = _CausalConv1d(h, h, 3)
        self.gru   = nn.GRU(h, h, batch_first=True)
        self.attention = nn.MultiheadAttention(h, 4, batch_first=True)
        # wrap to match checkpoint key structure
        self.attention = type('A', (), {
            'attn': nn.MultiheadAttention(h, 4, batch_first=True),
            'norm': nn.LayerNorm(h),
        })()
        # rebuild as module so state_dict works
        self.attention = _Attn(h)
    def forward(self, x):  # x: (B, T, 3)
        z = torch.relu(self.conv1(x.transpose(1,2))).transpose(1,2)
        z = torch.relu(self.conv2(z.transpose(1,2))).transpose(1,2)
        z, _ = self.gru(z)
        z2, _ = self.attention.attn(z, z, z)
        z = self.attention.norm(z + z2)
        return z[:, -1, :]  # (B, h)

class _Attn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(h, 4, batch_first=True)
        self.norm = nn.LayerNorm(h)

class _QEncoder(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.proj = nn.Linear(4, h // 2)
        self.gru  = nn.GRU(h // 2, h // 2, batch_first=True)
    def forward(self, q):  # q: (B, T, 4)
        z = torch.relu(self.proj(q))
        z, _ = self.gru(z)
        return z[:, -1, :]  # (B, h//2)

class DSCGNetCompat(nn.Module):
    def __init__(self):
        super().__init__()
        h = 32
        self.x_encoder = _MotionEncoder(h)
        self.y_encoder = _MotionEncoder(h)
        self.q_encoder = _QEncoder(h)
        self.fusion    = nn.Sequential(nn.Linear(h*2 + h//2, 64), nn.ReLU())
        self.delta_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.gate_head  = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.r_max = nn.Parameter(torch.tensor([12.0, 12.0]))
    def forward(self, x_s, y_s, q):
        ex = self.x_encoder(x_s)
        ey = self.y_encoder(y_s)
        eq = self.q_encoder(q)
        f  = self.fusion(torch.cat([ex, ey, eq], dim=-1))
        delta = self.delta_head(f)
        gate  = torch.sigmoid(self.gate_head(f))
        e_f   = torch.cat([x_s[:, -1, 0:1], y_s[:, -1, 0:1]], dim=-1)
        return e_f + gate * (self.r_max * torch.tanh(delta))

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = Path(__file__).resolve().parents[1] / "data" / "train" / "dscgnet_legacy_control_best.pt"
SEQ_LEN    = 16
WARMUP     = 200
MEASURE    = 500
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ────────────────────────────────────────────────────────────────
ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
model = DSCGNetCompat().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
USE_CAM = cap.isOpened()
if not USE_CAM:
    cap.release()
    print("[warn] No camera found — using random frames.")

def grab_frame():
    if USE_CAM:
        ret, frame = cap.read()
        if ret:
            return frame
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# ── Fake YOLO (simulate detection on CPU) ─────────────────────────────────────
def fake_yolo(frame):
    # Simulate bounding-box extraction (resize + argmax as proxy work)
    small = cv2.resize(frame, (320, 320))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    idx   = np.unravel_index(np.argmax(gray), gray.shape)
    cx, cy = float(idx[1]) / 320.0, float(idx[0]) / 320.0
    conf  = float(gray[idx]) / 255.0
    area  = 0.05
    return cx, cy, conf, area  # normalised centre + conf + area

# ── Kalman state (2-D constant-velocity) ──────────────────────────────────────
_kx = np.zeros(4, dtype=np.float64)   # [x, vx, y, vy]
_kP = np.eye(4, dtype=np.float64) * 10.0
_kF = np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]], dtype=np.float64)
_kH = np.array([[1,0,0,0],[0,0,1,0]], dtype=np.float64)
_kQ = np.eye(4, dtype=np.float64) * 1e-3
_kR = np.eye(2, dtype=np.float64) * 1e-2

def kalman_update(cx, cy):
    global _kx, _kP
    _kx = _kF @ _kx
    _kP = _kF @ _kP @ _kF.T + _kQ
    z   = np.array([cx, cy])
    S   = _kH @ _kP @ _kH.T + _kR
    K   = _kP @ _kH.T @ np.linalg.inv(S)
    _kx = _kx + K @ (z - _kH @ _kx)
    _kP = (np.eye(4) - K @ _kH) @ _kP
    return float(_kx[0]), float(_kx[2])  # filtered x, y

# ── LPF (exponential) ─────────────────────────────────────────────────────────
_lpf = np.zeros(2, dtype=np.float64)
ALPHA = 0.3

def lpf(x, y):
    global _lpf
    _lpf = ALPHA * np.array([x, y]) + (1 - ALPHA) * _lpf
    return float(_lpf[0]), float(_lpf[1])

# ── Sliding window for DSCGNet features ───────────────────────────────────────
_hist = np.zeros((SEQ_LEN, 10), dtype=np.float32)  # [fx,dx,ddx, fy,dy,ddy, conf,log_area,miss,dt]

def push_features(fx, fy, conf, area, dt):
    global _hist
    prev  = _hist[-1]
    dx    = fx - prev[0]
    dy    = fy - prev[3]
    ddx   = dx - (prev[0] - _hist[-2, 0]) if _hist.shape[0] >= 2 else 0.0
    ddy   = dy - (prev[3] - _hist[-2, 3]) if _hist.shape[0] >= 2 else 0.0
    row   = np.array([fx, dx, ddx, fy, dy, ddy, conf, np.log1p(area), 0.0, dt], dtype=np.float32)
    _hist = np.roll(_hist, -1, axis=0)
    _hist[-1] = row

def build_tensors():
    x_s = torch.from_numpy(_hist[:, :3]).unsqueeze(0).to(DEVICE)   # (1, T, 3)
    y_s = torch.from_numpy(_hist[:, 3:6]).unsqueeze(0).to(DEVICE)  # (1, T, 3)
    q   = torch.from_numpy(_hist[:, 6:]).unsqueeze(0).to(DEVICE)   # (1, T, 4)
    return x_s, y_s, q

# ── Benchmark loop ────────────────────────────────────────────────────────────
STAGES = ["camera", "yolo", "kalman", "lpf", "features", "dscgnet", "motor"]
times  = {s: [] for s in STAGES}

def tick():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()

total = WARMUP + MEASURE
for i in range(total):
    record = i >= WARMUP

    t0 = tick()
    frame = grab_frame()
    t1 = tick()
    cx, cy, conf, area = fake_yolo(frame)
    t2 = tick()
    fx, fy = kalman_update(cx, cy)
    t3 = tick()
    fx, fy = lpf(fx, fy)
    t4 = tick()
    dt = 1.0 / 30.0
    push_features(fx, fy, conf, area, dt)
    x_s, y_s, q = build_tensors()
    t5 = tick()
    with torch.no_grad():
        pred = model(x_s, y_s, q)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t6 = tick()
    # Motor stage: clamp + scale (trivial CPU op)
    ux = float(pred[0, 0].cpu()) * 20.0
    uy = float(pred[0, 1].cpu()) * 20.0
    t7 = tick()

    if record:
        ms = lambda a, b: (b - a) * 1e3
        times["camera"].append(ms(t0, t1))
        times["yolo"].append(ms(t1, t2))
        times["kalman"].append(ms(t2, t3))
        times["lpf"].append(ms(t3, t4))
        times["features"].append(ms(t4, t5))
        times["dscgnet"].append(ms(t5, t6))
        times["motor"].append(ms(t6, t7))

if USE_CAM:
    cap.release()

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\nDevice: {DEVICE}  |  Warmup: {WARMUP}  |  Measured: {MEASURE} frames\n")
fmt = "{:<12} {:>10} {:>10} {:>10} {:>10}"
print(fmt.format("Stage", "mean_ms", "std_ms", "min_ms", "max_ms"))
print("-" * 54)
total_mean = 0.0
for s in STAGES:
    arr = np.array(times[s])
    m = arr.mean(); total_mean += m
    print(fmt.format(s, f"{m:.3f}", f"{arr.std():.3f}", f"{arr.min():.3f}", f"{arr.max():.3f}"))
print("-" * 54)
print(fmt.format("TOTAL", f"{total_mean:.3f}", "", "", ""))
