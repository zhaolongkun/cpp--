import numpy as np
import torch
from models.causal_cnn_gru import CausalCNNGRU
from prepare_data import causal_moving_average, prepare_sequences, split_data, normalize_data

def compute_metrics(u, e_ref_future, threshold=10):
    mae = np.mean(np.abs(u - e_ref_future))
    jitter = np.mean(np.abs(u[1:] - u[:-1]))
    spike_rate = np.mean(np.abs(u[1:] - u[:-1]) > threshold)
    return mae, jitter, spike_rate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = np.load('checkpoints/data.npz')
model = CausalCNNGRU().to(device)
model.load_state_dict(torch.load('checkpoints/causal_cnn_gru.pth'))
model.eval()

# 表3: 窗口大小消融
print("=== Table 3: Window Size Ablation ===")
for W in [2, 3]:
    e_raw = data['e_raw']
    e_ref = causal_moving_average(e_raw, window_size=W)
    e_ref_future = np.roll(e_ref, -1, axis=0)
    
    n_total = len(e_raw) - 8
    start_idx = int(n_total * 0.85) + 8
    e_ref_test = e_ref[start_idx:start_idx+len(data['X_test'])]
    e_ref_future_test = e_ref_future[start_idx:start_idx+len(data['X_test'])]
    
    with torch.no_grad():
        delta_pred = model(torch.FloatTensor(data['X_test']).to(device)).cpu().numpy()
    
    u = e_ref_test + 0.4 * np.clip(delta_pred, -10, 10)
    mae, jitter, spike = compute_metrics(u, e_ref_future_test)
    print(f"$W={W}$ & {mae:.2f} & {jitter:.2f} & {spike:.3f} \\\\")

# 表7: 融合策略消融
print("\n=== Table 7: Fusion Strategy Ablation ===")
e_ref = causal_moving_average(data['e_raw'], window_size=2)
n_total = len(data['e_raw']) - 8
start_idx = int(n_total * 0.85) + 8
e_ref_test = e_ref[start_idx:start_idx+len(data['X_test'])]
e_ref_future = np.roll(e_ref_test, -1, axis=0)

with torch.no_grad():
    delta_pred = model(torch.FloatTensor(data['X_test']).to(device)).cpu().numpy()

strategies = [
    ("Ref Only", e_ref_test),
    ("Prediction Only", e_ref_test + delta_pred),
    ("Fixed Fusion", e_ref_test + 0.4 * np.clip(delta_pred, -10, 10)),
]

for name, u in strategies:
    mae, jitter, spike = compute_metrics(u, e_ref_future)
    print(f"{name} & {mae:.2f} & {jitter:.2f} & {spike:.3f} \\\\")

velocity = np.abs(e_ref_test[1:] - e_ref_test[:-1]).sum(axis=1)
alpha_adaptive = np.clip(0.2 + velocity / 20, 0.2, 0.6)
u_adaptive = e_ref_test[:-1] + alpha_adaptive[:, None] * np.clip(delta_pred[:-1], -10, 10)
mae, jitter, spike = compute_metrics(u_adaptive, e_ref_future[:-1])
print(f"Adaptive Fusion & {mae:.2f} & {jitter:.2f} & {spike:.3f} \\\\")

# 表8: 融合参数消融
print("\n=== Table 8: Fusion Parameter Ablation ===")
for alpha in [0.2, 0.4, 0.6]:
    u = e_ref_test + alpha * np.clip(delta_pred, -10, 10)
    mae, jitter, spike = compute_metrics(u, e_ref_future)
    print(f"{alpha} & 10 & {mae:.2f} & {jitter:.2f} & {spike:.3f} \\\\")

for delta_val in [5, 15]:
    u = e_ref_test + 0.4 * np.clip(delta_pred, -delta_val, delta_val)
    mae, jitter, spike = compute_metrics(u, e_ref_future)
    print(f"0.4 & {delta_val} & {mae:.2f} & {jitter:.2f} & {spike:.3f} \\\\")
