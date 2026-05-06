import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'D:/kun-data/kun-code-data/反无/cpp智能控制/模型-new2')
from models.causal_cnn_gru import CausalCNNGRU
from models.ablation_variants import NoAttnCausalCNNGRU, NoConvGRUAttn, SingleStreamCausalCNNGRUAttn

# ── data ──────────────────────────────────────────────────────────────────────
CSV = 'D:/kun-data/kun-code-data/反无/cpp智能控制/data/track-fusion-move-baseline.csv'
T, MIN_SEG = 8, 12

df = pd.read_csv(CSV)
segments = []
for tid, grp in df.groupby('track_id'):
    grp = grp.sort_values('frame_id').reset_index(drop=True)
    start = 0
    for i in range(1, len(grp)):
        if grp.loc[i, 'frame_id'] != grp.loc[i-1, 'frame_id'] + 1:
            if i - start >= MIN_SEG:
                segments.append(grp.iloc[start:i].reset_index(drop=True))
            start = i
    if len(grp) - start >= MIN_SEG:
        segments.append(grp.iloc[start:].reset_index(drop=True))

def make_samples(segs):
    Xs, Ys, refs = [], [], []
    for seg in segs:
        ex = seg['window8_10_no.3_first_filter_x'].values.astype(np.float32)
        ey = seg['window8_10_no.3_first_filter_y'].values.astype(np.float32)
        dx = np.diff(ex, prepend=ex[0])
        dy = np.diff(ey, prepend=ey[0])
        feat = np.stack([ex, ey, dx, dy], axis=1)  # (N,4)
        for i in range(T, len(seg)):
            Xs.append(feat[i-T:i])
            Ys.append(np.array([ex[i]-ex[i-1], ey[i]-ey[i-1]], dtype=np.float32))
            refs.append(np.array([ex[i-1], ey[i-1], ex[i], ey[i]], dtype=np.float32))
    return np.array(Xs), np.array(Ys), np.array(refs)

n = len(segments)
n_tr = int(n * 0.70)
n_va = int(n * 0.15)
tr_segs = segments[:n_tr]
va_segs = segments[n_tr:n_tr+n_va]
te_segs = segments[n_tr+n_va:]

X_tr, Y_tr, _ = make_samples(tr_segs)
X_va, Y_va, _ = make_samples(va_segs)
X_te, Y_te, refs_te = make_samples(te_segs)

mu = X_tr.reshape(-1, 4).mean(0).astype(np.float32)
sd = X_tr.reshape(-1, 4).std(0).astype(np.float32) + 1e-8

def norm(X): return (X - mu) / sd

Xtn = torch.tensor(norm(X_tr)); Ytn = torch.tensor(Y_tr)
Xvn = torch.tensor(norm(X_va)); Yvn = torch.tensor(Y_va)
Xen = torch.tensor(norm(X_te)); Yen = torch.tensor(Y_te)

tr_loader = DataLoader(TensorDataset(Xtn, Ytn), batch_size=128, shuffle=True)
va_loader = DataLoader(TensorDataset(Xvn, Yvn), batch_size=256)

# ── train ─────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS, PATIENCE, LR, WD = 80, 20, 1e-3, 1e-4

def train_model(model):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.HuberLoss()
    best_val, wait, best_state = float('inf'), 0, None
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
                           for xb, yb in va_loader) / len(Xvn)
        if val_loss < best_val - 1e-6:
            best_val, wait = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model

# ── evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model):
    model.eval()
    with torch.no_grad():
        pred = model(Xen.to(DEVICE)).cpu().numpy()
    Y_np = Yen.numpy()
    mae = float(np.mean(np.abs(pred - Y_np)))

    # fusion: u(t) = e_ref(t) + 0.4*clip(pred,-10,10)
    e_prev = refs_te[:, :2]   # e_ref(t-1) actually stored as e_ref(t) at index 2,3 is next
    # refs_te: [ex[i-1], ey[i-1], ex[i], ey[i]]
    e_cur  = refs_te[:, :2]   # ex[i-1], ey[i-1]
    e_next = refs_te[:, 2:]   # ex[i],   ey[i]
    u = e_cur + 0.4 * np.clip(pred, -10, 10)
    fmae = float(np.mean(np.abs(u - e_next)))
    diff = np.abs(np.diff(u, axis=0))
    jitter = float(np.mean(diff))
    mut_rate = float(np.mean(diff > 5))
    return {'mae': mae, 'fusion_mae': fmae, 'jitter': jitter, 'mutation_rate': mut_rate}

variants = {
    'full':          CausalCNNGRU(input_dim=4, hidden_dim=64, output_dim=2),
    'no_attn':       NoAttnCausalCNNGRU(input_dim=4, hidden_dim=64, output_dim=2),
    'no_conv':       NoConvGRUAttn(input_dim=4, hidden_dim=64, output_dim=2, max_len=8),
    'single_stream': SingleStreamCausalCNNGRUAttn(input_dim=4, hidden_dim=64, output_dim=2, max_len=8),
}

results = {}
for name, model in variants.items():
    print(f'Training {name}...')
    train_model(model)
    results[name] = evaluate(model)
    print(f'  {results[name]}')

OUT_DIR = 'D:/kun-data/kun-code-data/反无/cpp智能控制/paper/仪器与测量汇刊/new2/picture/fig_ablation_modules'
with open(f'{OUT_DIR}/ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── plot ──────────────────────────────────────────────────────────────────────
labels = ['Full', 'w/o Attn', 'w/o Conv', 'Single Stream']
keys   = ['full', 'no_attn', 'no_conv', 'single_stream']
metrics = [('Fusion MAE', 'fusion_mae'), ('Jitter', 'jitter'), ('Mutation Rate', 'mutation_rate')]

x = np.arange(len(labels))
fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
colors = ['#2166ac', '#d1e5f0', '#f4a582', '#d6604d']

for ax, (title, key) in zip(axes, metrics):
    vals = [results[k][key] for k in keys]
    bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor='black', linewidth=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(title, fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('D:/kun-data/kun-code-data/反无/cpp智能控制/paper/仪器与测量汇刊/new2/picture/fig_ablation_modules.png',
            dpi=150, bbox_inches='tight')
print('Saved fig_ablation_modules.png')
print(json.dumps(results, indent=2))
