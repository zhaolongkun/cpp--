import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('D:/kun-data/kun-code-data/反无/cpp智能控制/data/track-fusion-move-baseline.csv')
t1 = df[df['track_id'] == 1].sort_values('frame_id').reset_index(drop=True)
ex = t1['window8_10_no.3_first_filter_x'].values.astype(float)
ey = t1['window8_10_no.3_first_filter_y'].values.astype(float)

# ── Train/val/test split ───────────────────────────────────────────────────
n = len(ex)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)
# test starts at n_train + n_val
ts = n_train + n_val
ex_test = ex[ts:]
ey_test = ey[ts:]

# ── Methods ────────────────────────────────────────────────────────────────
def kalman_predict(sig):
    """Simple scalar Kalman filter, returns one-step-ahead prediction."""
    x_est, p = sig[0], 1.0
    Q, R = 0.1, 1.0
    pred = np.zeros_like(sig)
    for i in range(len(sig)):
        # predict
        x_pred = x_est
        p_pred = p + Q
        # update with current measurement
        K = p_pred / (p_pred + R)
        x_est = x_pred + K * (sig[i] - x_pred)
        p = (1 - K) * p_pred
        pred[i] = x_est  # filtered estimate used as prediction for next step
    # shift by 1: prediction for t is estimate at t-1
    return np.roll(pred, 1)

def compute_metrics(u, ref_next):
    """MAE vs ref(t+1), jitter, mutation rate."""
    mae = np.mean(np.abs(u - ref_next))
    jitter = np.mean(np.abs(np.diff(u)))
    diff_u = np.diff(u)
    mutation = np.mean(np.abs(diff_u) > 5)
    return mae, jitter, mutation

alpha, delta = 0.4, 10.0
SHOW = 100  # frames to display

def build_methods(sig):
    n = len(sig)
    ref = sig                                          # e_ref(t)
    future = np.roll(sig, -1); future[-1] = sig[-1]   # e_ref(t+1)
    diff1 = np.concatenate([[0], np.diff(sig)])

    kalman = kalman_predict(sig)
    smith  = ref + diff1                               # e_ref(t) + delta_1
    pred   = diff1                                     # approximate prediction
    fusion = ref + alpha * np.clip(pred, -delta, delta)

    return ref, kalman, smith, fusion, future

ref_x, kal_x, smi_x, fus_x, fut_x = build_methods(ex_test)
ref_y, kal_y, smi_y, fus_y, fut_y = build_methods(ey_test)

# ── Metrics ────────────────────────────────────────────────────────────────
methods_x = [ref_x, kal_x, smi_x, fus_x]
methods_y = [ref_y, kal_y, smi_y, fus_y]
labels = ['Ref Only', 'Kalman', 'Smith', 'Adaptive Fusion']

rows = []
for lbl, mx, my in zip(labels, methods_x, methods_y):
    mae_x, jit_x, mut_x = compute_metrics(mx, fut_x)
    mae_y, jit_y, mut_y = compute_metrics(my, fut_y)
    rows.append((lbl,
                 (mae_x + mae_y) / 2,
                 (jit_x + jit_y) / 2,
                 (mut_x + mut_y) / 2))

# ── Plot ───────────────────────────────────────────────────────────────────
# pick a dynamic segment: find window with largest variance
def best_window(sig, win=SHOW):
    vars_ = [np.var(sig[i:i+win]) for i in range(len(sig)-win)]
    return int(np.argmax(vars_))

start = best_window(ex_test)
t = np.arange(SHOW)

# distinct styles: color + linestyle + linewidth
plot_cfg = [
    ('Ref Only',          '#1f77b4', '-',  1.8, 0.9),
    ('Kalman',            '#ff7f0e', '--', 1.8, 0.9),
    ('Smith',             '#2ca02c', '-.',  2.0, 0.9),
    ('Adaptive Fusion',   '#d62728', '-',  2.5, 1.0),
    ('Future Ref (ideal)','#9467bd', ':',  1.5, 0.7),
]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.subplots_adjust(hspace=0.35, bottom=0.28)

for ax, signals, ch in zip(
        axes,
        [(ref_x, kal_x, smi_x, fus_x, fut_x),
         (ref_y, kal_y, smi_y, fus_y, fut_y)],
        ['X', 'Y']):
    for sig, (lbl, color, ls, lw, alpha_v) in zip(signals, plot_cfg):
        ax.plot(t, sig[start:start+SHOW], color=color, ls=ls, lw=lw,
                alpha=alpha_v, label=lbl)
    ax.set_ylabel(f'Displacement {ch} (px)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    # clip y-axis to exclude Kalman initialization spikes
    data_all = np.concatenate([s[start:start+SHOW] for s in signals if s is not signals[1]])
    ymin, ymax = np.min(data_all), np.max(data_all)
    margin = (ymax - ymin) * 0.15
    ax.set_ylim(ymin - margin, ymax + margin)

axes[1].set_xlabel('Frame Index', fontsize=11)
axes[0].set_title('Method Comparison on Test Set', fontsize=13, fontweight='bold')

# Metrics table below
col_labels = ['Method', 'MAE', 'Jitter', 'Mut.Rate']
table_data = [[r[0], f'{r[1]:.3f}', f'{r[2]:.3f}', f'{r[3]:.3f}'] for r in rows]
table = fig.add_axes([0.08, 0.02, 0.88, 0.20])
table.axis('off')
tbl = table.table(cellText=table_data, colLabels=col_labels,
                  loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)

out = 'D:/kun-data/kun-code-data/反无/cpp智能控制/paper/仪器与测量汇刊/new2/picture/fig_comparison_methods.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('Saved:', out)
print('Metrics:')
for r in rows:
    print(f'  {r[0]:20s}  MAE={r[1]:.3f}  Jitter={r[2]:.3f}  MutRate={r[3]:.3f}')
