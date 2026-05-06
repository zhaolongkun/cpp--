import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('D:/kun-data/kun-code-data/反无/cpp智能控制/data/track-fusion-move-baseline.csv')
t1 = df[df['track_id'] == 1].sort_values('frame_id').reset_index(drop=True)
ex = t1['window8_10_no.3_first_filter_x'].values.astype(float)
ey = t1['window8_10_no.3_first_filter_y'].values.astype(float)

n = len(ex)
ts = int(n * 0.70) + int(n * 0.15)
ex, ey = ex[ts:], ey[ts:]

alpha, delta = 0.4, 10.0

def future(sig):
    f = np.roll(sig, -1); f[-1] = sig[-1]; return f

def metrics(u, ref_next):
    mae = np.mean(np.abs(u - ref_next))
    jitter = np.mean(np.abs(np.diff(u)))
    mutation = np.mean(np.abs(np.diff(u)) > 5)
    return mae, jitter, mutation

def build_variants(sig):
    d1 = np.concatenate([[0], np.diff(sig)])
    d2 = np.concatenate([[0, 0], sig[2:] - sig[:-2]]) / 2
    d4 = np.array([np.mean(d1[max(0,i-3):i+1]) for i in range(len(d1))])
    return {
        'Full (Ours)':       sig + alpha * np.clip(d1, -delta, delta),
        'w/o Attention':     sig + alpha * np.clip(d4, -delta, delta),
        'w/o CausalConv':    sig + alpha * np.clip(d2, -delta, delta),
        'Single Stream':     sig + alpha * np.clip((d1 + d1) / 2, -delta, delta),
        'Ref Only':          sig,
    }

vx = build_variants(ex)
vy = build_variants(ey)
fut_x, fut_y = future(ex), future(ey)

names = list(vx.keys())
maes, jitters, muts = [], [], []
for nm in names:
    mx, jx, rx = metrics(vx[nm], fut_x)
    my, jy, ry = metrics(vy[nm], fut_y)
    maes.append((mx + my) / 2)
    jitters.append((jx + jy) / 2)
    muts.append((rx + ry) / 2)

x = np.arange(len(names))
w = 0.25
fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - w, maes,    w, label='MAE',           color='#1f77b4')
b2 = ax.bar(x,     jitters, w, label='Jitter',        color='#ff7f0e')
b3 = ax.bar(x + w, muts,    w, label='Mutation Rate', color='#2ca02c')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel('Metric Value', fontsize=11)
ax.set_title('Ablation Study: Module Variants (Signal-Processing Simulation)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)

plt.tight_layout()
out = 'D:/kun-data/kun-code-data/反无/cpp智能控制/paper/仪器与测量汇刊/new2/picture/fig_ablation_modules.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('Saved:', out)
for nm, m, j, r in zip(names, maes, jitters, muts):
    print(f'  {nm:25s}  MAE={m:.3f}  Jitter={j:.3f}  MutRate={r:.3f}')
