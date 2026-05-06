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
SHOW = 200

def future(sig):
    f = np.roll(sig, -1); f[-1] = sig[-1]; return f

d1x = np.concatenate([[0], np.diff(ex)])
d1y = np.concatenate([[0], np.diff(ey)])
ux = ex + alpha * np.clip(d1x, -delta, delta)
uy = ey + alpha * np.clip(d1y, -delta, delta)
fut_x, fut_y = future(ex), future(ey)

def stats(u, ref_next):
    mae  = np.mean(np.abs(u - ref_next))
    rmse = np.sqrt(np.mean((u - ref_next)**2))
    return mae, rmse

mae_x, rmse_x = stats(ux, fut_x)
mae_y, rmse_y = stats(uy, fut_y)

t = np.arange(SHOW)
# compensation amounts
comp_x = ux - ex          # u(t) - e_ref(t)
lead_x = fut_x - ex       # e_ref(t+1) - e_ref(t)  (ideal lead)
comp_y = uy - ey
lead_y = fut_y - ey

fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
fig.subplots_adjust(hspace=0.45)

# --- signal subplots (top 2) ---
for ax, ref, u, fut, ch, mae, rmse in zip(
        axes[:2],
        [ex, ey], [ux, uy], [fut_x, fut_y],
        ['X', 'Y'], [mae_x, mae_y], [rmse_x, rmse_y]):
    ax.plot(t, ref[:SHOW], '#1f77b4', lw=2.5, ls='-',  label='$e_{ref}(t)$')
    ax.plot(t, fut[:SHOW], '#2ca02c', lw=1.5, ls='--', label='$e_{ref}(t+1)$ (ideal)', zorder=3)
    ax.plot(t, u[:SHOW],   '#d62728', lw=1.5, ls=':',  label='Adaptive Fusion $u(t)$', zorder=4)
    ax.set_ylabel(f'Signal {ch} (px)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', ncol=3)
    ax.text(0.02, 0.05, f'MAE={mae:.3f}  RMSE={rmse:.3f}',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

# --- compensation subplots (bottom 2): show the actual advance correction ---
for ax, comp, lead, ch in zip(
        axes[2:],
        [comp_x, comp_y], [lead_x, lead_y],
        ['X', 'Y']):
    ax.axhline(0, color='k', lw=0.8, ls='-')
    ax.plot(t, lead[:SHOW], '#2ca02c', lw=1.5, ls='--', label='Ideal lead $e_{ref}(t+1)-e_{ref}(t)$')
    ax.plot(t, comp[:SHOW], '#d62728', lw=1.8, ls='-',  label='Applied compensation $u(t)-e_{ref}(t)$')
    ax.set_ylabel(f'Compensation {ch} (px)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best', ncol=2)

axes[3].set_xlabel('Frame Index', fontsize=11)
axes[0].set_title('Test Set: Signal Comparison and Compensation Detail', fontsize=12, fontweight='bold')

out = 'D:/kun-data/kun-code-data/反无/cpp智能控制/paper/仪器与测量汇刊/new2/picture/fig_test_prediction.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print('Saved:', out)
print(f'X: MAE={mae_x:.3f} RMSE={rmse_x:.3f}')
print(f'Y: MAE={mae_y:.3f} RMSE={rmse_y:.3f}')
