import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('D:/kun-data/kun-code-data/反无/cpp智能控制/paper/仪器与测量汇刊/new2/picture/fig_ablation_modules/ablation_results.json') as f:
    res = json.load(f)

labels = ['Full\n(Proposed)', 'w/o\nAttention', 'w/o\nCausalConv', 'Single\nStream']
keys = ['full', 'no_attn', 'no_conv', 'single_stream']

mae      = [res[k]['mae'] for k in keys]
fmae     = [res[k]['fusion_mae'] for k in keys]
jitter   = [res[k]['jitter'] for k in keys]
mut_rate = [res[k]['mutation_rate'] * 100 for k in keys]  # to %

x = np.arange(len(labels))
w = 0.55

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
colors = ['#2166ac', '#92c5de', '#f4a582', '#d6604d']

for ax, vals, ylabel, title in zip(
    axes,
    [mae, jitter, mut_rate],
    ['Prediction MAE', 'Jitter', 'Mutation Rate (%)'],
    ['(a) Prediction MAE\n(lower is better)',
     '(b) Jitter\n(lower is better)',
     '(c) Mutation Rate (%)\n(lower is better)']
):
    bars = ax.bar(x, vals, width=w, color=colors, edgecolor='k', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{v:.4f}' if v < 1 else f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Ablation Study: Module Contribution on Test Set', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('D:/kun-data/kun-code-data/反无/cpp智能控制/paper/仪器与测量汇刊/new2/picture/fig_ablation_modules.png',
            dpi=150, bbox_inches='tight')
print("saved")
