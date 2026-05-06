import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
import numpy as np

# Canvas: drawio ~1200x900, scale to figure
SCALE = 1 / 72  # drawio px -> inches (approx)
FIG_W, FIG_H = 14.0, 11.0

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1200)
ax.set_ylim(0, 900)
ax.invert_yaxis()
ax.axis('off')

# ── node data ──────────────────────────────────────────────────────────────
vertices = [
    # id, label, x, y, w, h, fill, stroke, fontColor, shape
    (8,  "X-stream Input\n[B,T,3]: fx,Δfx,Δ²fx",   105,212,120,60,'#dae8fc','#6c8ebf','#000000','rect'),
    (9,  "Y-stream Input\n[B,T,3]: fy,Δfy,Δ²fy",    276,212,120,60,'#dae8fc','#6c8ebf','#000000','rect'),
    (10, "Quality Input\n[B,T,4]: conf,logA,miss,dt",442,212,120,60,'#dae8fc','#6c8ebf','#000000','rect'),
    (11, "CausalConv1d\n(3→32,k=3,d=1)",             105,290,120,60,'#d5e8d4','#82b366','#000000','rect'),
    (13, "CausalConv1d\n(32→32,k=3,d=2)",            105,370,120,60,'#d5e8d4','#82b366','#000000','rect'),
    (14, "GRU\n(32→32)",                              105,450,120,60,'#fff2cc','#d6b656','#000000','rect'),
    (15, "CausalSelfAttn\n(dim=32)",                  105,530,120,60,'#f8cecc','#b85450','#000000','rect'),
    (16, "Last step→hx\n[B,32]",                     105,610,120,60,'#6d8764','#3A5431','#ffffff','rect'),
    (40, "CausalConv1d\n(3→32,k=3,d=1)",             276,290,120,60,'#d5e8d4','#82b366','#000000','rect'),
    (42, "CausalConv1d\n(32→32,k=3,d=2)",            276,370,120,60,'#d5e8d4','#82b366','#000000','rect'),
    (44, "GRU\n(32→32)",                              276,450,120,60,'#fff2cc','#d6b656','#000000','rect'),
    (46, "CausalSelfAttn\n(dim=32)",                  276,530,120,60,'#f8cecc','#b85450','#000000','rect'),
    (47, "Last step→hy\n[B,32]",                     276,610,120,60,'#6d8764','#3A5431','#ffffff','rect'),
    (29, "Linear(4→16)\n+ReLU",                      442,290,120,60,'#f5f5f5','#666666','#000000','rect'),
    (30, "GRU\n(16→16)",                              442,370,120,60,'#fff2cc','#d6b656','#000000','rect'),
    (31, "Last step→hq\n[B,16]",                     442,450,120,60,'#6d8764','#3A5431','#ffffff','rect'),
    (63, "Concat(hx,hy)\nh[B,64]",                   208,718, 87, 80,'#76608a','#432D57','#ffffff','ellipse'),
    (61, "Concat(h,hq)\nh[B,80]",                    462,718, 80, 80,'#76608a','#432D57','#ffffff','ellipse'),
    (67, "Δ-head\nLinear→ReLU→Linear(2)→z_t[B,2]",  604,287,241,60,'#ffe6cc','#d79b00','#000000','rect'),
    (68, "Gate-head\nLinear→ReLU→Linear(2)→Sigmoid→g_t[B,2]",603,510,242,60,'#ffe6cc','#d79b00','#000000','rect'),
    (71, "Δû=g_t⊙(r_max⊙tanh(z_t))",               929,287,120,60,'#d0cee2','#56517e','#000000','rect'),
    (72, "û_{t+1}=e_f(t)+Δû",                       929,456,120,60,'#d0cee2','#56517e','#000000','rect'),
    (73, "Output: û_{t+1}[B,2]\n(pred_ux,pred_uy)",  931,625,120,60,'#d5e8d4','#82b366','#000000','rect'),
]

edges = [
    (8,11),(9,40),(10,29),(11,13),(13,14),(14,15),(15,16),
    (40,42),(42,44),(44,46),(46,47),
    (29,30),(30,31),
    (16,63),(47,63),(31,61),
    (63,61),
    (61,67),(61,68),
    (67,71),(68,71),
    (71,72),(72,73),
]

# build id->node dict
node = {v[0]: v for v in vertices}

def center(v):
    _, _, x, y, w, h = v[0], v[1], v[2], v[3], v[4], v[5]
    return x + w/2, y + h/2

# ── title (node 76) ────────────────────────────────────────────────────────
ax.text(102 + 995/2, 77 + 31,
        "DSCGNet: Dual-Stream Causal GRU Network\nfor One-Frame-Ahead Control Prediction",
        ha='center', va='center', fontsize=9, fontweight='bold', color='#000000')

# ── draw edges first ───────────────────────────────────────────────────────
for src_id, dst_id in edges:
    s = node[src_id]
    d = node[dst_id]
    sx, sy = center(s)
    dx, dy = center(d)
    ax.annotate("", xy=(dx, dy), xytext=(sx, sy),
                arrowprops=dict(arrowstyle='->', color='#555555',
                                lw=0.8, connectionstyle='arc3,rad=0.0'),
                zorder=1)

SHADOW_OFF = 4   # shadow offset in drawio px
SHADOW_COLOR = '#b0b0b0'

# ── draw nodes ─────────────────────────────────────────────────────────────
for v in vertices:
    vid, label, x, y, w, h, fill, stroke, fc, shape = v
    cx, cy = x + w/2, y + h/2

    if shape == 'ellipse':
        # shadow
        ax.add_patch(Ellipse((cx + SHADOW_OFF, cy + SHADOW_OFF), w, h,
                             facecolor=SHADOW_COLOR, edgecolor='none', zorder=1))
        patch = Ellipse((cx, cy), w, h,
                        facecolor=fill, edgecolor=stroke, linewidth=1.2, zorder=2)
        ax.add_patch(patch)
    else:
        # shadow
        ax.add_patch(FancyBboxPatch((x + SHADOW_OFF, y + SHADOW_OFF), w, h,
                                    boxstyle="round,pad=3",
                                    facecolor=SHADOW_COLOR, edgecolor='none', zorder=1))
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=3",
                               facecolor=fill, edgecolor=stroke, linewidth=1.2, zorder=2)
        ax.add_patch(patch)

    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=7.5, color=fc, zorder=3,
            multialignment='center',
            wrap=False)

# ── save ───────────────────────────────────────────────────────────────────
out = r"D:\kun-data\kun-code-data\反无\cpp智能控制\paper\仪器与测量汇刊\new2\picture\模型.png"
plt.tight_layout(pad=0.2)
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
