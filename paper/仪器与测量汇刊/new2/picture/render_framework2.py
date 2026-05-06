import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from pathlib import Path
import numpy as np

# Canvas: drawio 2000x800 -> figure 14x6 inches
FW, FH = 14.0, 6.0
DW, DH = 2000.0, 800.0

def dx(x): return x / DW * FW
def dy(y): return (DH - y) / DH * FH  # flip y

def dw(w): return w / DW * FW
def dh(h): return h / DH * FH

fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')

# --- Node definitions: (key, x, y, w, h, fill, stroke, label_lines, bold, dashed)
nodes = [
    ('det',   232,  429, 140,  70, '#dae8fc', '#6c8ebf',
     ['Object Detection', 'Bounding box center', r'$c(t)=[x_c, y_c]$'], False, False),
    ('kalman',395,  429, 150,  70, '#f8cecc', '#b85450',
     ['Kalman Filter', 'Filter bbox coords', r'$\rightarrow \hat{c}(t), \hat{v}(t)$'], False, False),
    ('sub',   568,  429, 155,  70, '#fff2cc', '#d6b656',
     ['Subtract Cam Center', r'$e_f(t)=\hat{c}(t)-c_{cam}$'], False, False),
    ('clamp', 745,  429, 120,  70, '#fff2cc', '#d6b656',
     ['Step Clamp', r'$\rightarrow \tilde{e}(t)$'], False, False),
    ('lpf',   889,  429, 150,  70, '#f8cecc', '#b85450',
     ['Low-Pass Filter', r'$u_{legacy}(t)$', '[first_filter_dx/dy]'], True, False),
    ('feat',  883.5,239.5,165,115, '#d5e8d4', '#82b366',
     ['Feature Construction', '(T=16)', r'X: $[f_x,\Delta f_x,\Delta^2 f_x]$',
      r'Y: $[f_y,\Delta f_y,\Delta^2 f_y]$', 'Quality: [conf,logA,miss,dt]'], False, False),
    ('net',   1082, 262, 230,  70, '#f5f5f5', '#555555',
     ['DSCGNet', 'Dual-stream causal encoder', r'$\Delta$-head $(z_t)$ + Gate-head $(g_t)$'], True, True),
    ('trans', 1348, 257, 210,  80, '#d0cee2', '#56517e',
     ['State Transition Output',
      r'$\Delta\hat{u}=g_t\odot(r_{max}\odot\tanh(z_t))$',
      r'$\hat{u}_{t+1}=u_{legacy}(t)+\Delta\hat{u}$'], False, False),
    ('out',   1597, 267, 150,  60, '#d5e8d4', '#82b366',
     [r'Predicted $\hat{u}_{t+1}$', r'$(pred\_ux, pred\_uy)$'], True, False),
    ('motor', 1612, 411, 120,  50, '#f5f5f5', '#666666',
     ['Motor Control', 'Pan-tilt drive'], False, False),
    ('dn',    232,  340, 180,  26, '#fff2cc', '#d6b656',
     [r'$\approx$1-frame visual delay'], False, False),
]

# Draw nodes, store centers
centers = {}
for key, x, y, w, h, fill, stroke, lines, bold, dashed in nodes:
    cx = dx(x + w/2)
    cy = dy(y + h/2)
    centers[key] = (cx, cy, dw(w), dh(h))

    lw = 1.8 if bold else 1.2
    ls = '--' if dashed else '-'
    box = FancyBboxPatch((dx(x), dy(y+h)), dw(w), dh(h),
                         boxstyle="round,pad=0.01",
                         facecolor=fill, edgecolor=stroke,
                         linewidth=lw, linestyle=ls,
                         transform=ax.transData, zorder=2)
    ax.add_patch(box)

    n = len(lines)
    for i, line in enumerate(lines):
        yoff = (i - (n-1)/2) * (dh(h) / (n + 0.5))
        fs = 5.5 if bold else 5.2
        ax.text(cx, cy + yoff, line, ha='center', va='center',
                fontsize=fs, fontweight='bold' if bold else 'normal',
                wrap=False, zorder=3)

# Title
ax.text(dx(236 + 700), dy(141 + 20),
        'Legacy Control Signal Generation and End-to-End One-Frame-Ahead Prediction Framework',
        ha='center', va='center', fontsize=7.5, fontweight='bold', zorder=3)

# --- Arrows helper
def arrow(ax, x1, y1, x2, y2, color='#444444', lw=1.2, ls='-', label=None, label_side='top'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle='arc3,rad=0.0'),
                zorder=4)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        off = 0.08 if label_side == 'top' else -0.08
        ax.text(mx, my + off, label, ha='center', va='center',
                fontsize=5.5, color=color, zorder=5)

def right_edge(key):
    cx, cy, w, h = centers[key]
    return cx + w/2, cy

def left_edge(key):
    cx, cy, w, h = centers[key]
    return cx - w/2, cy

def top_edge(key):
    cx, cy, w, h = centers[key]
    return cx, cy + h/2

def bottom_edge(key):
    cx, cy, w, h = centers[key]
    return cx, cy - h/2

# Horizontal chain: det -> kalman -> sub -> clamp -> lpf
for a, b in [('det','kalman'),('kalman','sub'),('sub','clamp'),('clamp','lpf')]:
    x1,y1 = right_edge(a)
    x2,y2 = left_edge(b)
    arrow(ax, x1, y1, x2, y2)

# lpf -> feat (upward)
x1,y1 = top_edge('lpf')
x2,y2 = bottom_edge('feat')
arrow(ax, x1, y1, x2, y2, label='history', label_side='top')

# feat -> net -> trans -> out
for a, b in [('feat','net'),('net','trans'),('trans','out')]:
    x1,y1 = right_edge(a)
    x2,y2 = left_edge(b)
    arrow(ax, x1, y1, x2, y2)

# out -> motor (downward)
x1,y1 = bottom_edge('out')
x2,y2 = top_edge('motor')
arrow(ax, x1, y1, x2, y2)

# lpf -> trans (dashed red, labeled u_legacy(t))
# route: right of lpf, up, then right to left of trans
lx, ly = right_edge('lpf')
tx, ty = left_edge('trans')
# draw as two-segment path via annotation with angle connector
mid_y = dy(200)  # above feat
ax.annotate('', xy=(tx, ty), xytext=(lx, ly),
            arrowprops=dict(arrowstyle='->', color='#b85450', lw=1.2,
                            linestyle='dashed',
                            connectionstyle='angle,angleA=90,angleB=180,rad=5'),
            zorder=4)
ax.text((lx+tx)/2, mid_y + 0.15, r'$u_{legacy}(t)$',
        ha='center', va='center', fontsize=5.5, color='#b85450', zorder=5)

plt.tight_layout(pad=0.2)
output_dir = Path(__file__).resolve().parent
png_path = output_dir / "方法总体框架-2-hd.png"
pdf_path = output_dir / "方法总体框架-2-hd.pdf"
plt.savefig(str(png_path), dpi=400, bbox_inches='tight', facecolor='white')
plt.savefig(str(pdf_path), bbox_inches='tight', facecolor='white')
print(f"Saved PNG to {png_path}")
print(f"Saved PDF to {pdf_path}")
