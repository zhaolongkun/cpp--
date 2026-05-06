"""
DSCGNet architecture diagram — clean grid layout for publication.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── colours ──────────────────────────────────────────────────────────────────
C = {
    "input":  "#BDD7EE",   # blue
    "conv":   "#C6EFCE",   # green
    "gru":    "#FFEB9C",   # yellow
    "attn":   "#FCE4D6",   # orange
    "proj":   "#E2EFDA",   # light green
    "concat": "#D9D2E9",   # purple
    "head":   "#FCE4D6",   # salmon
    "trans":  "#EDEDED",   # grey
    "out":    "#D9EAD3",   # output green
    "qual":   "#CFE2F3",   # quality blue
}
EDGE = "#404040"
TITLE_COLOR = "#1F3864"

fig_w, fig_h = 18, 11
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── primitives ────────────────────────────────────────────────────────────────
def rect(x, y, w, h, color, label, sub=None, fs=8.5, lw=1.2):
    """Draw a rounded rectangle with centred label (and optional sub-label)."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.07",
                       linewidth=lw, edgecolor=EDGE, facecolor=color,
                       zorder=2)
    ax.add_patch(p)
    cy = y + h / 2 + (0.10 if sub else 0)
    ax.text(x + w/2, cy, label, ha="center", va="center",
            fontsize=fs, color="#1a1a1a", fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.14, sub, ha="center", va="center",
                fontsize=6.8, color="#555555", zorder=3)

def arr(x0, y0, x1, y1, color=EDGE, lw=1.2, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=11),
                zorder=4)

def section_label(x, y, text, color="#1F3864"):
    ax.text(x, y, text, ha="center", va="bottom",
            fontsize=8, color=color, fontweight="bold",
            style="italic", zorder=3)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
# Column x-starts
COL = {
    "input":  0.3,
    "enc":    3.0,
    "concat": 7.6,
    "head":   10.4,
    "trans":  13.5,
}
IW  = 2.4   # input block width
IH  = 0.72  # input block height
EW  = 3.8   # encoder block width
EH  = 0.42  # encoder block height
EG  = 0.10  # gap between encoder blocks
CW  = 2.2   # concat block width
CH  = 0.65
HW  = 2.6   # head block width
HH  = 0.65
TW  = 2.8   # transition block width
TH  = 0.55

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(fig_w/2, 10.55,
        "DSCGNet: Dual-Stream Causal GRU Network for One-Frame-Ahead Control Prediction",
        ha="center", va="center", fontsize=11, fontweight="bold", color=TITLE_COLOR)

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN 1 — Inputs
# ═══════════════════════════════════════════════════════════════════════════════
# X input  (top row)
XI_y = 8.2
rect(COL["input"], XI_y, IW, IH, C["input"],
     "X-stream Input", sub="[B, T, 3]:  fx,  Δfx,  Δ²fx")

# Y input  (middle row)
YI_y = 5.3
rect(COL["input"], YI_y, IW, IH, C["input"],
     "Y-stream Input", sub="[B, T, 3]:  fy,  Δfy,  Δ²fy")

# Quality input  (bottom row)
QI_y = 2.4
rect(COL["input"], QI_y, IW, IH, C["qual"],
     "Quality Input", sub="[B, T, 4]:  conf, logA, miss, dt")

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN 2 — Encoders
# ═══════════════════════════════════════════════════════════════════════════════
ex = COL["enc"]

# ── X encoder ─────────────────────────────────────────────────────────────────
X_blocks = [
    ("CausalConv1d  (3 → 32,  k=3, d=1)",  C["conv"]),
    ("CausalConv1d  (32 → 32,  k=3, d=2)", C["conv"]),
    ("GRU  (32 → 32)",                      C["gru"]),
    ("CausalSelfAttn  (dim=32)",            C["attn"]),
    ("Last step  →  hx   [B, 32]",         C["proj"]),
]
X_top = 9.55
y = X_top
for label, color in X_blocks:
    rect(ex, y - EH, EW, EH, color, label, fs=8)
    y -= EH + EG
X_bot = y + EG   # bottom of last block

section_label(ex + EW/2, X_top + 0.05,
              "MotionStreamEncoder  (X-axis)", "#1A5276")

# ── Y encoder ─────────────────────────────────────────────────────────────────
Y_blocks = [
    ("CausalConv1d  (3 → 32,  k=3, d=1)",  C["conv"]),
    ("CausalConv1d  (32 → 32,  k=3, d=2)", C["conv"]),
    ("GRU  (32 → 32)",                      C["gru"]),
    ("CausalSelfAttn  (dim=32)",            C["attn"]),
    ("Last step  →  hy   [B, 32]",         C["proj"]),
]
Y_top = 6.55
y = Y_top
for label, color in Y_blocks:
    rect(ex, y - EH, EW, EH, color, label, fs=8)
    y -= EH + EG
Y_bot = y + EG

section_label(ex + EW/2, Y_top + 0.05,
              "MotionStreamEncoder  (Y-axis)", "#1A5276")

# ── Quality encoder ────────────────────────────────────────────────────────────
Q_blocks = [
    ("Linear  (4 → 16)  +  ReLU",       C["qual"]),
    ("GRU  (16 → 16)",                   C["gru"]),
    ("Last step  →  hq   [B, 16]",      C["proj"]),
]
Q_top = 3.55
y = Q_top
for label, color in Q_blocks:
    rect(ex, y - EH, EW, EH, color, label, fs=8)
    y -= EH + EG
Q_bot = y + EG

section_label(ex + EW/2, Q_top + 0.05,
              "QualityBranchEncoder", "#6C3483")

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN 3 — Concat
# ═══════════════════════════════════════════════════════════════════════════════
cx = COL["concat"]

# Concat hx + hy
C1_y = 6.8
rect(cx, C1_y, CW, CH, C["concat"],
     "Concat (hx, hy)", sub="h  [B, 64]")

# Concat h + hq
C2_y = 4.5
rect(cx, C2_y, CW, CH, C["concat"],
     "Concat (h, hq)", sub="h  [B, 80]")

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN 4 — Dual output heads
# ═══════════════════════════════════════════════════════════════════════════════
hx = COL["head"]

# Δ-head
D_y = 6.5
rect(hx, D_y, HW, HH, C["head"],
     "Δ-head", sub="Linear → ReLU → Linear(2)  →  z_t  [B,2]")

# Gate-head
G_y = 4.8
rect(hx, G_y, HW, HH, C["head"],
     "Gate-head", sub="Linear → ReLU → Linear(2) → Sigmoid  →  g_t  [B,2]")

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN 5 — State transition + output
# ═══════════════════════════════════════════════════════════════════════════════
tx = COL["trans"]

T1_y = 6.8
rect(tx, T1_y, TW, TH, C["trans"],
     "Δû = g_t ⊙ (r_max ⊙ tanh(z_t))", fs=8)

T2_y = 5.6
rect(tx, T2_y, TW, TH, C["trans"],
     "û_{t+1} = e_f(t) + Δû", fs=8.5)

OUT_y = 4.1
rect(tx, OUT_y, TW, 0.75, C["out"],
     "Output:  û_{t+1}   [B, 2]",
     sub="(pred_ux,   pred_uy)", fs=9)

# e_f(t) side annotation
ax.annotate("e_f(t)\n(current frame error)",
            xy=(tx + TW/2, T2_y + TH/2),
            xytext=(tx + TW/2, 3.0),
            ha="center", fontsize=7.5, color="#922B21",
            arrowprops=dict(arrowstyle="-|>", color="#922B21", lw=1.0),
            zorder=4)

# ═══════════════════════════════════════════════════════════════════════════════
# ARROWS
# ═══════════════════════════════════════════════════════════════════════════════
# helper: mid-y of a block
def my(y, h): return y + h/2

# ── inputs → encoders ─────────────────────────────────────────────────────────
arr(COL["input"]+IW, my(XI_y, IH),   ex,        my(X_top-EH, EH))
arr(COL["input"]+IW, my(YI_y, IH),   ex,        my(Y_top-EH, EH))
arr(COL["input"]+IW, my(QI_y, IH),   ex,        my(Q_top-EH, EH))

# ── encoders → concat1 ────────────────────────────────────────────────────────
# hx (bottom of X encoder) → concat1
hx_out_y = X_bot + EH/2 + 0.05   # approx centre of last X block
arr(ex+EW, X_bot + EH/2,   cx,  my(C1_y, CH))

# hy (bottom of Y encoder) → concat1
arr(ex+EW, Y_bot + EH/2,   cx,  my(C1_y, CH))

# ── concat1 → concat2 ─────────────────────────────────────────────────────────
arr(cx + CW/2, C1_y,   cx + CW/2,  C2_y + CH)

# ── hq → concat2 ──────────────────────────────────────────────────────────────
arr(ex+EW, Q_bot + EH/2,   cx,  my(C2_y, CH))

# ── concat2 → heads ───────────────────────────────────────────────────────────
arr(cx+CW, my(C2_y, CH),   hx,  my(D_y, HH))
arr(cx+CW, my(C2_y, CH),   hx,  my(G_y, HH))

# ── heads → state transition ──────────────────────────────────────────────────
arr(hx+HW, my(D_y, HH),   tx,  my(T1_y, TH))
arr(hx+HW, my(G_y, HH),   tx,  my(T1_y, TH))

# ── T1 → T2 → output ─────────────────────────────────────────────────────────
arr(tx+TW/2, T1_y,         tx+TW/2,  T2_y+TH)
arr(tx+TW/2, T2_y,         tx+TW/2,  OUT_y+0.75)

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(facecolor=C["input"],  edgecolor=EDGE, label="Input"),
    mpatches.Patch(facecolor=C["conv"],   edgecolor=EDGE, label="Causal Conv1d"),
    mpatches.Patch(facecolor=C["gru"],    edgecolor=EDGE, label="GRU"),
    mpatches.Patch(facecolor=C["attn"],   edgecolor=EDGE, label="Causal Self-Attn"),
    mpatches.Patch(facecolor=C["proj"],   edgecolor=EDGE, label="Projection / Last step"),
    mpatches.Patch(facecolor=C["concat"], edgecolor=EDGE, label="Concat"),
    mpatches.Patch(facecolor=C["head"],   edgecolor=EDGE, label="Output Head"),
    mpatches.Patch(facecolor=C["trans"],  edgecolor=EDGE, label="State Transition"),
    mpatches.Patch(facecolor=C["out"],    edgecolor=EDGE, label="Output"),
    mpatches.Patch(facecolor=C["qual"],   edgecolor=EDGE, label="Quality Branch"),
]
ax.legend(handles=legend_items, loc="lower center",
          fontsize=7.5, ncol=5, framealpha=0.9,
          bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(pad=0.3)
output_dir = Path(__file__).resolve().parent
png_path = output_dir / "dscgnet_arch_hd.png"
pdf_path = output_dir / "dscgnet_arch_hd.pdf"
plt.savefig(str(png_path), dpi=400, bbox_inches="tight", facecolor="white")
plt.savefig(str(pdf_path), bbox_inches="tight", facecolor="white")
print("Saved PNG:", png_path)
print("Saved PDF:", pdf_path)
raise SystemExit(0)
out = (r"d:\kun-data\kun-code-data\反无\cpp智能控制"
       r"\paper\仪器与测量汇刊\new2\picture\fig_model_arch\dscgnet_arch.png")
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("Saved:", out)
