"""
生成图1和图3：结构示意图（草图模板）
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10

def draw_fig1_method_overview():
    """图1: 方法总体框架"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 定义框的样式
    box_style = dict(boxstyle='round,pad=0.1', facecolor='lightblue', edgecolor='black', linewidth=2)
    arrow_style = dict(arrowstyle='->', lw=2, color='black')

    # 第一行：原始信号处理
    # e_raw(t)
    ax.text(1, 4.5, 'Raw Error\ne_raw(t)', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6E6', edgecolor='black', linewidth=2),
            fontsize=11, weight='bold')

    # 箭头
    ax.annotate('', xy=(2.5, 4.5), xytext=(1.8, 4.5), arrowprops=arrow_style)

    # Causal Moving Average
    ax.text(3.5, 4.5, 'Causal Moving\nAverage (W=2)', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6F3FF', edgecolor='black', linewidth=2),
            fontsize=11, weight='bold')

    # 箭头
    ax.annotate('', xy=(5.2, 4.5), xytext=(4.3, 4.5), arrowprops=arrow_style)

    # e_ref(t)
    ax.text(6.2, 4.5, 'Reliable\nBaseline\ne_ref(t)', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6FFE6', edgecolor='black', linewidth=2),
            fontsize=11, weight='bold')

    # 第二行：预测分支
    # 从e_ref(t)向下的箭头
    ax.annotate('', xy=(6.2, 3.2), xytext=(6.2, 3.8), arrowprops=arrow_style)

    # Feature Construction
    ax.text(6.2, 2.5, 'Feature\nConstruction\n[e_ref, d_ref]', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9E6', edgecolor='black', linewidth=2),
            fontsize=10)

    # 箭头
    ax.annotate('', xy=(7.8, 2.5), xytext=(7.0, 2.5), arrowprops=arrow_style)

    # Temporal Network
    ax.text(9.5, 2.5, 'Temporal Network\n(Causal CNN+GRU)\nPredict Delta(t)', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6F9', edgecolor='black', linewidth=2),
            fontsize=10, weight='bold')

    # 箭头向上
    ax.annotate('', xy=(9.5, 3.8), xytext=(9.5, 3.2), arrowprops=arrow_style)

    # 第三行：融合
    # e_ref(t)向右的箭头
    ax.annotate('', xy=(8.5, 4.5), xytext=(7.0, 4.5), arrowprops=arrow_style)

    # Bounded Fusion
    ax.text(10.5, 4.5, 'Bounded Fusion\nu(t) = e_ref(t) +\nalpha*clip(Delta)', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBE6', edgecolor='black', linewidth=2),
            fontsize=11, weight='bold')

    # 箭头
    ax.annotate('', xy=(12.2, 4.5), xytext=(11.3, 4.5), arrowprops=arrow_style)

    # Control Signal
    ax.text(13, 4.5, 'Control\nSignal\nu(t)', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6FFE6', edgecolor='black', linewidth=2),
            fontsize=11, weight='bold')

    # 标题
    ax.text(7, 5.5, 'Method Overview: Bounded Compensation for One-Frame Delay',
            ha='center', va='center', fontsize=13, weight='bold')

    # 添加说明文字
    ax.text(7, 0.5, 'Key Idea: Supplement reliable baseline with bounded prediction, not replace it',
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', linewidth=1))

    plt.tight_layout()
    plt.savefig('../paper/仪器与测量汇刊/new2/picture/fig1_method_overview.png', dpi=300, bbox_inches='tight')
    print("已保存: fig1_method_overview.png")
    plt.close()

def draw_fig3_model_architecture():
    """图3: 模型架构"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    arrow_style = dict(arrowstyle='->', lw=2, color='black')

    # Input
    ax.text(2, 7, 'Input Sequence\n[batch, seq_len=8, features=4]\n[e_ref_x, e_ref_y, d_ref_x, d_ref_y]',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E6F3FF', edgecolor='black', linewidth=2),
            fontsize=10, weight='bold')

    # 箭头
    ax.annotate('', xy=(2, 5.8), xytext=(2, 6.5), arrowprops=arrow_style)

    # Causal Conv1D Layer 1
    ax.text(2, 5.2, 'Causal Conv1D\nchannels=32, kernel=3\ndilation=1', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6E6', edgecolor='black', linewidth=2),
            fontsize=9)

    # ReLU
    ax.text(5, 5.2, 'ReLU', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9E6', edgecolor='black', linewidth=1),
            fontsize=9)

    # 箭头
    ax.annotate('', xy=(2, 4.3), xytext=(2, 4.8), arrowprops=arrow_style)

    # Causal Conv1D Layer 2
    ax.text(2, 3.7, 'Causal Conv1D\nchannels=32, kernel=3\ndilation=2', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6E6', edgecolor='black', linewidth=2),
            fontsize=9)

    # ReLU
    ax.text(5, 3.7, 'ReLU', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9E6', edgecolor='black', linewidth=1),
            fontsize=9)

    # 箭头
    ax.annotate('', xy=(2, 2.8), xytext=(2, 3.3), arrowprops=arrow_style)

    # GRU Layer
    ax.text(2, 2.2, 'GRU Layer\nhidden_dim=64\nbatch_first=True', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6FFE6', edgecolor='black', linewidth=2),
            fontsize=9, weight='bold')

    # 箭头
    ax.annotate('', xy=(2, 1.3), xytext=(2, 1.8), arrowprops=arrow_style)

    # FC Layer
    ax.text(2, 0.7, 'Fully Connected\nLinear(64 -> 2)', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6F9', edgecolor='black', linewidth=2),
            fontsize=9)

    # 箭头向右
    ax.annotate('', xy=(4, 0.7), xytext=(3, 0.7), arrowprops=arrow_style)

    # Output
    ax.text(5.5, 0.7, 'Output\n[batch, 2]\n[Delta_x, Delta_y]', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E6FFE6', edgecolor='black', linewidth=2),
            fontsize=10, weight='bold')

    # 右侧说明
    ax.text(8.5, 6.5, 'Model Details:', ha='left', va='top', fontsize=11, weight='bold')
    ax.text(8.5, 6, '• Total Parameters: ~50K', ha='left', va='top', fontsize=9)
    ax.text(8.5, 5.6, '• Inference Time: <1ms', ha='left', va='top', fontsize=9)
    ax.text(8.5, 5.2, '• Causal: No future info', ha='left', va='top', fontsize=9)
    ax.text(8.5, 4.8, '• Lightweight: Real-time', ha='left', va='top', fontsize=9)

    ax.text(8.5, 4, 'Loss Function:', ha='left', va='top', fontsize=11, weight='bold')
    ax.text(8.5, 3.5, 'L = L_pred + 0.05*L_mag', ha='left', va='top', fontsize=9)
    ax.text(8.5, 3.1, '     + 0.1*L_dir', ha='left', va='top', fontsize=9)
    ax.text(8.5, 2.6, '• L_pred: Huber loss', ha='left', va='top', fontsize=9)
    ax.text(8.5, 2.2, '• L_mag: |Delta|_1', ha='left', va='top', fontsize=9)
    ax.text(8.5, 1.8, '• L_dir: Direction penalty', ha='left', va='top', fontsize=9)

    # 标题
    ax.text(6, 7.7, 'Model Architecture: Causal CNN + GRU',
            ha='center', va='center', fontsize=13, weight='bold')

    plt.tight_layout()
    plt.savefig('../paper/仪器与测量汇刊/new2/picture/fig3_model_architecture.png', dpi=300, bbox_inches='tight')
    print("已保存: fig3_model_architecture.png")
    plt.close()

if __name__ == '__main__':
    draw_fig1_method_overview()
    draw_fig3_model_architecture()
    print("\n结构示意图生成完成")
