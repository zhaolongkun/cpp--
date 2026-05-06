"""
生成论文所需的所有图表
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10

def plot_signal_comparison(e_raw, e_ref, start=0, length=200, save_path=None):
    """图2: 可靠基准构造示意图"""
    end = start + length
    t = np.arange(start, end)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # X轴
    ax1.plot(t, e_raw[start:end, 0], 'gray', alpha=0.6, linewidth=1, label='Raw Signal')
    ax1.plot(t, e_ref[start:end, 0], 'b', linewidth=2, label='Reliable Baseline (W=2)')
    ax1.set_ylabel('Error X (pixels)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Reliable Baseline Construction')

    # Y轴
    ax2.plot(t, e_raw[start:end, 1], 'gray', alpha=0.6, linewidth=1, label='Raw Signal')
    ax2.plot(t, e_ref[start:end, 1], 'b', linewidth=2, label='Reliable Baseline (W=2)')
    ax2.set_ylabel('Error Y (pixels)')
    ax2.set_xlabel('Frame')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()

def plot_method_comparison_bars(save_path=None):
    """图6: 主结果对比柱状图（与论文表2一致）"""
    methods = ['Raw', 'Ref Only', 'Pred Only', 'Fixed\nFusion', 'Adaptive\nFusion']
    mae_fusion = [0.54, 0.61, 0.96, 0.67, 0.63]
    jitter = [1.09, 0.62, 0.79, 0.67, 0.65]
    spike_rate = [0.082, 0.013, 0.028, 0.015, 0.012]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(methods))
    width = 0.6

    axes[0].bar(x, mae_fusion, width, color='steelblue')
    axes[0].set_ylabel('MAE Fusion')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, fontsize=9)
    axes[0].grid(True, axis='y', alpha=0.3)
    axes[0].set_title('(a) MAE Fusion')

    axes[1].bar(x, jitter, width, color='coral')
    axes[1].set_ylabel('Jitter')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, fontsize=9)
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].set_title('(b) Jitter')

    axes[2].bar(x, spike_rate, width, color='lightgreen')
    axes[2].set_ylabel('Spike Rate')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, fontsize=9)
    axes[2].grid(True, axis='y', alpha=0.3)
    axes[2].set_title('(c) Spike Rate')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    plt.close()

if __name__ == '__main__':
    # 加载数据
    data = np.loadtxt('../paper/仪器与测量汇刊/new2/data/tracking_errors_real.csv',
                      delimiter=',', skiprows=1)
    e_raw = data[:, 1:3]

    from prepare_data import causal_moving_average
    e_ref = causal_moving_average(e_raw, window_size=2)

    # 生成图2
    plot_signal_comparison(e_raw, e_ref, start=500, length=200,
                          save_path='../paper/仪器与测量汇刊/new2/picture/fig2_baseline.png')

    # 生成图6
    plot_method_comparison_bars(save_path='../paper/仪器与测量汇刊/new2/picture/fig6_comparison.png')

    print("图表生成完成")
