"""
生成融合策略详细示意图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10

def plot_fusion_strategy_detail():
    """融合策略详细示意图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 模拟不同场景的数据
    t = np.arange(50)

    # 场景1: 小预测值，正常融合
    e_ref1 = np.sin(t * 0.2) * 20
    delta_pred1 = np.random.randn(50) * 3  # 小预测值
    delta_clipped1 = np.clip(delta_pred1, -10, 10)
    u_fusion1 = e_ref1 + 0.4 * delta_clipped1

    ax = axes[0, 0]
    ax.plot(t, e_ref1, 'b-', linewidth=2, label='e_ref(t)')
    ax.plot(t, delta_pred1, 'g--', linewidth=1.5, alpha=0.7, label='Delta_pred(t)')
    ax.plot(t, delta_clipped1, 'orange', linewidth=1.5, label='Delta_clipped(t)')
    ax.plot(t, u_fusion1, 'r-', linewidth=2, label='u(t) = e_ref + 0.4*clip(Delta)')
    ax.set_title('(a) Normal Case: Small Prediction', fontsize=11, weight='bold')
    ax.set_ylabel('Signal Value')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(25, -35, 'Clipping inactive\nFusion = baseline + small correction',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # 场景2: 大预测值，限幅生效
    e_ref2 = np.sin(t * 0.2) * 20
    delta_pred2 = np.random.randn(50) * 15  # 大预测值
    delta_pred2[20:25] = 25  # 添加一些大的突变
    delta_clipped2 = np.clip(delta_pred2, -10, 10)
    u_fusion2 = e_ref2 + 0.4 * delta_clipped2

    ax = axes[0, 1]
    ax.plot(t, e_ref2, 'b-', linewidth=2, label='e_ref(t)')
    ax.plot(t, delta_pred2, 'g--', linewidth=1.5, alpha=0.7, label='Delta_pred(t)')
    ax.plot(t, delta_clipped2, 'orange', linewidth=1.5, label='Delta_clipped(t)')
    ax.plot(t, u_fusion2, 'r-', linewidth=2, label='u(t) = e_ref + 0.4*clip(Delta)')
    ax.set_title('(b) Clipping Active: Large Prediction', fontsize=11, weight='bold')
    ax.set_ylabel('Signal Value')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(25, -35, 'Clipping protects stability\nBounded compensation',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral'))

    # 场景3: 不同alpha值的影响
    e_ref3 = np.sin(t * 0.2) * 20
    delta3 = np.clip(np.random.randn(50) * 8, -10, 10)

    ax = axes[1, 0]
    ax.plot(t, e_ref3, 'b-', linewidth=2, label='e_ref(t)')
    ax.plot(t, e_ref3 + 0.2 * delta3, 'g-', linewidth=1.5, alpha=0.7, label='alpha=0.2 (conservative)')
    ax.plot(t, e_ref3 + 0.4 * delta3, 'r-', linewidth=1.5, label='alpha=0.4 (balanced)')
    ax.plot(t, e_ref3 + 0.6 * delta3, 'orange', linewidth=1.5, alpha=0.7, label='alpha=0.6 (aggressive)')
    ax.set_title('(c) Effect of Fusion Weight Alpha', fontsize=11, weight='bold')
    ax.set_ylabel('Signal Value')
    ax.set_xlabel('Frame')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(25, -35, 'Higher alpha: more responsive, less stable\nLower alpha: more stable, less responsive',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue'))

    # 场景4: 自适应融合
    e_ref4 = np.zeros(50)
    e_ref4[:20] = np.sin(np.arange(20) * 0.2) * 10  # 慢速运动
    e_ref4[20:35] = 10 + np.arange(15) * 3  # 快速运动
    e_ref4[35:] = 55 + np.sin(np.arange(15) * 0.3) * 5  # 恢复慢速

    # 计算运动强度
    velocity = np.zeros(50)
    velocity[1:] = np.abs(np.diff(e_ref4))
    alpha_adaptive = np.clip(0.2 + velocity / 10, 0.2, 0.6)

    delta4 = np.clip(np.random.randn(50) * 5, -10, 10)
    u_fixed = e_ref4 + 0.4 * delta4
    u_adaptive = e_ref4 + alpha_adaptive * delta4

    ax = axes[1, 1]
    ax.plot(t, e_ref4, 'b-', linewidth=2, label='e_ref(t)')
    ax.plot(t, u_fixed, 'orange', linewidth=1.5, alpha=0.7, label='Fixed Fusion (alpha=0.4)')
    ax.plot(t, u_adaptive, 'r-', linewidth=2, label='Adaptive Fusion')
    ax2 = ax.twinx()
    ax2.plot(t, alpha_adaptive, 'g--', linewidth=1.5, alpha=0.5, label='alpha(t)')
    ax2.set_ylabel('Alpha Value', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(0, 0.8)
    ax.set_title('(d) Adaptive Fusion Strategy', fontsize=11, weight='bold')
    ax.set_ylabel('Signal Value')
    ax.set_xlabel('Frame')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvspan(20, 35, alpha=0.2, color='red', label='High motion')
    ax.text(27, 10, 'High motion\nHigher alpha', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Fusion Strategy Details: u(t) = e_ref(t) + alpha * clip(Delta(t), -delta, delta)',
                 fontsize=13, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('../paper/仪器与测量汇刊/new2/picture/fig7_fusion_strategy.png', dpi=300, bbox_inches='tight')
    print("已保存: fig7_fusion_strategy.png")
    plt.close()

if __name__ == '__main__':
    plot_fusion_strategy_detail()
    print("融合策略详细图生成完成")
