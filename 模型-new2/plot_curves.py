import matplotlib.pyplot as plt
import numpy as np

def plot_time_series(e_raw, e_ref, u, e_ref_future, start=0, length=200, save_path=None):
    """绘制时序曲线对比"""
    end = start + length
    t = np.arange(start, end)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # X轴
    ax1.plot(t, e_raw[start:end, 0], 'gray', alpha=0.5, label='Raw')
    ax1.plot(t, e_ref[start:end, 0], 'b', label='Ref')
    ax1.plot(t, u[start:end, 0], 'r', label='Fusion')
    ax1.plot(t, e_ref_future[start:end, 0], 'g--', label='Ref(t+1)')
    ax1.set_ylabel('Error X')
    ax1.legend()
    ax1.grid(True)

    # Y轴
    ax2.plot(t, e_raw[start:end, 1], 'gray', alpha=0.5, label='Raw')
    ax2.plot(t, e_ref[start:end, 1], 'b', label='Ref')
    ax2.plot(t, u[start:end, 1], 'r', label='Fusion')
    ax2.plot(t, e_ref_future[start:end, 1], 'g--', label='Ref(t+1)')
    ax2.set_ylabel('Error Y')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Plotting module ready")
