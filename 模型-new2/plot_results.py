import matplotlib.pyplot as plt
import numpy as np
import json

def plot_method_comparison(results_path, save_path=None):
    """绘制方法对比柱状图"""
    with open(results_path) as f:
        results = json.load(f)

    methods = list(results.keys())
    metrics = ['mae_fusion', 'jitter', 'spike_rate']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in methods]
        axes[i].bar(methods, values)
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Results plotting module ready")
