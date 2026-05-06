import numpy as np

np.random.seed(42)

# 生成模拟的视觉伺服跟踪误差数据
n_samples = 2000
t = np.arange(n_samples)

# 生成基础轨迹（模拟目标运动）
freq1, freq2 = 0.01, 0.015
base_x = 50 * np.sin(2 * np.pi * freq1 * t) + 20 * np.sin(2 * np.pi * freq2 * t)
base_y = 40 * np.cos(2 * np.pi * freq1 * t) + 15 * np.cos(2 * np.pi * freq2 * t)

# 添加噪声（模拟检测不确定性）
noise_x = np.random.normal(0, 8, n_samples)
noise_y = np.random.normal(0, 8, n_samples)

# 添加偶发突变（模拟检测失败或遮挡）
spike_indices = np.random.choice(n_samples, size=20, replace=False)
noise_x[spike_indices] += np.random.normal(0, 30, 20)
noise_y[spike_indices] += np.random.normal(0, 30, 20)

# 原始误差信号
e_raw_x = base_x + noise_x
e_raw_y = base_y + noise_y

# 模拟检测置信度
confidence = np.random.uniform(0.7, 0.99, n_samples)
confidence[spike_indices] = np.random.uniform(0.3, 0.6, 20)

# 保存为CSV
timestamp = t * 0.033  # 假设30fps
with open('tracking_errors.csv', 'w') as f:
    f.write('timestamp,e_raw_x,e_raw_y,confidence\n')
    for i in range(n_samples):
        f.write(f'{timestamp[i]:.3f},{e_raw_x[i]:.6f},{e_raw_y[i]:.6f},{confidence[i]:.6f}\n')

print(f"生成了 {n_samples} 个样本的模拟数据")
print(f"数据范围: X=[{e_raw_x.min():.1f}, {e_raw_x.max():.1f}], Y=[{e_raw_y.min():.1f}, {e_raw_y.max():.1f}]")
