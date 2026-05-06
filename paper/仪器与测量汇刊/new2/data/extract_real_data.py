"""
从真实日志中提取需要的数据
"""
import numpy as np

# 读取原始日志
print("读取真实数据...")
log_path = r'D:\kun-data\kun-code-data\反无\cpp智能控制\logs\tracker_log-new.csv'
data = np.genfromtxt(log_path,
                     delimiter=',', skip_header=1,
                     usecols=(1, 16, 17, 15),  # timestamp_ms, dx_raw, dy_raw, det_conf
                     filling_values=0.0)

# 提取列
timestamp_ms = data[:, 0]
e_raw_x = data[:, 1]
e_raw_y = data[:, 2]
confidence = data[:, 3]

# 过滤掉lost状态的数据（dx_raw和dy_raw都为0的点）
valid_mask = ~((e_raw_x == 0) & (e_raw_y == 0))
timestamp_ms = timestamp_ms[valid_mask]
e_raw_x = e_raw_x[valid_mask]
e_raw_y = e_raw_y[valid_mask]
confidence = confidence[valid_mask]

print(f"有效数据点: {len(e_raw_x)}")
print(f"数据范围: X=[{e_raw_x.min():.1f}, {e_raw_x.max():.1f}], Y=[{e_raw_y.min():.1f}, {e_raw_y.max():.1f}]")

# 转换时间戳为相对时间（秒）
timestamp_s = (timestamp_ms - timestamp_ms[0]) / 1000.0

# 保存为CSV
with open('tracking_errors_real.csv', 'w') as f:
    f.write('timestamp,e_raw_x,e_raw_y,confidence\n')
    for i in range(len(timestamp_s)):
        f.write(f'{timestamp_s[i]:.3f},{e_raw_x[i]:.6f},{e_raw_y[i]:.6f},{confidence[i]:.6f}\n')

print("真实数据已保存到 tracking_errors_real.csv")
