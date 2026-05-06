"""
完整实验流程：使用真实数据训练和评估
"""
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare_data import prepare_sequences, split_data, normalize_data, causal_moving_average
import json

# 1. 加载真实数据
print("=" * 50)
print("步骤1: 加载真实数据")
print("=" * 50)
data_path = '../paper/仪器与测量汇刊/new2/data/tracking_errors_real.csv'
data = np.loadtxt(data_path, delimiter=',', skiprows=1)
e_raw = data[:, 1:3]  # e_raw_x, e_raw_y
print(f"数据形状: {e_raw.shape}")
print(f"数据范围: X=[{e_raw[:, 0].min():.1f}, {e_raw[:, 0].max():.1f}], Y=[{e_raw[:, 1].min():.1f}, {e_raw[:, 1].max():.1f}]")

# 2. 准备训练数据（W=2）
print("\n" + "=" * 50)
print("步骤2: 准备训练序列 (W=2)")
print("=" * 50)
X, Y, e_ref = prepare_sequences(e_raw, window_size=2, history_len=8)
print(f"输入形状: {X.shape}, 标签形状: {Y.shape}")

# 3. 切分数据
data_dict = split_data(X, Y)
print(f"训练集: {data_dict['X_train'].shape}")
print(f"验证集: {data_dict['X_val'].shape}")
print(f"测试集: {data_dict['X_test'].shape}")

# 4. 标准化
data_dict, stats = normalize_data(data_dict)

# 5. 保存
np.savez('processed_data_w2.npz', **data_dict)
with open('stats_w2.json', 'w') as f:
    json.dump({k: v.tolist() for k, v in stats.items()}, f)
print("W=2 数据已保存")

# 6. 准备W=3数据
print("\n" + "=" * 50)
print("步骤3: 准备训练序列 (W=3)")
print("=" * 50)
X3, Y3, e_ref3 = prepare_sequences(e_raw, window_size=3, history_len=8)
data_dict3 = split_data(X3, Y3)
data_dict3, stats3 = normalize_data(data_dict3)
np.savez('processed_data_w3.npz', **data_dict3)
print("W=3 数据已保存")

print("\n" + "=" * 50)
print("数据准备完成！")
print("=" * 50)
print("下一步: 运行 python train.py 训练模型")
