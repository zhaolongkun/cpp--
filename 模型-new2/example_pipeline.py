"""
端到端示例：从数据加载到模型训练和评估
"""
import numpy as np
import sys
sys.path.append('..')

from prepare_data import prepare_sequences, split_data, normalize_data

# 1. 加载生成的数据
print("加载数据...")
data = np.loadtxt('../paper/仪器与测量汇刊/new2/data/tracking_errors.csv',
                  delimiter=',', skiprows=1)
e_raw = data[:, 1:3]  # e_raw_x, e_raw_y

print(f"数据形状: {e_raw.shape}")

# 2. 准备训练数据
print("\n准备训练序列...")
X, Y, e_ref = prepare_sequences(e_raw, window_size=2, history_len=8)
print(f"输入形状: {X.shape}, 标签形状: {Y.shape}")

# 3. 切分数据
print("\n切分数据集...")
data_dict = split_data(X, Y)
print(f"训练集: {data_dict['X_train'].shape}")
print(f"验证集: {data_dict['X_val'].shape}")
print(f"测试集: {data_dict['X_test'].shape}")

# 4. 标准化
print("\n标准化数据...")
data_dict, stats = normalize_data(data_dict)

# 5. 保存处理后的数据
print("\n保存处理后的数据...")
np.savez('processed_data.npz', **data_dict)
print("数据准备完成！")

print("\n下一步:")
print("1. 运行 'python train.py' 训练模型")
print("2. 运行 'python evaluate.py' 评估模型")
