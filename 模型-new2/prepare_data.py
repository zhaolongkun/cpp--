import numpy as np
from pathlib import Path
import json

def causal_moving_average(signal, window_size=2):
    """因果滑动均值 - 只使用当前和过去的数据"""
    result = np.zeros_like(signal)
    for i in range(len(signal)):
        start_idx = max(0, i - window_size + 1)
        result[i] = np.mean(signal[start_idx:i+1], axis=0)
    return result

def compute_diff(signal):
    """计算一阶差分"""
    diff = np.zeros_like(signal)
    diff[1:] = signal[1:] - signal[:-1]
    return diff

def prepare_sequences(e_raw, window_size=2, history_len=8):
    """
    准备训练序列
    Args:
        e_raw: 原始误差信号 [N, 2] (x, y)
        window_size: 滑动窗口大小
        history_len: 历史窗口长度
    Returns:
        X: 输入特征 [N-history_len, history_len, feature_dim]
        Y: 标签 [N-history_len, 2]
    """
    # 构造可靠基准
    e_ref = causal_moving_average(e_raw, window_size)

    # 计算差分
    d_ref = compute_diff(e_ref)

    # 构造特征: [e_ref_x, e_ref_y, d_ref_x, d_ref_y]
    features = np.concatenate([e_ref, d_ref], axis=1)

    # 构造序列样本
    X, Y = [], []
    for i in range(history_len, len(features)):
        # 输入: 历史窗口
        X.append(features[i-history_len:i])
        # 标签: 下一帧可靠基准的增量
        Y.append(e_ref[i] - e_ref[i-1])

    return np.array(X), np.array(Y), e_ref

def split_data(X, Y, train_ratio=0.7, val_ratio=0.15):
    """按时间顺序切分数据"""
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        'X_train': X[:train_end],
        'Y_train': Y[:train_end],
        'X_val': X[train_end:val_end],
        'Y_val': Y[train_end:val_end],
        'X_test': X[val_end:],
        'Y_test': Y[val_end:]
    }

def normalize_data(data_dict):
    """标准化数据 - 只使用训练集统计量"""
    X_train = data_dict['X_train']

    # 计算训练集统计量
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1)) + 1e-8

    # 标准化所有集合
    for key in ['X_train', 'X_val', 'X_test']:
        data_dict[key] = (data_dict[key] - mean) / std

    return data_dict, {'mean': mean, 'std': std}

if __name__ == '__main__':
    # 示例: 加载数据并准备
    # e_raw = np.load('raw_error.npy')  # [N, 2]
    # X, Y, e_ref = prepare_sequences(e_raw, window_size=2, history_len=8)
    # data_dict = split_data(X, Y)
    # data_dict, stats = normalize_data(data_dict)
    # np.savez('processed_data.npz', **data_dict)
    # json.dump({k: v.tolist() for k, v in stats.items()}, open('stats.json', 'w'))
    print("Data preparation module ready")
