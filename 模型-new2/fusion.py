import numpy as np

def clip_compensation(delta, delta_max):
    """限幅补偿量"""
    return np.clip(delta, -delta_max, delta_max)

def fixed_fusion(e_ref, delta_pred, alpha=0.4, delta_max=10.0):
    """
    固定融合策略
    u(t) = e_ref(t) + α·clip(Δ̂(t), -δ, δ)
    """
    delta_clipped = clip_compensation(delta_pred, delta_max)
    u = e_ref + alpha * delta_clipped
    return u

def adaptive_fusion(e_ref, delta_pred, alpha_0=0.4, delta_0=10.0,
                   k_alpha=0.1, k_delta=0.5, alpha_min=0.2, alpha_max=0.6):
    """
    自适应融合策略
    根据运动强度调整补偿权重和限幅
    """
    # 计算运动强度
    v = np.zeros(len(e_ref))
    v[1:] = np.abs(e_ref[1:] - e_ref[:-1]).sum(axis=1)

    # 自适应权重
    alpha = np.clip(alpha_0 + k_alpha * v, alpha_min, alpha_max)

    # 自适应限幅
    delta_max = delta_0 + k_delta * v

    # 融合
    u = np.zeros_like(e_ref)
    for i in range(len(e_ref)):
        delta_clipped = clip_compensation(delta_pred[i], delta_max[i])
        u[i] = e_ref[i] + alpha[i] * delta_clipped

    return u, alpha, delta_max
