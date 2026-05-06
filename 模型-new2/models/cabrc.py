"""
CABRC: Causal Anchor-guided Bounded Residual Compensation
面向视觉伺服控制的因果锚定有界残差补偿方法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.pad, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.pad > 0:
            x = x[:, :, :-self.pad]
        return x


class CausalSelfAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.scale = dim ** -0.5

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, v)


class SingleStreamEncoder(nn.Module):
    """单轴因果特征提取器"""
    def __init__(self, hidden=32):
        super().__init__()
        self.conv1 = CausalConv1d(2, hidden, kernel_size=3, dilation=1)
        self.conv2 = CausalConv1d(hidden, hidden, kernel_size=3, dilation=2)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.attn = CausalSelfAttn(hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, T, 2]
        h = self.relu(self.conv1(x.transpose(1, 2))).transpose(1, 2)
        h = self.relu(self.conv2(h.transpose(1, 2))).transpose(1, 2)
        h, _ = self.gru(h)
        h = self.attn(h)
        return h[:, -1, :]  # [B, hidden]


class CABRC(nn.Module):
    """
    CABRC: Causal Anchor-guided Bounded Residual Compensation

    创新点：
    C1 - 因果锚点参考（外部传入 e_anchor）
    C2 - 可靠性感知残差门控（g(t) ∈ [0,1]）
    C3 - 稳定约束有界融合（clip + gate）

    前向传播：
        h(t) = DualStreamEncoder(x_{t-T+1:t})
        Δ̂(t) = ResidualHead(h(t))
        g(t)  = GateHead(h(t))  ∈ [0,1]
        u(t)  = e_anchor(t) + g(t) · clip(Δ̂(t), -δ, δ)
    """
    def __init__(self, hidden=32, delta_max=10.0,
                 ablate_gate=False, ablate_clip=False, ablate_anchor=False):
        super().__init__()
        self.delta_max = delta_max
        self.ablate_gate = ablate_gate      # 消融C2：g固定=1
        self.ablate_clip = ablate_clip      # 消融C3：去掉限幅
        self.ablate_anchor = ablate_anchor  # 消融C1：直接预测绝对值

        self.enc_x = SingleStreamEncoder(hidden)
        self.enc_y = SingleStreamEncoder(hidden)

        feat_dim = hidden * 2  # 64

        # 残差预测头（C1的载体）
        self.residual_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 2)
        )

        # 可靠性门控头（C2的载体）
        self.gate_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 2),
            nn.Sigmoid()
        )

    def encode(self, x):
        # x: [B, T, 4] = [e_ref_x, e_ref_y, d_ref_x, d_ref_y]
        x_stream = x[:, :, [0, 2]]  # [B, T, 2]
        y_stream = x[:, :, [1, 3]]  # [B, T, 2]
        hx = self.enc_x(x_stream)
        hy = self.enc_y(y_stream)
        return torch.cat([hx, hy], dim=-1)  # [B, 64]

    def forward(self, x, e_anchor):
        """
        x:        [B, T, 4]  历史特征序列
        e_anchor: [B, 2]     当前帧因果锚点 e_anchor(t)
        返回:
            u:    [B, 2]     控制输出
            delta:[B, 2]     预测残差
            gate: [B, 2]     可靠性门控值
        """
        h = self.encode(x)                          # [B, 64]
        delta = self.residual_head(h)               # [B, 2]
        gate = self.gate_head(h)                    # [B, 2] ∈ [0,1]

        # 消融开关
        if self.ablate_gate:
            gate = torch.ones_like(gate)
        if self.ablate_clip:
            delta_bounded = delta
        else:
            delta_bounded = torch.clamp(delta, -self.delta_max, self.delta_max)

        if self.ablate_anchor:
            # 消融C1：直接输出绝对预测值
            u = delta
        else:
            # C1+C2+C3：锚点 + 门控有界残差
            u = e_anchor + gate * delta_bounded

        return u, delta, gate
