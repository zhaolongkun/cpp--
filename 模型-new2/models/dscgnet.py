"""
DSCGNet: Direct Stability-Constrained Control Generation Network
端到端一帧超前控制信号预测，状态转移参数化输出
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
        return torch.bmm(F.softmax(attn, dim=-1), v)


class MotionStreamEncoder(nn.Module):
    """单轴运动流编码器，输入 [f, d, dd] 3维特征"""
    def __init__(self, hidden=32, use_attention=True):
        super().__init__()
        self.conv1 = CausalConv1d(3, hidden, kernel_size=3, dilation=1)
        self.conv2 = CausalConv1d(hidden, hidden, kernel_size=3, dilation=2)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.attn = CausalSelfAttn(hidden) if use_attention else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, T, 3]
        h = self.relu(self.conv1(x.transpose(1, 2))).transpose(1, 2)
        h = self.relu(self.conv2(h.transpose(1, 2))).transpose(1, 2)
        h, _ = self.gru(h)
        if self.attn is not None:
            h = self.attn(h)
        return h[:, -1, :]  # [B, hidden]


class QualityBranchEncoder(nn.Module):
    """质量特征分支，输入 [conf, log(area+1), miss_flag, dt] 4维"""
    def __init__(self, hidden=16):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(4, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, x):
        # x: [B, T, 4]
        h = self.proj(x)
        h, _ = self.gru(h)
        return h[:, -1, :]  # [B, hidden]


class DSCGNet(nn.Module):
    """
    Direct Stability-Constrained Control Generation Network

    输入：历史T帧特征序列
      base模式(6维): [f_x, f_y, d_x, d_y, dd_x, dd_y]
      quality模式(10维): 上述6维 + [conf, log(area+1), miss_flag, dt]

    输出头（状态转移参数化）：
      z_t  = delta_head(h)          原始转移量
      g_t  = gate_head(h)           转移门控 ∈ [0,1]
      Δû_t = g_t ⊙ (r_max ⊙ tanh(z_t))
      û_{t+1} = e_f(t) + Δû_t      直接预测下一帧控制信号
    """
    def __init__(self, hidden=32, use_attention=True,
                 use_quality=False,
                 rmax_x=20.0, rmax_y=20.0):
        super().__init__()
        self.use_quality = use_quality
        self.register_buffer('r_max', torch.tensor([rmax_x, rmax_y]))

        self.enc_x = MotionStreamEncoder(hidden, use_attention)
        self.enc_y = MotionStreamEncoder(hidden, use_attention)

        feat_dim = hidden * 2  # 64
        if use_quality:
            self.enc_q = QualityBranchEncoder(16)
            feat_dim += 16  # 80

        self.delta_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 2)
        )
        self.gate_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 2),
            nn.Sigmoid()
        )

    def forward(self, x, quality=None):
        """
        x:       [B, T, 6]  base特征
        quality: [B, T, 4]  可选质量特征
        """
        # 拆分双流：X流=[f_x,d_x,dd_x], Y流=[f_y,d_y,dd_y]
        x_stream = x[:, :, [0, 2, 4]]
        y_stream = x[:, :, [1, 3, 5]]

        hx = self.enc_x(x_stream)  # [B, 32]
        hy = self.enc_y(y_stream)  # [B, 32]
        h = torch.cat([hx, hy], dim=-1)  # [B, 64]

        if self.use_quality and quality is not None:
            hq = self.enc_q(quality)
            h = torch.cat([h, hq], dim=-1)  # [B, 80]

        z_t = self.delta_head(h)           # 原始转移量 [B, 2]
        g_t = self.gate_head(h)            # 门控 [B, 2]

        # 状态转移参数化输出
        delta_pred = g_t * (self.r_max * torch.tanh(z_t))  # Δû_t
        e_f_t = x[:, -1, :2]                               # 当前状态 [f_x, f_y]
        u_pred = e_f_t + delta_pred                         # û_{t+1}

        return {
            "u_pred": u_pred,
            "delta_pred": delta_pred,
            "gate": g_t,
            "raw_delta": z_t,
            "e_f_t": e_f_t,
        }


# ── Baselines ──────────────────────────────────────────────────────────────

class LastValueBaseline(nn.Module):
    """û_{t+1} = e_f(t)"""
    def forward(self, x, **kwargs):
        e_f_t = x[:, -1, :2]
        return {"u_pred": e_f_t, "delta_pred": torch.zeros_like(e_f_t),
                "gate": torch.ones_like(e_f_t), "raw_delta": torch.zeros_like(e_f_t)}


class LinearExtrapolationBaseline(nn.Module):
    """û_{t+1} = 2*e_f(t) - e_f(t-1)"""
    def forward(self, x, **kwargs):
        e_f_t = x[:, -1, :2]
        e_f_tm1 = x[:, -2, :2]
        u_pred = 2 * e_f_t - e_f_tm1
        return {"u_pred": u_pred, "delta_pred": u_pred - e_f_t,
                "gate": torch.ones_like(e_f_t), "raw_delta": u_pred - e_f_t}


class GRUDirectBaseline(nn.Module):
    """GRU直接输出绝对控制信号"""
    def __init__(self, input_dim=6, hidden=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x, **kwargs):
        h, _ = self.gru(x)
        u_pred = self.fc(h[:, -1, :])
        e_f_t = x[:, -1, :2]
        return {"u_pred": u_pred, "delta_pred": u_pred - e_f_t,
                "gate": torch.ones_like(e_f_t), "raw_delta": u_pred - e_f_t}
