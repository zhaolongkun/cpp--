import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class CausalTemporalAttn(nn.Module):
    def __init__(self, dim, max_len=8):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, v)


class CausalCNNGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.conv_x1 = CausalConv1d(2, 32, kernel_size=3, dilation=1)
        self.conv_x2 = CausalConv1d(32, 32, kernel_size=3, dilation=2)
        self.conv_y1 = CausalConv1d(2, 32, kernel_size=3, dilation=1)
        self.conv_y2 = CausalConv1d(32, 32, kernel_size=3, dilation=2)
        self.relu = nn.ReLU()

        self.gru_x = nn.GRU(32, hidden_dim // 2, batch_first=True)
        self.gru_y = nn.GRU(32, hidden_dim // 2, batch_first=True)

        self.attn_x = CausalTemporalAttn(hidden_dim // 2, max_len=8)
        self.attn_y = CausalTemporalAttn(hidden_dim // 2, max_len=8)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def _stream(self, x, conv1, conv2, gru, attn):
        h = x.transpose(1, 2)
        h = self.relu(conv1(h))
        h = self.relu(conv2(h))
        h = h.transpose(1, 2)
        h, _ = gru(h)
        h = attn(h)
        return h[:, -1, :]

    def forward(self, x):
        x_stream = x[:, :, [0, 2]]
        y_stream = x[:, :, [1, 3]]
        hx = self._stream(x_stream, self.conv_x1, self.conv_x2, self.gru_x, self.attn_x)
        hy = self._stream(y_stream, self.conv_y1, self.conv_y2, self.gru_y, self.attn_y)
        return self.fc(torch.cat([hx, hy], dim=-1))
