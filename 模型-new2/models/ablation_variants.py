import torch
import torch.nn as nn

from models.causal_cnn_gru import CausalConv1d, CausalTemporalAttn


class NoAttnCausalCNNGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.conv_x1 = CausalConv1d(2, 32, kernel_size=3, dilation=1)
        self.conv_x2 = CausalConv1d(32, 32, kernel_size=3, dilation=2)
        self.conv_y1 = CausalConv1d(2, 32, kernel_size=3, dilation=1)
        self.conv_y2 = CausalConv1d(32, 32, kernel_size=3, dilation=2)
        self.relu = nn.ReLU()

        self.gru_x = nn.GRU(32, hidden_dim // 2, batch_first=True)
        self.gru_y = nn.GRU(32, hidden_dim // 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def _stream(self, x, conv1, conv2, gru):
        h = x.transpose(1, 2)
        h = self.relu(conv1(h))
        h = self.relu(conv2(h))
        h = h.transpose(1, 2)
        h, _ = gru(h)
        return h[:, -1, :]

    def forward(self, x):
        x_stream = x[:, :, [0, 2]]
        y_stream = x[:, :, [1, 3]]
        hx = self._stream(x_stream, self.conv_x1, self.conv_x2, self.gru_x)
        hy = self._stream(y_stream, self.conv_y1, self.conv_y2, self.gru_y)
        return self.fc(torch.cat([hx, hy], dim=-1))


class NoConvGRUAttn(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2, max_len=8):
        super().__init__()
        self.gru_x = nn.GRU(2, hidden_dim // 2, batch_first=True)
        self.gru_y = nn.GRU(2, hidden_dim // 2, batch_first=True)
        self.attn_x = CausalTemporalAttn(hidden_dim // 2, max_len=max_len)
        self.attn_y = CausalTemporalAttn(hidden_dim // 2, max_len=max_len)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def _stream(self, x, gru, attn):
        h, _ = gru(x)
        h = attn(h)
        return h[:, -1, :]

    def forward(self, x):
        x_stream = x[:, :, [0, 2]]
        y_stream = x[:, :, [1, 3]]
        hx = self._stream(x_stream, self.gru_x, self.attn_x)
        hy = self._stream(y_stream, self.gru_y, self.attn_y)
        return self.fc(torch.cat([hx, hy], dim=-1))


class SingleStreamCausalCNNGRUAttn(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2, max_len=8):
        super().__init__()
        self.conv1 = CausalConv1d(input_dim, 32, kernel_size=3, dilation=1)
        self.conv2 = CausalConv1d(32, 32, kernel_size=3, dilation=2)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(32, hidden_dim, batch_first=True)
        self.attn = CausalTemporalAttn(hidden_dim, max_len=max_len)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        h = h.transpose(1, 2)
        h, _ = self.gru(h)
        h = self.attn(h)
        return self.fc(h[:, -1, :])
