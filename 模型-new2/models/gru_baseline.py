import torch
import torch.nn as nn

class GRUBaseline(nn.Module):
    """纯GRU基线模型"""
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))
