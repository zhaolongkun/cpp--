import torch
import torch.nn as nn

class MLPBaseline(nn.Module):
    """MLP基线模型 - 将历史窗口展平"""
    def __init__(self, input_dim=4, seq_len=8, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.flatten(1)  # [batch, seq_len * features]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
