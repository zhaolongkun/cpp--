import torch
import torch.nn as nn

class GRUBaseline(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

class MLPBaseline(nn.Module):
    def __init__(self, input_dim=4, seq_len=8, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
