import torch
import torch.nn as nn


class ModelBounds:
    def __init__(
        self,
        alpha_q_min: float = 0.35,
        alpha_q_max: float = 3.00,
        alpha_r_min: float = 0.35,
        alpha_r_max: float = 4.50,
        bias_limit_px: float = 4.0,
        outlier_prob_min: float = 0.05,
        outlier_prob_max: float = 0.95,
    ) -> None:
        self.alpha_q_min = alpha_q_min
        self.alpha_q_max = alpha_q_max
        self.alpha_r_min = alpha_r_min
        self.alpha_r_max = alpha_r_max
        self.bias_limit_px = bias_limit_px
        self.outlier_prob_min = outlier_prob_min
        self.outlier_prob_max = outlier_prob_max

    def to_dict(self):
        return {
            "alpha_q_min": float(self.alpha_q_min),
            "alpha_q_max": float(self.alpha_q_max),
            "alpha_r_min": float(self.alpha_r_min),
            "alpha_r_max": float(self.alpha_r_max),
            "bias_limit_px": float(self.bias_limit_px),
            "outlier_prob_min": float(self.outlier_prob_min),
            "outlier_prob_max": float(self.outlier_prob_max),
        }


class NCEGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bounds: ModelBounds = ModelBounds(),
    ) -> None:
        super().__init__()
        do = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=do,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

        self.register_buffer("alpha_q_min", torch.tensor(float(bounds.alpha_q_min)))
        self.register_buffer("alpha_q_max", torch.tensor(float(bounds.alpha_q_max)))
        self.register_buffer("alpha_r_min", torch.tensor(float(bounds.alpha_r_min)))
        self.register_buffer("alpha_r_max", torch.tensor(float(bounds.alpha_r_max)))
        self.register_buffer("bias_limit_px", torch.tensor(float(bounds.bias_limit_px)))
        self.register_buffer("outlier_prob_min", torch.tensor(float(bounds.outlier_prob_min)))
        self.register_buffer("outlier_prob_max", torch.tensor(float(bounds.outlier_prob_max)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        y, _ = self.gru(x)
        h = y[:, -1, :]
        raw = self.head(h)

        bias = self.bias_limit_px * torch.tanh(raw[:, 0:2])

        aq = torch.sigmoid(raw[:, 2:3])
        alpha_q = self.alpha_q_min + (self.alpha_q_max - self.alpha_q_min) * aq

        ar = torch.sigmoid(raw[:, 3:4])
        alpha_r = self.alpha_r_min + (self.alpha_r_max - self.alpha_r_min) * ar

        op = torch.sigmoid(raw[:, 4:5])
        outlier_prob = self.outlier_prob_min + (self.outlier_prob_max - self.outlier_prob_min) * op

        return torch.cat([bias, alpha_q, alpha_r, outlier_prob], dim=1)
