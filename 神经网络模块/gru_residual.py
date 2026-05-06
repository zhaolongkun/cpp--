#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2S-LAVS-Lite: GRU 残差补偿网络

输入:
  x: [B, T, 12]
输出:
  y: [B, 2]  -> [delta_cmd_x, delta_cmd_y]
"""

import torch
import torch.nn as nn


class GRUResidualCompensator(nn.Module):
    """
    轻量 GRU 残差补偿网络
    默认配置:
      input_dim=12, hidden_size=64, num_layers=1, fc_hidden=32, output_dim=2
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        fc_hidden: int = 32,
        output_dim: int = 2,
    ):
        super().__init__()

        # PyTorch 约定: num_layers=1 时 GRU dropout 必须为 0 才生效
        gru_dropout = float(dropout) if int(num_layers) > 1 else 0.0

        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(int(hidden_size), int(fc_hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(fc_hidden), int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        return: [B, 2]
        """
        out, _ = self.gru(x)        # out: [B, T, H]
        last = out[:, -1, :]        # [B, H]
        pred = self.head(last)      # [B, 2]
        return pred

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        额外提供序列输出，用于平滑损失:
        return: [B, T, 2]
        """
        out, _ = self.gru(x)  # [B, T, H]
        b, t, h = out.shape
        out2d = out.reshape(b * t, h)
        pred2d = self.head(out2d)  # [B*T, 2]
        pred3d = pred2d.reshape(b, t, -1)
        return pred3d


def build_model_from_cfg(cfg: dict) -> GRUResidualCompensator:
    return GRUResidualCompensator(
        input_dim=int(cfg.get("input_dim", 12)),
        hidden_size=int(cfg.get("hidden_size", 64)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.1)),
        fc_hidden=int(cfg.get("fc_hidden", 32)),
        output_dim=int(cfg.get("output_dim", 2)),
    )

