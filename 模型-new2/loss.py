import torch
import torch.nn as nn

class CompensationLoss(nn.Module):
    """组合损失函数: Huber + 幅值正则 + 方向一致性"""
    def __init__(self, lambda_pred=1.0, lambda_mag=0.05, lambda_dir=0.1):
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_mag = lambda_mag
        self.lambda_dir = lambda_dir
        self.huber = nn.HuberLoss()

    def forward(self, pred, target):
        # L_pred: Huber loss
        loss_pred = self.huber(pred, target)

        # L_mag: 幅值正则
        loss_mag = torch.mean(torch.abs(pred))

        # L_dir: 方向一致性 (惩罚预测方向与目标相反的情况)
        dot_product = torch.sum(pred * target, dim=1)
        loss_dir = torch.mean(torch.clamp(-dot_product, min=0))

        # 总损失
        total_loss = (self.lambda_pred * loss_pred +
                     self.lambda_mag * loss_mag +
                     self.lambda_dir * loss_dir)

        return total_loss, {
            'loss_pred': loss_pred.item(),
            'loss_mag': loss_mag.item(),
            'loss_dir': loss_dir.item()
        }
