import torch
import torch.nn as nn


class MaskedRMSELoss(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, y_pred, y_true):
    mask = y_true != 0
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    mse = ((y_pred - y_true) ** 2).mean()
    return torch.sqrt(mse)