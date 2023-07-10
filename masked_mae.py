import torch
import torch.nn as nn


class MaskedMAELoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, y_pred, y_true):
    mask = y_true != 0
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    abs_error = torch.abs(y_pred - y_true)
    mae = abs_error.mean()
    return mae