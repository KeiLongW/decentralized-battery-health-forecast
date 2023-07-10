import torch
import torch.nn as nn


class MaskedMAPELoss(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, y_pred, y_true):
    mask = y_true != 0
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    abs_error = torch.abs(y_pred - y_true)
    rel_error = abs_error / y_true
    mape = rel_error.mean() * 100
    return mape