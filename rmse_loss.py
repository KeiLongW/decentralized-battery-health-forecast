import torch
import torch.nn as nn


class RMSELoss(nn.Module):
  def __init__(self):
    super(RMSELoss, self).__init__()
    self.mse = nn.MSELoss()

  def forward(self, y_pred, y_true):
    return torch.sqrt(self.mse(y_pred, y_true))