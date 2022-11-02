import numpy as np
import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction = 'none')
    
    def forward(self, pred : torch.Tensor, target : torch.Tensor):
        loss = torch.sqrt(self.mse_loss(pred, target).mean(dim = 0)) / target.abs().mean(dim = 0)
        loss = loss.view(-1,).sum()
        return loss