import torch
import torch.nn as nn

class DFwrapper(nn.Module):
    def __init__(self, model : nn.Module, input_dim : int, scale : float = 0.1):
        super().__init__()
        self.scale = scale
        self.input_dims = input_dim
        self.input_0D_seq_len = model.input_0D_seq_len
        self.output_0D_pred_len = model.output_0D_pred_len
        self.model = model
        self.forgetting = None
        self.eta = nn.Parameter(torch.ones(input_dim), requires_grad=False)
    
    def generate_forgetting_matrix(self, x : torch.Tensor):
        seq = torch.arange(x.size()[1]).flip(0).expand(x.size()[0], x.size()[2], x.size()[1]).permute(0,2,1)
        self.forgetting = torch.exp(seq.to(x.device) * self.scale * self.eta.to(x.device) * (-1))
        
    def forward(self, x_0D : torch.Tensor, x_ctrl : torch.Tensor):
        if self.forgetting is None or self.forgetting.size() != x_0D.size():
            self.generate_forgetting_matrix(x_0D)
        x_0D *= self.forgetting.to(x_0D.device)
        x = self.model(x_0D, x_ctrl)
        return x
    
    def summary(self):
        self.model.summary()