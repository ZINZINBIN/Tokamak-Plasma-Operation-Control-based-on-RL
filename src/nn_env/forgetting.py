''' Wrapper for differentiable forgetting algorithm
    Model adaptation to distribution shift via weighted empirical risk minimization using differentiable Bi-Level optimization
    
    Paper: Time series prediction under distribution shift using differentiable forgetting
    Training process : model parameter optimization with fixed hyperparameter
    Validation process : Hyperprameter optimization 
    
    For validation process, we have to update the hyperparameter using sum of one-step prediction loss
    However, in training process, we have to compute weighted empirical risk minimization
'''
import torch
import torch.nn as nn
from typing import Union, Optional
from src.nn_env.transformer import Transformer

class DFwrapper:
    def __init__(self, model : Union[nn.Module, Transformer], scale : float = 0.1, trainable : bool = False):
        super().__init__()
        self.scale = scale
        self.input_dim = model.input_0D_dim
        self.input_0D_seq_len = model.input_0D_seq_len
        self.output_0D_pred_len = model.output_0D_pred_len
        self.model = model
        
        self.exp_eta = None
        self.exp_theta = None
        
        self.trainable = trainable
        
        if trainable: 
            self.eta = nn.Parameter(torch.ones(model.input_0D_dim), requires_grad=True)
            self.theta = nn.Parameter(torch.ones(model.input_ctrl_dim), requires_grad=True)
        else:
            self.eta = nn.Parameter(torch.ones(model.input_0D_dim), requires_grad=False)
            self.theta = nn.Parameter(torch.ones(model.input_ctrl_dim), requires_grad=False)
    
    def generate_forgetting_matrix(self, x : torch.Tensor, y : Optional[torch.Tensor] = None):
        seq = torch.arange(x.size()[1]).flip(0).expand(x.size()[0], x.size()[2], x.size()[1]).permute(0,2,1)
        self.exp_eta = torch.exp(seq.to(x.device) * self.scale * self.eta.to(x.device) * (-1))
        
        if y is not None:
            seq = torch.arange(y.size()[1]).flip(0).expand(y.size()[0], y.size()[2], y.size()[1]).permute(0,2,1)
            self.exp_theta = torch.exp(seq.to(y.device) * self.scale * self.theta.to(y.device) * (-1))
        
    def forward(self, x_0D : torch.Tensor, x_ctrl : torch.Tensor):
        if self.exp_eta is None or self.exp_eta.size() != x_0D.size():
            self.generate_forgetting_matrix(x_0D, x_ctrl)
            
        x_0D *= self.exp_eta.to(x_0D.device)
        x_ctrl *= self.exp_theta.to(x_ctrl.device)
        
        x = self.model(x_0D, x_ctrl)
        return x
    
    def __call__(self, x_0D : torch.Tensor, x_ctrl : torch.Tensor):
        return self.forward(x_0D, x_ctrl)
        
    def __getattr__(self, name):
        return getattr(self.model, name)