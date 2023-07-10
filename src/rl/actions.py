import gym
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

gauss_noise = torch.distributions.Normal(0,1)

def add_noise_to_action(action : torch.Tensor):
    noise = gauss_noise.sample(sample_shape=action.size()).to(action.device)
    return action + noise

# Smoothness-inducing regularizer for smooth control
def smoothness_inducing_regularizer(
    loss_fn : nn.Module, 
    mu : torch.Tensor,
    next_mu : torch.Tensor,
    near_mu : torch.Tensor,
    lamda_temporal_smoothness : float, 
    lamda_spatial_smoothness : float,
    ):
    
    Lt = lamda_temporal_smoothness * loss_fn(mu, next_mu)
    Ls = lamda_spatial_smoothness * loss_fn(mu, near_mu)
    
    return Lt + Ls

class NormalizedActions(gym.ActionWrapper):
    def action(self, action:torch.Tensor):
        
        low_bound = self.action_space['low']
        upper_bound = self.action_space['upper']
        
        low_bound_ = torch.Tensor(low_bound).repeat(1,action.size()[1],1)
        upper_bound_ = torch.Tensor(upper_bound).repeat(1,action.size()[1],1)
        
        action = low_bound_ + (action + 1.0) * 0.5 * (upper_bound_ - low_bound_)
        
        return action

    def reverse_action(self, action:torch.Tensor):
        low_bound   = self.action_space['low']
        upper_bound = self.action_space['upper']
        
        low_bound_ = torch.Tensor(low_bound).repeat(1,action.size()[1],1)
        upper_bound_ = torch.Tensor(upper_bound).repeat(1,action.size()[1],1)
        
        action = 2 * (action - (low_bound_ + upper_bound_)/2) / (upper_bound_ - low_bound_)
        
        return action
    
class ClippingActions(gym.ActionWrapper):
    
    def action(self, action : torch.Tensor):
        low_bound = self.action_space['rate-low']
        upper_bound = self.action_space['rate-upper']
        
        low_bound = torch.Tensor(low_bound).view(1,1,-1)
        upper_bound = torch.Tensor(upper_bound).view(1,1,-1)
        
        action_prev = self.get_action()[:,-1,:].unsqueeze(1)
        action = torch.clip(action, action_prev - low_bound, action_prev + upper_bound)
        
        return action
        
    def reverse_action(self, action : torch.Tensor):
        return action