import gym
import numpy as np
import torch
from typing import List, Dict

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