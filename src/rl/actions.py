import gym
import numpy as np
import torch
from typing import List, Dict

class NormalizedActions(gym.ActionWrapper):
    def reverse_action(self, action:torch.Tensor):
        
        low_bound = self.action_space['low']
        upper_bound = self.action_space['upper']
        
        low_bound_ = torch.Tensor(low_bound).repeat(1,action.size()[1],1)
        upper_bound_ = torch.Tensor(upper_bound).repeat(1,action.size()[1],1)
        
        action = low_bound_ + (action + 1.0) * 0.5 * (upper_bound_ - low_bound_)
        
        return action

    def action(self, action:torch.Tensor):
        low_bound   = self.action_space['low']
        upper_bound = self.action_space['upper']
        
        low_bound_ = torch.Tensor(low_bound).repeat(1,action.size()[1],1)
        upper_bound_ = torch.Tensor(upper_bound).repeat(1,action.size()[1],1)
        
        action = 2 * (action - (low_bound_ + upper_bound_)/2) / (upper_bound_ - low_bound_)
        
        return action