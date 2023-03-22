''' Customized reward functions

    Reference
    -   https://towardsdatascience.com/how-to-design-reinforcement-learning-reward-function-for-a-lunar-lander-562a24c393f6 
'''
from typing import Optional, Union, Literal, List, Dict
import numpy as np
import torch
import torch.nn.functional as F

def compute_reward(inputs : torch.Tensor, targets : torch.Tensor, residue : float = 1e-3, scale : float = 1.0):
    diff = F.mse_loss(inputs, targets, reduction = 'mean')
    reward = F.tanh(1 / (diff + targets.mean() * residue)) * scale
    return reward

class RewardSender:
    def __init__(self, targets_dict : Dict, total_cols : List):
        self.targets_dict = targets_dict
        self.targets_cols = list(targets_dict.keys())
        self.targets_value = list(targets_dict.values())
        self.total_cols = total_cols
        self._extract_target_index(total_cols)

    def __call__(self, new_state : torch.Tensor):
        return self._compute_reward(new_state)
    
    def _compute_reward(self, state:Union[torch.Tensor, np.ndarray]):
        reward = 0
        for target_value, idx in zip(self.targets_value, self.target_cols_indices):
            state_per_idx = state[:,:,idx]
            target_per_idx = torch.ones(state_per_idx.size()) * target_value
            reward += compute_reward(state_per_idx, target_per_idx)
        return reward
    
    def _extract_target_index(self, total_cols : List):
        indices = []
        for col in self.targets_cols:
            indices.append(total_cols.index(col))
        
        self.target_cols_indices = indices