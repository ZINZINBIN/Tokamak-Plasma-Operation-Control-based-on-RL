''' Customized reward functions

    Reference
    -   https://towardsdatascience.com/how-to-design-reinforcement-learning-reward-function-for-a-lunar-lander-562a24c393f6
    
'''
from typing import Optional, Union, Literal, List, Dict
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_TARGETS = {
    "\\betap" : 3.0,
    "\\betan" : 4.0,
    "\\q95" : 4.0,
    "\\q0" : 1.0
}

def compute_reward(inputs : torch.Tensor, targets : torch.Tensor, a : float = 1e-3):
    loss = F.mse_loss(inputs, targets, size_average=True, reduce = True)
    return 1 / (loss + a)

class RewardSender:
    def __init__(self, targets_dict : Dict, total_cols : List):
        self.targets_dict = targets_dict
        self.targets_cols = targets_dict.keys()
        self.targets_value = targets_dict.values()
        self._extract_target_index(total_cols)

    def __call__(self, new_state : torch.Tensor):
        return self._compute_reward(new_state)
    
    def _compute_reward(self, state:Union[torch.Tensor, np.ndarray]):
        reward = 0
        device = state.device
        
        for idx_dict, idx in zip(self.targets_value, self.target_cols_indices):
            state_per_idx = state[:,:,idx]
            target_per_idx = torch.ones(state_per_idx.size()).to(device) * self.targets_value[idx_dict]
            reward += compute_reward(state_per_idx, target_per_idx)
        return reward
    
    def _extract_target_index(self, total_cols : List):
        indices = []
        for col in self.targets_cols:
            indices.append(total_cols.index(col))
        
        self.target_cols_indices = indices