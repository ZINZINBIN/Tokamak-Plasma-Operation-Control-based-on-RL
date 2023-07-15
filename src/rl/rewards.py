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
    def __init__(self, targets_dict : Dict, total_cols : List, targets_weight : Optional[List] = None):
        self.targets_dict = targets_dict
        self.targets_weight = targets_weight
        self.targets_cols = list(targets_dict.keys())
        self.targets_value = list(targets_dict.values())
        self.total_cols = total_cols
        self._extract_target_index(total_cols)

    def __call__(self, new_state : torch.Tensor):
        return self._compute_reward(new_state)
    
    def _compute_reward(self, state:Union[torch.Tensor, np.ndarray]):
        reward = 0
        for i, (target_value, idx) in enumerate(zip(self.targets_value, self.target_cols_indices)):
            state_per_idx = state[:,:,idx]
            target_per_idx = torch.ones(state_per_idx.size()) * target_value
            
            if self.targets_weight is not None:
                weight = self.targets_weight[i]
                reward += weight * compute_reward(state_per_idx, target_per_idx)
            else:
                reward += compute_reward(state_per_idx, target_per_idx)
                
        return reward
    
    # compute reward as a vector for Multi-objective Reinforcement Learning
    def compute_vectorized_reward(self, state : Union[torch.Tensor, np.ndarray]):
        reward = torch.zeros((len(self.targets_weight),))
        for i, (target_value, idx) in enumerate(zip(self.targets_value, self.target_cols_indices)):
            state_per_idx = state[:,:,idx]
            target_per_idx = torch.ones(state_per_idx.size()) * target_value
            reward[i] += compute_reward(state_per_idx, target_per_idx)
        return reward
    
    def _extract_target_index(self, total_cols : List):
        indices = []
        for col in self.targets_cols:
            indices.append(total_cols.index(col))
        
        self.target_cols_indices = indices
        
    def update_target_weight(self, target_weight : List):
        self.targets_weight = target_weight