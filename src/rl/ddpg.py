# ddpg for continous action space
# Actor-critic based model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from typing import Optional
from src.rl.buffer import Transition, ReplayBuffer

# Encoder : plasma state -> hidden vector
class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        pass
    def forward(self, x : torch.Tensor):
        pass

# Actor
class Actor(nn.Module):
    def __init__(self, *args, **kwargs):
        self.encoder = Encoder(*args, **kwargs)
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions

        linear_input_dim =  self.encoder.linear_input_dim

        self.mlp = nn.Sequential(
            nn.Linear(linear_input_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, n_actions)
        )
        
    def forward(self, x : torch.Tensor):
        x = self.encoder(x)
        x = torch.tanh(self.mlp(x))
        return x
    
    def get_action(self, x : torch.Tensor):
        x = x.unsqueeze(0).to(x.device)
        action = self.forward(x)
        return action.detach().cpu().numpy()[0,0]
    
# Critic
class Critic(nn.Module):
    def __init__(self, *args, **kwargs):
        self.encoder = Encoder(*args, **kwargs)
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions

        linear_input_dim =  self.encoder.linear_input_dim

        self.mlp = nn.Sequential(
            nn.Linear(linear_input_dim + n_actions, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, n_actions)
        )
        
    def forward(self, x:torch.Tensor, action : torch.Tensor):
        x = self.encoder(x)
        x = torch.cat([x, action], dim = 1)
        x = self.critic(x)
        return x
