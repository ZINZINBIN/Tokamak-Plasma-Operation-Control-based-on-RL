# ddpg for continous action space
# Actor-critic based model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Encoder : plasma state -> hidden vector
class Encoder(nn.Module):
    def __init__(self, input_dim : int, seq_len : int, conv_dim : int = 32, conv_kernel : int = 3, conv_stride : int = 2, conv_padding : int = 1):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = input_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels = conv_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
        )
        
        self.linear_input_dim = self.compute_conv1d_output_dim(self.compute_conv1d_output_dim(seq_len, conv_kernel, conv_stride, conv_padding, 1), conv_kernel, conv_stride, conv_padding, 1) * conv_dim

    def forward(self, x : torch.Tensor):
        
        if len(x.size()) < 3:
            x = x.unsqueeze(0)
        
        if x.size()[2] != self.seq_len:
            x = x.permute(0,2,1)
            
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        return x
    
    def compute_conv1d_output_dim(self, input_dim : int, kernel_size : int = 3, stride : int = 1, padding : int = 1, dilation : int = 1):
        return int((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

# Actor
class Actor(nn.Module):
    def __init__(self, input_dim : int, seq_len : int, mlp_dim : int, n_actions : int):
        super(Actor, self).__init__()
        self.encoder = Encoder(input_dim, seq_len)
        self.n_actions = n_actions
        self.mlp_dim = mlp_dim
        linear_input_dim =  self.encoder.linear_input_dim

        self.mlp = nn.Sequential(
            nn.Linear(linear_input_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, n_actions)
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
    def __init__(self, input_dim : int, seq_len : int, mlp_dim : int, n_actions : int):
        super(Critic, self).__init__()
        self.encoder = Encoder(input_dim, seq_len)
        self.n_actions = n_actions
        self.mlp_dim = mlp_dim
        linear_input_dim =  self.encoder.linear_input_dim

        self.mlp = nn.Sequential(
            nn.Linear(linear_input_dim + n_actions, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1)
        )
        
    def forward(self, x:torch.Tensor, action : torch.Tensor):
        x = self.encoder(x)
        x = torch.cat([x, action], dim = 1)
        x = self.mlp(x)
        return x
    
# Ornstein-Uhlenbeck Process
# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise:
    def __init__(self, action_space, mu : float = 0, theta : float = 0.15, max_sigma : float = 0.3, min_sigma : float = 0.3, decay_period : int = 100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.state = np.empty_like(np.ones(self.action_dim))
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def evolve_state(self):
        x = self.state
        dx  = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t = 0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)