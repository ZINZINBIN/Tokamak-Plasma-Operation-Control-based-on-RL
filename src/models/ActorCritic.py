import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from typing import Optional

class Actor(nn.Module):
    def __init__(self, env, mode : Optional[str] = "continuous", hidden_dim : int = 256):
        super(Actor, self).__init__()
        self.env = env

        if mode is None:
            self.mode = "continuous"
        elif mode is not None and mode != "continuous":
            self.mode = "discrete"
        else:
            self.mode = mode

        if self.mode == "continuous":
            self.ds = env.observation_space.shape[0]
            self.da = env.action_space.shape[0]    
        else:
            self.ds = env.observation_space.shape[0]
            self.da = env.action_space.n

        self.lin1 = nn.Linear(self.ds, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        if self.mode == "continuous":
            self.mean_layer = nn.Linear(hidden_dim, self.da)
            self.cholesky_layer = nn.Linear(hidden_dim, (self.da * (self.da + 1)) // 2)
            self.out = None
        else:
            self.mean_layer = None
            self.cholesky_layer = None
            self.out = nn.Linear(hidden_dim, self.da)

    def get_continuous_output(self, state : torch.Tensor):
        B = state.size(0)
        ds = self.ds
        da = self.da
        device = state.device

        action_low = torch.from_numpy(self.env.action_space.low)[None, ...].to(device)
        action_high = torch.from_numpy(self.env.action_space.high)[None, ...].to(device)

        
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))

        mean = torch.sigmoid(self.mean_layer(x)) 
        mean = action_low + (action_high - action_low) * mean
        cholesky_vector = self.cholesky_layer(x) # B, da*(da+1)/2
        cholesky_diag_index = torch.arange(da, dtype = torch.long) + 1
        cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        cholesky_vector[:, cholesky_diag_index] = F.softplus(cholesky_vector[:, cholesky_diag_index])
        tril_indices = torch.tril_indices(row = da, col = da, offset = 0) # return lower triangular part of matrix
        cholesky = torch.zeros(size = (B, da, da), dtype = torch.float32).to(device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        
        return mean, cholesky

    def get_discrete_output(self, state : torch.Tensor):
        B = state.size(0)
        h = F.relu(self.lin1(state))
        h = F.relu(self.lin2(h))
        h = self.out(h)
        return torch.softmax(h, dim=-1)

    def get_continuous_action(self, state : torch.Tensor):
        with torch.no_grad():
            mean, cholesky = self.get_continuous_output(state[None, ...])
            action_distribution = MultivariateNormal(mean, scale_tril = cholesky)
            action = action_distribution.sample()
        return action[0]

    def get_discrete_action(self, state : torch.Tensor):
        with torch.no_grad():
            p = self.get_discrete_output(state[None, ...])
            action_distribution = Categorical(probs = p[0])
            action = action_distribution.sample()
        return action

    def action(self, state : torch.Tensor):
        if self.mode == "continuous":
            return self.get_continuous_action(state)
        else:
            return self.get_discrete_action(state)

    def forward(self, state:torch.Tensor):
        if self.mode == "continuous":
            return self.get_continuous_output(state)
        else:
            return self.get_discrete_output(state)

# Critic : MLP model
class Critic(nn.Module):
    def __init__(self, env, mode : Optional[str] = "continuous", hidden_dim : int = 256):
        super(Critic, self).__init__()
        self.env = env

        if mode is None:
            self.mode = "continuous"
        elif mode is not None and mode != "continuous":
            self.mode = "discrete"
        else:
            self.mode = mode

        if self.mode == "continuous":
            self.ds = env.observation_space.shape[0]
            self.da = env.action_space.shape[0]
        else:
            self.ds = env.observation_space.shape[0]
            self.da = env.action_space.n
        
        self.lin1 = nn.Linear(self.ds + self.da, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state : torch.Tensor, action : torch.Tensor)->torch.Tensor:
        h = torch.cat([state, action], dim = 1)
        h = F.relu(self.lin1(h))
        h = F.relu(self.lin2(h))
        q = self.lin3(h)

        return q