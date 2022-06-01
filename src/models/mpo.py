'''
Maximum a Posteriori Policy optimisation
- see https://arxiv.org/abs/1806.06920
- use Expectation Maximum algorithm to solve "get the actions which maximize future rewards"
- reference code 1: https://github.com/daisatojp/mpo
- reference code 2: https://github.com/acyclics/MPO
'''
import torch 
import torch.nn as nn
import numpy as np
import gym
import random
import math
import gc
from torch import long
import matplotlib.pyplot as plt
from itertools import count
from src.models.ActorCritic import Actor, Critic
from pyvirtualdisplay import Display
from tqdm import tqdm

def bt(m:torch.Tensor):
    return m.transpose(dim0 = -2, dim1 = -1)

def btr(m:torch.Tensor):
    return m.diagonal(dim1 = -2, dim2 = -1).sum(-1)

def gaussian_kl(mu_i:torch.Tensor, mu:torch.Tensor, Ai:torch.Tensor, A:torch.Tensor):
    # calculate the decoupled KL between two gaussian distribution
    n = A.size(-1)
    mu_i = mu_i.unsqueeze(-1) # B,n,1
    mu = mu.unsqueeze(-1)

    sigma_i = Ai @ bt(Ai) # B,n,n
    sigma = A @ bt(A) # B,n,n
    sigma_i_det = sigma_i.det()
    sigma_det = sigma.det()

    sigma_i_det = torch.clamp_min(sigma_i_det, 1e-8)
    sigma_det = torch.clamp_min(sigma_det, 1e-8)

    sigma_i_inv = sigma_i.inverse() # B,n,n
    sigma_inv = sigma.inverse() # B,n,n

    inner_mu = ((mu - mu_i).transpose(-2,-1) @ sigma_i_inv @ (mu - mu_i)).squeeze() # (B,)
    inner_sigma = torch.log(sigma_det / sigma_i_det) - n + btr(sigma_inv @ sigma_i) #(B,)

    c_mu = 0.5 * torch.mean(inner_mu)
    c_sigma = 0.5 * torch.mean(inner_sigma)

    return c_mu, c_sigma, torch.mean(sigma_i_det), torch.mean(sigma_det)

def categorical_kl(p1 : torch.Tensor, p2 : torch.Tensor):
    # calculate KL between two categorical distributions
    p1 = torch.clamp_min(p1, 1e-8)
    p2 = torch.clamp_min(p2, 1e-8)
    return torch.mean((p1 * torch.log(p1 / p2)).sum(dim = -1))

if __name__ == "__main__":

    display = Display(visible = False, size = (400,300))
    display.start()

    env = gym.make("Breakout-v0").unwrapped
    env.reset()

    n_actions = env.action_space.n

    # device allocation(GPU)
    if torch.cuda.is_available():
        print("cuda available : ", torch.cuda.is_available())
        print("cuda device count : ", torch.cuda.device_count())
        device = "cuda:0"
    else:
        device = "cpu" 

    
    # training process


    print("training MPO done .... !")
    env.close()