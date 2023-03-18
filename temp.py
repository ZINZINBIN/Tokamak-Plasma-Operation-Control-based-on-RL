from src.rl.ddpg import Actor, Critic
from src.rl.sac import GaussianPolicy
import torch
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

# torch device state
print("############### device setup ###################")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()
    
if __name__ == "__main__":
    
    # policy_network = Actor(12 + 14, 16, 4, 64, 14)
    # value_network = Critic(12 + 14, 16, 4,  64, 14)
    
    # device = 'cuda:3'
    
    # policy_network.to(device)
    # value_network.to(device)
    
    # sample_data = torch.zeros((1, 16, 26))
    # sample_action = torch.zeros((1, 4, 14))
    
    # print("policy network output : ", policy_network(sample_data.to(device)).size())
    # print("value network output : ", value_network(sample_data.to(device), sample_action.to(device)).size())
    
    # device = 'cuda:3'
    # model = Transformer(
    #     n_layers = 2, 
    #     n_heads = 8, 
    #     dim_feedforward = 1024, 
    #     dropout = 0.1,        
    #     RIN = True,
    #     input_0D_dim = 4,
    #     input_ctrl_dim = 14,
    #     input_seq_len = 16,
    #     output_pred_len = 4,
    #     output_0D_dim = 4,
    #     feature_dim = 128,
    #     range_info = {
    #         "a" : [1.0, 2.0],
    #         "b" : [1.0, 2.0],
    #         "c" : [1.0, 2.0],
    #         "d" : [1.0, 2.0],
    #     },
    #     noise_mean = 0,
    #     noise_std = 0.81,
    # )
    # model.summary()
    
    policy = GaussianPolicy(26, 10, 1, 64, 15, -10, 1)
    sample_data = torch.zeros((1, 10, 26))
    
    action, entropy, _ = policy.sample(sample_data)
    
    print("action : ", action.size())
    print("entropy : ", entropy.size())