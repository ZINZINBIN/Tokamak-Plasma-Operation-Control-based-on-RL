from src.rl.ddpg import Actor, Critic
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
    
    policy_network = Actor(12 + 14, 16, 4, 64, 14)
    value_network = Critic(12 + 14, 16, 4,  64, 14)
    
    device = 'cuda:3'
    
    policy_network.to(device)
    value_network.to(device)
    
    sample_data = torch.zeros((1, 16, 26))
    sample_action = torch.zeros((1, 4, 14))
    
    print("policy network output : ", policy_network(sample_data.to(device)).size())
    print("value network output : ", value_network(sample_data.to(device), sample_action.to(device)).size())