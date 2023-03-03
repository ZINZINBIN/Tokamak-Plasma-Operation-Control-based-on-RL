from src.rl.env import NeuralEnv
from src.nn_env.model import TStransformer
from src.rl.rewards import RewardSender
from src.rl.utility import InitGenerator
from src.rl.train import train_ddpg
from src.rl.ddpg import Actor, Critic
from src.rl.buffer import ReplayBuffer
import torch
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

def parsing():
    parser = argparse.ArgumentParser(description="training ddpg algorithms for tokamak plasma control")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "DDPG")
    parser.add_argument("--save_dir", type = str, default = "./result")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)


    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--num_episode", type = int, default = 256)
    parser.add_argument("--seq_len", type = int, default = 21)
    parser.add_argument("--pred_len", type = int, default = 1)
    
    parser.add_argument("--t_init", type = float, default = 1.5)
    
    args = vars(parser.parse_args())

    return args

# torch device state
print("############### device setup ###################")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()
    
if __name__ == "__main__":
    
    # parsing
    args = parsing()
    tag = args['tag']
    save_dir = args['save_dir']
    batch_size = args['batch_size']
    num_episode = args['num_episode']
    seq_len = args['seq_len']
    pred_len = args['pred_len']
    t_init = args['t_init']
        
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
    
    # 0D parameters
    cols_0D = [
        '\\q0', '\\q95', '\\ipmhd', '\\kappa', 
        '\\tritop', '\\tribot','\\betap','\\betan',
        '\\li', '\\WTOT_DLM03','\\ne_inter01',
    ]
    
    # else diagnostics
    cols_diag = [
        '\\ne_tci01', '\\ne_tci02', '\\ne_tci03', '\\ne_tci04', '\\ne_tci05',
    ]

    # control value / parameter
    cols_control = [
        '\\nb11_pnb','\\nb12_pnb','\\nb13_pnb',
        '\\RC01', '\\RC02', '\\RC03',
        '\\VCM01', '\\VCM02', '\\VCM03',
        '\\EC2_PWR', '\\EC3_PWR', 
        '\\ECSEC2TZRTN', '\\ECSEC3TZRTN',
        '\\LV01'
    ]

    # predictor
    model = TStransformer(
        n_features = len(cols_0D + cols_control), 
        feature_dims = 128, 
        max_len = seq_len, 
        n_layers = 4, 
        n_heads = 8, 
        dim_feedforward = 512, 
        dropout = 0.25, 
        mlp_dim = 64, 
        pred_len = pred_len,
        output_dim = len(cols_0D)
    )

    model.to(device)
    model.load_state_dict(torch.load("./weights/TStransformer_best.pt"))

    # reward 
    targets_dict = {
        "\\betap" : 3.0,
        "\\betan" : 4.0,
        "\\q95" : 4.0,
        "\\q0" : 1.0
    }

    # reward
    reward_sender = RewardSender(targets_dict, total_cols = cols_0D + cols_control)
    
    # environment
    env = NeuralEnv(predictor=model, device = device, reward_sender = reward_sender, seq_len = seq_len, pred_len = pred_len)
    
    # initial state generator
    # step 1. real data loaded
    df = pd.read_csv("./dataset/KSTAR_Disruption_ts_data_extend.csv").reset_index()
    df_disruption = pd.read_csv("./dataset/KSTAR_Disruption_Shot_List.csv", encoding='euc-kr').reset_index()
    
    # nan interpolation
    df.interpolate(method = 'linear', limit_direction = 'forward')
    
    # columns for use
    ts_cols = cols_0D + cols_control

    # float type
    for col in ts_cols:
        df[col] = df[col].astype(np.float32)
        
    df[cols_control] = df[cols_control].fillna(0)

    df[cols_0D] = df[cols_0D].fillna(method = 'ffill')

    scaler = RobustScaler()
    df[ts_cols] = scaler.fit_transform(df[ts_cols].values)

    init_generator = InitGenerator(df, t_init, cols_0D, cols_control, seq_len, True, None)
    
    # Replay Buffer
    memory = ReplayBuffer(capacity=100000)
    
    # Actor and Critic Network
    input_dim = len(cols_0D)
    mlp_dim = 64
    n_actions = len(cols_control)
    
    policy_network = Actor(input_dim, seq_len, mlp_dim, n_actions)
    target_policy_network = Actor(input_dim, seq_len, mlp_dim, n_actions)
    
    value_network = Critic(input_dim, seq_len, mlp_dim, n_actions)
    target_value_network = Critic(input_dim, seq_len, mlp_dim, n_actions)
    
    policy_network.to(device)
    target_policy_network.to(device)

    value_network.to(device)
    target_value_network.to(device)
    
    lr = 1e-3
    gamma = 0.995
    min_value = -1.0
    max_value = 1.0
    tau = 0.01
    verbose = 4
    
    value_optimizer = torch.optim.AdamW(value_network.parameters(), lr = lr)
    policy_optimizer = torch.optim.AdamW(policy_network.parameters(), lr = lr)

    value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'mean')
    
    # optimization
    print("############### DDPG Training Process ###################")
    save_best = os.path.join("./weights/", "{}_best.pt".format(tag))
    save_last = os.path.join("./weights/", "{}_last.pt".format(tag))
    
    episode_durations, episode_rewards = train_ddpg(
        env, 
        init_generator,
        memory,
        policy_network,
        value_network,
        target_policy_network,
        target_value_network,
        policy_optimizer,
        value_optimizer,
        value_loss_fn,
        batch_size,
        gamma,
        device,
        min_value,
        max_value,
        tau,
        num_episode,
        verbose,
        save_best,
        save_last
    )
    
    plt.subplot(1,2,1)
    plt.plot(range(1, num_episode + 1), episode_durations, 'r--', label = 'episode duration')
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, num_episode + 1), episode_rewards, 'b--', label = 'episode reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.savefig("./result/DDPG_episode_reward.png")